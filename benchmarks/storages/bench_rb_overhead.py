"""ReplayBuffer bookkeeping overhead over a low-dim, in-RAM torch DictStorage (storage cost is negligible here).

  append / append_wrap            per-step append, filling / wrapping a buffer, us per step
  append_segmeta_wrap             the same with maintain_segment_metadata=True, wrapping, us per step
  extend64_wrap(_segmeta)         extend(64) into a wrapping buffer, us per call
  get_segments_ms                 get_segments() with segment metadata
  rand256_us                      get_at() of 256 random indices
  mp_len_ns / mp_get_int_us       lock-free reads of a multiprocessing=True buffer (count/offset snapshot)
  mp_sampler_rand256_us           get_at(256) in 4 sampler processes while the main process appends

  python benchmarks/storages/bench_rb_overhead.py [--quick] [--src CHECKOUT] [--out results.jsonl]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common  # noqa: E402

KEYS = ["joint_pos", "ee_pos", "action", "reward", "terminated"]


def _space():
    import torch
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    from unienv_interface.space import BinarySpace, BoxSpace, DictSpace
    box = lambda low, high, shape: BoxSpace(TB, low, high, torch.float32, shape=shape, device="cpu")
    return DictSpace(TB, {
        "joint_pos": box(-10.0, 10.0, (7,)), "ee_pos": box(-10.0, 10.0, (3,)), "action": box(-1.0, 1.0, (7,)),
        "reward": box(-10.0, 10.0, ()), "terminated": BinarySpace(TB, shape=(), dtype=torch.bool, device="cpu"),
    }, device="cpu")


def _make_rb(capacity, **kwargs):
    from unienv_data.replay_buffer import ReplayBuffer
    from unienv_data.storages.dict_storage import DictStorage
    from unienv_data.storages.pytorch import PytorchTensorStorage
    return ReplayBuffer.create(DictStorage, _space(), storage_cls_map={key: PytorchTensorStorage for key in KEYS},
                               capacity=capacity, is_memmap=False, **kwargs)


def _batch(n, seed=0):
    import torch
    generator = torch.Generator().manual_seed(seed)
    return {
        "joint_pos": torch.rand(n, 7, generator=generator), "ee_pos": torch.rand(n, 3, generator=generator),
        "action": torch.rand(n, 7, generator=generator), "reward": torch.rand(n, generator=generator),
        "terminated": torch.rand(n, generator=generator) < 0.01,
    }


def _append_us(rb, steps, episode=200):
    start = time.perf_counter()
    for i, step in enumerate(steps):
        rb.append(step)
        if (i + 1) % episode == 0:
            rb.mark_segment_end()
    rb.mark_segment_end()
    return (time.perf_counter() - start) / len(steps) * 1e6


def _sampler(rb, n, queue):
    import numpy as np
    import torch
    rng = np.random.default_rng(os.getpid())
    start = time.perf_counter()
    for _ in range(n):
        rb.get_at(torch.from_numpy(rng.integers(0, max(len(rb), 1), 256)))
    queue.put((time.perf_counter() - start) / n * 1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bench_common.add_common_args(parser)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    bench_common.import_src(args.src)
    import multiprocessing as mp
    import numpy as np
    import torch
    torch.set_num_threads(1)

    n = 2000 if args.quick else 20_000
    capacity, wrap_capacity = (20_000, 500) if args.quick else (100_000, 5000)
    data = _batch(n)
    steps = [{key: value[i] for key, value in data.items()} for i in range(n)]
    batches = [{key: value[s:s + 64] for key, value in data.items()} for s in range(0, n, 64)]
    result = {"bench": "rb_overhead", "n": n}

    result["append_us"] = _append_us(_make_rb(capacity), steps)
    result["append_wrap_us"] = _append_us(_make_rb(wrap_capacity), steps)
    rb = _make_rb(wrap_capacity, maintain_segment_metadata=True)
    result["append_segmeta_wrap_us"] = _append_us(rb, steps)
    start = time.perf_counter()
    for _ in range(20):
        rb.get_segments()
    result["get_segments_ms"] = (time.perf_counter() - start) / 20 * 1e3
    for label, kwargs in (("extend64_wrap_us", {}), ("extend64_wrap_segmeta_us", dict(maintain_segment_metadata=True))):
        rb = _make_rb(wrap_capacity, **kwargs)
        start = time.perf_counter()
        for b in batches:
            rb.extend(b)
        result[label] = (time.perf_counter() - start) / len(batches) * 1e6

    rb = _make_rb(capacity)
    rb.extend(_batch(capacity, seed=1))
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for _ in range(500):
        rb.get_at(torch.from_numpy(rng.integers(0, capacity, 256)))
    result["rand256_us"] = (time.perf_counter() - start) / 500 * 1e6

    rb = _make_rb(capacity, multiprocessing=True)
    rb.extend(_batch(capacity - 10, seed=2))
    start = time.perf_counter()
    for _ in range(100_000):
        len(rb)
    result["mp_len_ns"] = (time.perf_counter() - start) / 100_000 * 1e9
    start = time.perf_counter()
    for _ in range(20_000):
        rb.get_at(5)
    result["mp_get_int_us"] = (time.perf_counter() - start) / 20_000 * 1e6
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    samplers = [ctx.Process(target=_sampler, args=(rb, 200 if args.quick else 1000, queue)) for _ in range(4)]
    for sampler in samplers:
        sampler.start()
    result["mp_append_with_samplers_us"] = _append_us(rb, steps[:5000])
    result["mp_sampler_rand256_us"] = float(np.mean([queue.get() for _ in samplers]))
    for sampler in samplers:
        sampler.join()
    bench_common.emit(result, args.out)


if __name__ == "__main__":
    main()
