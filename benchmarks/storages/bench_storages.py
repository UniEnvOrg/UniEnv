"""Throughput of every storage backend behind a ReplayBuffer: the backend x operation table.

Each backend stores the same transition (7-dof joints, 3-d end-effector, 7-d action, reward, terminated and a
uint8 camera image, 128x128x3 by default) and runs:

  append      per-step `append()` in episodes of 100 (environment collection), us per step
  extend64    `extend()` with 64 samples per call, samples / s
  extend1024  `extend()` with 1024 samples per call until the buffer is full, samples / s
  rand256     `get_at()` of 256 uniformly random indices of the full buffer (training sampling), ms per call
  seq256      `get_at()` of consecutive 256-sample slices, samples / s
  dump_close  `dumps()` + `close()`, s
  disk_MB     size on disk after dumping
  ro_rand256  like rand256, after reopening the dumped buffer read-only, ms per call

  python benchmarks/storages/bench_storages.py                 # full table (capacity 10k, ~10+ min)
  python benchmarks/storages/bench_storages.py --quick         # small sizes, ~1 min
  python benchmarks/storages/bench_storages.py --backends hdf5,parquet --out after.jsonl
  python benchmarks/storages/bench_storages.py --compare before.jsonl after.jsonl
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common  # noqa: E402

BACKENDS = [
    "torch-ram", "torch-memmap", "flattened-memmap", "hdf5", "hdf5-gzip", "hdf5-per-step-lzf", "parquet",
    "dict-video+hdf5", "dict-png+hdf5", "dict-jpeg+hdf5", "dict-npz+hdf5",
]
#: Backends whose per-step append is so slow that they append fewer steps.
SLOW_APPEND = {"hdf5-gzip": 0.2, "dict-png+hdf5": 0.3, "dict-npz+hdf5": 0.3, "hdf5": 0.3}
COLUMNS = [
    ("append_us", "append (us/step)"), ("extend64_tps", "extend64 (samples/s)"),
    ("extend1024_tps", "extend1024 (samples/s)"), ("rand256_ms", "rand256 (ms)"),
    ("seq256_tps", "seq256 (samples/s)"), ("dump_close_s", "dump+close (s)"),
    ("disk_MB", "disk (MB)"), ("ro_rand256_ms", "ro rand256 (ms)"),
]


def build(args):
    import numpy as np
    import torch
    from unienv_interface.backends.numpy import NumpyComputeBackend as NB
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    from unienv_interface.space import BinarySpace, BoxSpace, DictSpace
    from unienv_data.replay_buffer import ReplayBuffer
    from unienv_data.storages.dict_storage import DictStorage
    from unienv_data.storages.flattened import FlattenedStorage
    from unienv_data.storages.hdf5 import HDF5Storage
    from unienv_data.storages.image_storage import ImageStorage
    from unienv_data.storages.npz_storage import NPZStorage
    from unienv_data.storages.parquet import ParquetStorage
    from unienv_data.storages.pytorch import PytorchTensorStorage

    size = args.img

    def space(backend):
        if backend is TB:
            kw = dict(device="cpu")
            box = lambda low, high, dtype, shape: BoxSpace(TB, low, high, dtype, shape=shape, **kw)
            return DictSpace(TB, {
                "joint_pos": box(-10.0, 10.0, torch.float32, (7,)), "ee_pos": box(-10.0, 10.0, torch.float32, (3,)),
                "image": box(0, 255, torch.uint8, (3, size, size)), "action": box(-1.0, 1.0, torch.float32, (7,)),
                "reward": box(-10.0, 10.0, torch.float32, ()),
                "terminated": BinarySpace(TB, shape=(), dtype=torch.bool, **kw),
            }, device="cpu")
        return DictSpace(NB, {
            "joint_pos": BoxSpace(NB, -10.0, 10.0, np.float32, shape=(7,)),
            "ee_pos": BoxSpace(NB, -10.0, 10.0, np.float32, shape=(3,)),
            "image": BoxSpace(NB, 0, 255, np.uint8, shape=(size, size, 3)),
            "action": BoxSpace(NB, -1.0, 1.0, np.float32, shape=(7,)),
            "reward": BoxSpace(NB, -10.0, 10.0, np.float32, shape=()),
            "terminated": BinarySpace(NB, shape=(), dtype=np.bool_),
        })

    keys = ["joint_pos", "ee_pos", "image", "action", "reward", "terminated"]
    per_key = lambda cls: {key: cls for key in keys}
    video_kwargs = dict(codec="libx264", file_ext="mp4", hardware_acceleration=None, decode_backend="pyav",
                        encoder_options={"crf": "23", "g": "15"})

    def video(path, cap):
        from unienv_data.storages.video_storage import VideoStorage
        return ReplayBuffer.create(DictStorage, space(NB), storage_cls_map={"image": VideoStorage, "*": HDF5Storage},
                                   cache_path=path, capacity=cap, key_kwargs={"image": video_kwargs})

    def image_dict(cls, **kwargs):
        return lambda path, cap: ReplayBuffer.create(
            DictStorage, space(NB), storage_cls_map={"image": cls, "*": HDF5Storage}, cache_path=path, capacity=cap,
            key_kwargs={"image": kwargs} if kwargs else {})

    return {
        "torch-ram": (TB, lambda path, cap: ReplayBuffer.create(
            DictStorage, space(TB), storage_cls_map=per_key(PytorchTensorStorage), cache_path=path, capacity=cap,
            is_memmap=False), dict(is_memmap=False)),
        "torch-memmap": (TB, lambda path, cap: ReplayBuffer.create(
            DictStorage, space(TB), storage_cls_map=per_key(PytorchTensorStorage), cache_path=path, capacity=cap,
            is_memmap=True), dict(is_memmap=True)),
        "flattened-memmap": (TB, lambda path, cap: ReplayBuffer.create(
            FlattenedStorage, space(TB), inner_storage_cls=PytorchTensorStorage, cache_path=path, capacity=cap,
            is_memmap=True), dict(is_memmap=True)),
        "hdf5": (NB, lambda path, cap: ReplayBuffer.create(HDF5Storage, space(NB), cache_path=path, capacity=cap), {}),
        "hdf5-gzip": (NB, lambda path, cap: ReplayBuffer.create(
            HDF5Storage, space(NB), cache_path=path, capacity=cap, chunks=True, compression="gzip"), {}),
        "hdf5-per-step-lzf": (NB, lambda path, cap: ReplayBuffer.create(
            HDF5Storage, space(NB), cache_path=path, capacity=cap, chunks={"image": (1,), "*": (1024,)},
            compression="lzf"), {}),
        "parquet": (NB, lambda path, cap: ReplayBuffer.create(ParquetStorage, space(NB), cache_path=path, capacity=cap), {}),
        "dict-video+hdf5": (NB, video, dict(key_kwargs={"image": dict(hardware_acceleration=None, decode_backend="pyav",
                                                                        codec="libx264")})),
        "dict-png+hdf5": (NB, image_dict(ImageStorage, format="PNG"), {}),
        "dict-jpeg+hdf5": (NB, image_dict(ImageStorage, format="JPEG", quality=90), {}),
        "dict-npz+hdf5": (NB, image_dict(NPZStorage), {}),
    }, space


def run_backend(name, args, data_dir):
    import numpy as np
    import torch
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    from unienv_data.replay_buffer import ReplayBuffer

    backends, space_fn = build(args)
    backend, create, load_kwargs = backends[name]
    space = space_fn(backend)
    capacity = args.capacity
    frames = bench_common.make_frames(256, args.img, chw=backend is TB)
    rng = np.random.default_rng(0)

    def batch(start, n):
        out = {}
        for key, sub in space.spaces.items():
            if key == "image":
                value = frames[np.arange(start, start + n) % len(frames)]
            elif key == "terminated":
                value = rng.random(n) < 0.01
            else:
                value = rng.uniform(-1, 1, (n, *sub.shape)).astype(np.float32)
            out[key] = torch.from_numpy(np.ascontiguousarray(value)) if backend is TB else value
        return out

    def close(rb):
        storage = rb.storage
        rb.close()
        if storage is not None:
            storage.close()
        gc.collect()

    result = {"backend": name, "capacity": capacity, "img": args.img}

    # Per-step append in episodes of 100
    path = data_dir.fresh(f"{name}-append")
    rb = create(path, capacity)
    n_append = max(100, int(args.append_steps * SLOW_APPEND.get(name, 1.0)))
    n_append = min(n_append, capacity)
    data = batch(0, n_append)
    steps = [{key: value[i] for key, value in data.items()} for i in range(n_append)]
    start = time.perf_counter()
    for i, step in enumerate(steps):
        rb.append(step)
        if (i + 1) % 100 == 0:
            rb.mark_segment_end()
    rb.mark_segment_end()
    result["append_us"] = (time.perf_counter() - start) / n_append * 1e6
    close(rb)
    data_dir.remove(f"{name}-append")

    # extend(64) for a while, then extend(1024) until full
    for chunk in (64, 1024):
        path = data_dir.fresh(f"{name}-extend{chunk}")
        rb = create(path, capacity)
        total = capacity if chunk == 1024 else min(capacity, args.extend64_samples)
        batches = [batch(s, chunk) for s in range(0, total, chunk)]
        start = time.perf_counter()
        for b in batches:
            rb.extend(b)
        result[f"extend{chunk}_tps"] = len(batches) * chunk / (time.perf_counter() - start)
        del batches
        if chunk == 64:
            close(rb)
            data_dir.remove(f"{name}-extend64")

    def random_index():
        index = rng.integers(0, len(rb), 256)
        return torch.from_numpy(index) if backend is TB else index

    rb.get_at(random_index())  # warm up
    start = time.perf_counter()
    for _ in range(args.iters):
        rb.get_at(random_index())
    result["rand256_ms"] = (time.perf_counter() - start) / args.iters * 1e3

    start, n = time.perf_counter(), 0
    for s in range(0, min(len(rb), 256 * 20), 256):
        rb.get_at(slice(s, s + 256))
        n += 256
    result["seq256_tps"] = n / (time.perf_counter() - start)

    start = time.perf_counter()
    rb.dumps(path)
    close(rb)
    result["dump_close_s"] = time.perf_counter() - start
    result["disk_MB"] = bench_common.disk_bytes(path) / 1e6

    rb = ReplayBuffer.load_from(path, backend=backend, device="cpu" if backend is TB else None, **load_kwargs)
    rb.get_at(random_index())
    start = time.perf_counter()
    for _ in range(args.iters):
        rb.get_at(random_index())
    result["ro_rand256_ms"] = (time.perf_counter() - start) / args.iters * 1e3
    close(rb)
    data_dir.remove(f"{name}-extend1024")
    return result


def print_table(results):
    print("| backend | " + " | ".join(title for _, title in COLUMNS) + " |")
    print("|---|" + "---|" * len(COLUMNS))
    for result in results:
        cells = [bench_common.format_number(result.get(key, "-")) for key, _ in COLUMNS]
        print(f"| {result['backend']} | " + " | ".join(cells) + " |")


def print_comparison(before, after):
    """Markdown table of `before -> after` per backend and column."""
    before = {result["backend"]: result for result in before if "error" not in result}
    print("| backend | " + " | ".join(title for _, title in COLUMNS) + " |")
    print("|---|" + "---|" * len(COLUMNS))
    for result in after:
        old = before.get(result["backend"])
        if old is None or "error" in result:
            continue
        cells = []
        for key, _ in COLUMNS:
            if key not in result or key not in old:
                cells.append("-")
                continue
            cells.append(f"{bench_common.format_number(old[key])} → {bench_common.format_number(result[key])}")
        print(f"| {result['backend']} | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bench_common.add_common_args(parser)
    parser.add_argument("--backends", default=",".join(BACKENDS), help="comma-separated subset of: " + ", ".join(BACKENDS))
    parser.add_argument("--quick", action="store_true", help="small sizes for a fast smoke run")
    parser.add_argument("--capacity", type=int, default=None, help="buffer capacity (default 10000, --quick 1024)")
    parser.add_argument("--img", type=int, default=128, help="image height / width (default 128)")
    parser.add_argument("--iters", type=int, default=None, help="random-read iterations (default 20, --quick 3)")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="print a before -> after table from two --out files and exit")
    args = parser.parse_args()
    if args.compare:
        print_comparison(bench_common.load_results(args.compare[0]), bench_common.load_results(args.compare[1]))
        return
    args.capacity = args.capacity or (1024 if args.quick else 10_000)
    args.iters = args.iters or (3 if args.quick else 20)
    args.append_steps = 200 if args.quick else 3000
    args.extend64_samples = 640 if args.quick else 5120

    bench_common.import_src(args.src)
    import torch
    torch.set_num_threads(args.threads)
    data_dir = bench_common.DataDir(args.data_dir)
    results = []
    try:
        for name in args.backends.split(","):
            try:
                result = run_backend(name, args, data_dir)
            except Exception as error:  # keep going, report the failure in the table
                import traceback
                traceback.print_exc()
                result = {"backend": name, "error": repr(error)}
            bench_common.emit(result, args.out)
            results.append(result)
    finally:
        data_dir.close()
    print()
    print_table([result for result in results if "error" not in result])


if __name__ == "__main__":
    main()
