"""Training-loop data pipelines over an in-memory ReplayBuffer (batch views, samplers, JPEG decode).

    python benchmarks/batches/bench_pipeline.py [--quick] [--only NAME,...] [--ref main]

Times ``StepSampler.sample()`` over frame/slice-stacked views, their ``get_column`` views and
``CombinedBatch``es, ``MultiprocessingSampler`` (fork) with and without ``shared_memory``, and
batch JPEG decoding by ``ImageDecompressTransformation``. Cases an older revision does not
support are reported as n/a.
"""
import argparse
import inspect
import multiprocessing as mp
import sys

import _common

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_common.add_common_args(parser)
parser.add_argument("--n", type=int, default=None, help="transitions in the replay buffer (default 20000, quick 2000)")
parser.add_argument("--batch", type=int, default=None, help="batch size (default 256, quick 64)")
parser.add_argument("--workers", type=int, default=4, help="MultiprocessingSampler workers")
parser.add_argument("--mp-batch", type=int, default=None, help="MultiprocessingSampler batch size (default: --batch)")
ARGV = sys.argv[1:]
ARGS = parser.parse_args(ARGV)

EP_LEN = 200


def build(args):
    import torch
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    from unienv_interface.space import BinarySpace, BoxSpace, DictSpace
    from unienv_data import ReplayBuffer
    from unienv_data.storages.dict_storage import DictStorage
    from unienv_data.storages.pytorch import PytorchTensorStorage

    img = (3, 64, 64) if args.quick else (3, 128, 128)
    space = DictSpace(TB, {
        "obs": DictSpace(TB, {
            "state": BoxSpace(TB, -1e6, 1e6, torch.float32, shape=(10,)),
            "image": BoxSpace(TB, 0, 255, torch.uint8, shape=img),
        }),
        "action": BoxSpace(TB, -1, 1, torch.float32, shape=(7,)),
        "reward": BoxSpace(TB, -1e6, 1e6, torch.float32, shape=()),
        "episode_id": BoxSpace(TB, 0, 10 ** 9, torch.int64, shape=()),
        "episode_end": BinarySpace(TB, shape=()),
    })
    keys = ["obs/state", "obs/image", "action", "reward", "episode_id", "episode_end"]

    def make_rb(n, start=0, chunk=5000):
        rb = ReplayBuffer.create(
            DictStorage, space, {k: PytorchTensorStorage for k in keys}, capacity=n, is_memmap=False,
        )
        g = torch.Generator().manual_seed(start)
        for s in range(0, n, chunk):
            m = min(chunk, n - s)
            i = torch.arange(start + s, start + s + m)
            rb.extend({
                "obs": {"state": torch.randn(m, 10, generator=g),
                        "image": torch.randint(0, 256, (m, *img), dtype=torch.uint8, generator=g)},
                "action": torch.rand(m, 7, generator=g) * 2 - 1,
                "reward": torch.randn(m, generator=g),
                "episode_id": i // EP_LEN,
                "episode_end": (i % EP_LEN) == EP_LEN - 1,
            })
        return rb
    return make_rb


def run(rec):
    import torch
    import numpy as np
    from unienv_data.batches import CombinedBatch, FrameStackedBatch, SliceStackedBatch
    from unienv_data.samplers import MultiprocessingSampler, StepSampler

    args = ARGS
    n = args.n or (2000 if args.quick else 20000)
    bs = args.batch or (64 if args.quick else 256)
    n_iter = 3 if args.quick else 20
    make_rb = build(args)
    rb = make_rb(n)
    halves = [make_rb(n // 2, start=0), make_rb(n // 2, start=n // 2)]
    quarters = [make_rb(n // 4, start=i * (n // 4)) for i in range(4)]
    ep_id = SliceStackedBatch.get_valid_mask_function_with_episodeid_key("episode_id")
    ep_end = SliceStackedBatch.get_valid_mask_function_with_episode_end_key("episode_end")
    h16 = torch.arange(0, 16)

    def touch(x):
        """Read one byte per 4 KiB page of every tensor, as a consumer would (pages of a batch
        received through shared memory are only mapped in on first access)."""
        if isinstance(x, dict):
            return sum(touch(v) for v in x.values())
        return int(x.reshape(-1).view(torch.uint8)[::4096].sum())

    def sampler(view, batch=bs):
        return StepSampler(view, batch, seed=0).sample

    # --- worker processes ---
    # Run first: before the fork deadlock fix, fork-ed workers deadlocked once the parent had
    # run a parallel torch op (e.g. sampled a large batch in-process). The watchdog turns
    # such a hang into an error.
    import faulthandler
    from unienv_data.samplers import multiprocessing_sampler as mps
    if not hasattr(mps, "_limit_worker_intraop_threads"):
        # Revisions before that fix need its workaround to run at all (this only limits the
        # workers' intra-op threads, as the fix does; the IPC being measured is unchanged).
        loop = mps.worker_loop
        def worker_loop(*a, **kw):
            torch.set_num_threads(1)
            return loop(*a, **kw)
        mps.worker_loop = worker_loop
    has_shm = "shared_memory" in inspect.signature(MultiprocessingSampler.__init__).parameters
    seconds = 1.0 if args.quick else 3.0
    mp_bs = args.mp_batch or bs
    for label, view in (("replay buffer", rb), ("framestack k=2", FrameStackedBatch(rb, 1, get_valid_mask_function=ep_id))):
        for shm in (False, True):
            if shm and not has_shm:
                continue
            kwargs = {"shared_memory": True} if shm else {}
            s = MultiprocessingSampler(StepSampler(view, mp_bs, seed=0), n_workers=args.workers, n_buffers=8,
                                       ctx=mp.get_context("fork"), **kwargs)
            try:
                name = f"MultiprocessingSampler {label} W={args.workers}{' shared_memory' if shm else ''} B={mp_bs}"
                faulthandler.dump_traceback_later(120, exit=True)
                rec.time_for(name, lambda: touch(s.sample()), seconds)
                faulthandler.cancel_dump_traceback_later()
            finally:
                s.close()
    rec.time(f"in-process StepSampler replay buffer B={bs}", sampler(rb), n_iter)
    rec.time(f"in-process StepSampler framestack k=2 B={bs}",
             sampler(FrameStackedBatch(rb, 1, get_valid_mask_function=ep_id)), n_iter)

    # --- in-process views ---
    rec.time(f"framestack k=4 (episode_id mask) B={bs}",
             sampler(FrameStackedBatch(rb, 3, get_valid_mask_function=ep_id)), n_iter)
    rec.time(f"slicestack h=16 (episode_id mask) B={bs}",
             sampler(SliceStackedBatch(rb, h16, get_valid_mask_function=ep_id)), max(1, n_iter // 4))
    rec.time(f"slicestack h=16 (episode_id mask) get_column(action) B={bs}",
             sampler(SliceStackedBatch(rb, h16, get_valid_mask_function=ep_id).get_column(["action"])), n_iter)
    rec.time(f"slicestack h=16 (episode_end mask) get_column(obs/state) B={bs}",
             sampler(SliceStackedBatch(rb, h16, get_valid_mask_function=ep_end).get_column(["obs", "state"])), n_iter)
    rec.time(f"framestack k=2 (episode_id mask) get_column(obs/image) B={bs}",
             sampler(FrameStackedBatch(rb, 1, get_valid_mask_function=ep_id).get_column(["obs", "image"])), n_iter)
    rec.time(f"combined x2 of framestack k=2, thread pool B={bs}",
             sampler(CombinedBatch([FrameStackedBatch(r, 1, get_valid_mask_function=ep_id) for r in halves],
                                   use_thread_pool=True)), n_iter)
    rec.time("combined x4, thread pool B=32",
             sampler(CombinedBatch(quarters, use_thread_pool=True), 32), n_iter * 5)
    rec.time("combined x4, no thread pool B=32",
             sampler(CombinedBatch(quarters, use_thread_pool=False), 32), n_iter * 5)
    rec.time(f"combined x2 of slicestack h=16 get_column(action) B={bs}",
             sampler(CombinedBatch([SliceStackedBatch(r, h16, get_valid_mask_function=ep_id) for r in halves])
                     .get_column(["action"])), n_iter)

    # --- JPEG decode ---
    from unienv_interface.backends.numpy import NumpyComputeBackend as NB
    from unienv_interface.space import BoxSpace
    from unienv_data.transformations.image_compress import ImageCompressTransformation
    hw = 64 if args.quick else 128
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:hw, 0:hw]
    base = np.stack([(xx + yy) % 256, (2 * xx) % 256, (3 * yy) % 256], -1).astype(np.float32)
    images = np.clip(base[None] + rng.normal(0, 8, (64, hw, hw, 3)), 0, 255).astype(np.uint8)
    img_space = BoxSpace(NB, 0, 255, np.uint8, shape=images.shape)
    encode = ImageCompressTransformation(init_quality=75)
    encoded = encode.transform(img_space, images)
    decode = encode.direction_inverse(img_space)
    encoded_space = encode.get_target_space_from_source(img_space)
    rec.time(f"ImageDecompressTransformation 64 x {hw}x{hw} JPEG",
             lambda: decode.transform(encoded_space, encoded), n_iter)



if __name__ == "__main__":
    _common.main(__file__, ARGS, ARGV, run)
