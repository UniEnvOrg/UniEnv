"""Throughput of ``AsyncVecEnv`` transfer: pipe vs shared memory.

The environment does no work: it returns a preallocated observation, so the
numbers measure the transport (pickling / copying / synchronisation) of the
observations and of the (2,) float32 actions only. ``pipe`` passes
``shared_memory=False``; the ``shared-*`` modes use the default shared memory
(observations, contexts and actions, ``copy=True`` / ``copy=False``).

Usage (from the repository root)::

    python benchmarks/vec_env/bench_async_vec_env.py                 # everything
    python benchmarks/vec_env/bench_async_vec_env.py --num-envs 8 --obs image --backends numpy

To compare with an older version, run this script with that version on the
``PYTHONPATH`` (it only needs the ``shared_memory`` / ``copy`` arguments).

Every configuration uses the ``spawn`` start method. Each configuration is run
``--repeats`` times with the modes interleaved (rotating their order) and the
median is reported, which keeps the comparison fair on a busy machine.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import time
from typing import Optional

import numpy as np

from unienv_interface.env_base.env import Env
from unienv_interface.env_base.vec_env import AsyncVecEnv
from unienv_interface.space import BoxSpace

OBS_SHAPES = {
    "vector": ((32,), "float32"),
    "image": ((3, 128, 128), "uint8"),
}


class TransportBenchEnv(Env):
    """Unbatched env returning a fixed observation (defined in ``__main__``, so
    cloudpickle ships it to spawned workers by value)."""

    render_mode = None
    render_fps = None
    metadata = {}

    def __init__(self, backend: str, device: Optional[str], obs: str):
        shape, dtype = OBS_SHAPES[obs]
        data = np.full(shape, 7, dtype=dtype)
        if backend == "numpy":
            from unienv_interface.backends.numpy import NumpyComputeBackend as B
            self.device = None
            space_dtype, act_dtype = np.dtype(dtype), np.float32
            self._obs = data
        else:
            import torch
            from unienv_interface.backends.pytorch import PyTorchComputeBackend as B
            self.device = torch.device(device)
            space_dtype, act_dtype = getattr(torch, dtype), torch.float32
            self._obs = torch.from_numpy(data).to(self.device)
        self.backend = B
        self.batch_size = None
        low, high = (0, 255) if dtype == "uint8" else (-np.inf, np.inf)
        self.observation_space = BoxSpace(B, low, high, space_dtype, shape=shape, device=self.device)
        self.action_space = BoxSpace(B, -1.0, 1.0, act_dtype, shape=(2,), device=self.device)
        self.context_space = None
        self.rng = B.random.random_number_generator(0, device=self.device)

    def reset(self, *, mask=None, seed=None, **kwargs):
        return None, self._obs, {}

    def step(self, action):
        return self._obs, 0.0, False, False, {}

    def render(self):
        return None


def bench(backend: str, device: Optional[str], obs: str, num_envs: int, mode: str, steps: int, warmup: int) -> float:
    kwargs = {"ctx": mp.get_context("spawn"), "shared_memory": mode != "pipe"}
    if mode != "pipe":
        kwargs["copy"] = mode == "shared-copy"
    env = AsyncVecEnv([lambda: TransportBenchEnv(backend, device, obs) for _ in range(num_envs)], **kwargs)
    try:
        env.reset(seed=0)
        action = env.backend.zeros((num_envs, 2), dtype=env.action_space.dtype, device=env.device)
        for _ in range(warmup):
            env.step(action)
        if backend == "pytorch" and device == "cuda":
            import torch
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            obs_batch, *_ = env.step(action)
        if backend == "pytorch" and device == "cuda":
            import torch
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        env.close()
    return steps / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--obs", nargs="+", default=list(OBS_SHAPES), choices=list(OBS_SHAPES))
    parser.add_argument("--backends", nargs="+", default=["numpy", "torch-cpu", "torch-cuda"],
                        choices=["numpy", "torch-cpu", "torch-cuda"])
    parser.add_argument("--modes", nargs="+", default=["pipe", "shared-copy", "shared-view"],
                        choices=["pipe", "shared-copy", "shared-view"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    backends = []
    for name in args.backends:
        if name == "numpy":
            backends.append(("numpy", None))
            continue
        try:
            import torch
        except ImportError:
            continue
        if name == "torch-cpu":
            backends.append(("pytorch", "cpu"))
        elif torch.cuda.is_available():
            backends.append(("pytorch", "cuda"))

    print("| backend | obs | N | " + " | ".join(f"{m} (steps/s)" for m in args.modes) + " |")
    print("|---|---|---|" + "---|" * len(args.modes))
    for backend, device in backends:
        label = backend if device is None else f"torch-{device}"
        for obs in args.obs:
            for num_envs in args.num_envs:
                samples = {mode: [] for mode in args.modes}
                for repeat in range(args.repeats):
                    shift = repeat % len(args.modes)
                    for mode in args.modes[shift:] + args.modes[:shift]:
                        samples[mode].append(bench(backend, device, obs, num_envs, mode, args.steps, args.warmup))
                rates = [statistics.median(samples[mode]) for mode in args.modes]
                cells = " | ".join(f"{rate:,.0f}" for rate in rates)
                print(f"| {label} | {obs} {OBS_SHAPES[obs][0]} | {num_envs} | {cells} |", flush=True)


if __name__ == "__main__":
    main()
