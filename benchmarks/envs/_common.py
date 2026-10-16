"""Shared helpers for the env-pipeline benchmarks in this folder.

The scripts import ``unienv_interface`` from the repository this folder lives in.
Set ``UNIENV_ROOT=/path/to/other/checkout`` to benchmark another checkout instead
(e.g. a ``git worktree`` of ``main``) for before/after comparisons.
"""
import os
import sys
import time
from typing import Callable, Optional

ROOT = os.environ.get("UNIENV_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    torch.set_num_threads(1)  # small tensors; avoid thread-pool noise
except ImportError:  # pragma: no cover - torch is optional
    torch = None

import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.env_base.env import Env
from unienv_interface.space import BoxSpace, DictSpace

QUICK = "--quick" in sys.argv


def cli_args() -> list:
    """Positional (non-flag) command line arguments."""
    return [a for a in sys.argv[1:] if not a.startswith("--")]


def bench(step_fn: Callable[[], object], *, min_time: float = 0.5, warmup: int = 50, repeats: int = 5) -> float:
    """Best-of-``repeats`` microseconds per call of ``step_fn``."""
    if QUICK:
        min_time, warmup, repeats = min_time / 10, min(warmup, 5), 2
    for _ in range(warmup):
        step_fn()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            step_fn()
        dt = time.perf_counter() - t0
        if dt > 0.02:
            break
        n *= 2
    n = max(1, int(n * min_time / dt))
    results = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            step_fn()
        results.append((time.perf_counter() - t0) / n * 1e6)
    return min(results)


def report(name: str, us: float, batch: Optional[int] = None) -> None:
    b = "-" if batch is None else str(batch)
    print(f"{name:<52} B={b:<5} {us:10.1f} us/step {1e6 / us:10.0f} steps/s", flush=True)


class ImageEnv(Env):
    """Stateful numpy env with a ``state`` vector and an optional uint8 ``image``."""

    backend, device = B, None
    render_mode = None
    metadata = {}

    def __init__(self, batch_size: Optional[int] = None, img=(84, 84, 3)):
        self.batch_size = batch_size
        bs = () if batch_size is None else (batch_size,)
        self.action_space = BoxSpace(B, -1.0, 1.0, np.float32, shape=bs + (7,))
        spaces = {"state": BoxSpace(B, -np.inf, np.inf, np.float32, shape=bs + (32,))}
        if img is not None:
            spaces["image"] = BoxSpace(B, 0, 255, np.uint8, shape=bs + tuple(img))
        self.observation_space = DictSpace(B, spaces)
        self.context_space = None
        self.state = np.zeros(bs + (32,), np.float32)
        self.image = None if img is None else np.random.default_rng(0).integers(0, 255, bs + tuple(img), dtype=np.uint8)

    def _obs(self):
        obs = {"state": self.state.copy()}
        if self.image is not None:
            obs["image"] = self.image
        return obs

    def reset(self, *, mask=None, seed=None, **kwargs):
        return None, self._obs(), {}

    def step(self, action):
        self.state[..., :7] += action
        if self.batch_size is None:
            return self._obs(), 0.0, False, False, {"k": 1}
        zeros = np.zeros(self.batch_size, np.float32)
        return self._obs(), zeros, zeros.astype(bool), zeros.astype(bool), {"k": 1}

    def render(self):
        return None
