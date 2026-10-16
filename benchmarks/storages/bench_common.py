"""Shared helpers for the storage benchmarks (see README.md in this folder)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--src", default=REPO_ROOT,
                        help="checkout whose unienv_data / unienv_interface to benchmark (default: this repository)")
    parser.add_argument("--data-dir", default=None,
                        help="where storages are created (default: a temporary directory, removed afterwards)")
    parser.add_argument("--out", default=None, help="append one JSON line per result to this file")
    parser.add_argument("--threads", type=int, default=4, help="torch.set_num_threads (default: 4)")


def import_src(src: str) -> None:
    """Import unienv from `src` (call before importing any unienv module)."""
    src = os.path.abspath(src)
    if src not in sys.path:
        sys.path.insert(0, src)
    import unienv_data
    assert os.path.dirname(os.path.dirname(os.path.abspath(unienv_data.__file__))) == src, \
        f"unienv_data was imported from {unienv_data.__file__}, not from {src}"


class DataDir:
    """A directory for benchmark storages; temporary ones are removed on exit."""

    def __init__(self, path: Optional[str]):
        self._temporary = path is None
        self.path = tempfile.mkdtemp(prefix="unienv-storage-bench-") if path is None else os.path.abspath(path)
        os.makedirs(self.path, exist_ok=True)

    def fresh(self, name: str) -> str:
        path = os.path.join(self.path, name)
        if os.path.exists(path):
            shutil.rmtree(path)
        return path

    def remove(self, name: str) -> None:
        shutil.rmtree(os.path.join(self.path, name), ignore_errors=True)

    def close(self) -> None:
        if self._temporary:
            shutil.rmtree(self.path, ignore_errors=True)


def disk_bytes(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def make_frames(n: int, size: int, chw: bool = False, seed: int = 0):
    """Smooth, slowly moving synthetic camera frames (compressible like real images), uint8 (n, H, W, 3)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    scale = size / 128.0
    frames = np.empty((n, size, size, 3), np.uint8)
    for t in range(n):
        cx = size / 2 + 30 * scale * np.sin(t / 20.0)
        cy = size / 2 + 30 * scale * np.cos(t / 25.0)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (300.0 * scale * scale))
        image = np.stack([80 + 0.5 * xx / scale + 120 * blob, 60 + 0.6 * yy / scale + 60 * blob,
                          120 + 0.2 * (xx + yy) / scale - 80 * blob], -1)
        frames[t] = np.clip(image + rng.normal(0, 3, (size, size, 3)), 0, 255).astype(np.uint8)
    if chw:
        frames = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
    return frames


def timed(fn: Callable[[], Any]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def emit(result: Dict[str, Any], out: Optional[str]) -> None:
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(result) + "\n")


def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"
