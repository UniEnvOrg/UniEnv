"""VideoStorage reads of GOP-encoded (libx264) files."""

import os

import numpy as np
import pytest

av = pytest.importorskip("av")
from av.codec import codecs_available  # noqa: E402

from unienv_data.storages import _io_pool  # noqa: E402
from unienv_data.storages.video_storage import VideoStorage  # noqa: E402
from unienv_interface.backends.numpy import NumpyComputeBackend  # noqa: E402
from unienv_interface.space import BoxSpace  # noqa: E402

B = NumpyComputeBackend
H, W = 32, 32

pytestmark = pytest.mark.skipif("libx264" not in codecs_available, reason="needs libx264")


def _frames(n: int, seed: int = 0) -> np.ndarray:
    """Smoothly moving content, so inter frames actually reference earlier frames."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rng = np.random.default_rng(seed)
    frames = np.empty((n, H, W, 3), np.uint8)
    for t in range(n):
        blob = np.exp(-((xx - 16 - 10 * np.sin(t / 5)) ** 2 + (yy - 16) ** 2) / 40.0)
        frames[t] = np.clip(np.stack([60 + 150 * blob, 3 * xx, 3 * yy], -1) + rng.normal(0, 2, (H, W, 3)), 0, 255)
    return frames


def _storage(tmp_path, name="v", gop=8, seek_mode="exact", bframes=True, capacity=None):
    options = {"g": str(gop), "crf": "18"}
    if not bframes:
        options["bf"] = "0"
    return VideoStorage.create(
        BoxSpace(B, 0, 255, np.uint8, shape=(H, W, 3)), capacity=capacity, cache_path=str(tmp_path / name),
        codec="libx264", file_ext="mp4", hardware_acceleration=None, decode_backend="pyav",
        encoder_options=options, seek_mode=seek_mode,
    )


def test_reads_spanning_many_files(tmp_path):
    storage = _storage(tmp_path, capacity=48, gop=4)
    frames = _frames(48)
    for start in range(0, 48, 6):  # 8 files
        storage.set(slice(start, start + 6), frames[start:start + 6])
    assert len(storage.get_segments()) == 8
    index = np.array([47, 0, 13, 13, 30, 7, 46, 1, 25, 40, 19])
    result = storage.get(index)
    expected = np.stack([storage.get(int(i)) for i in index])
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(storage.get(slice(None)), np.stack([storage.get(i) for i in range(48)]))
    # Reads issued from the shared IO pool's own workers (nested) decode serially instead of waiting on it
    for nested in _io_pool.map_io(lambda _: storage.get(index), range(2 * _io_pool.IO_POOL_WORKERS)):
        np.testing.assert_array_equal(nested, expected)
    storage.close()
