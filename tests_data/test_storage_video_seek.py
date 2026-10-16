"""
VideoStorage sparse reads of GOP-encoded files: decoding forward within a GOP (from a per-file keyframe index)
returns exactly the frames that seeking to every requested frame returns.
"""

import os

import numpy as np
import pytest

av = pytest.importorskip("av")
from av.codec import codecs_available  # noqa: E402

from unienv_data.storages import video_storage as video_storage_module  # noqa: E402
from unienv_data.storages import _io_pool  # noqa: E402
from unienv_data.storages.video_storage import PyAvVideoReader, VideoStorage  # noqa: E402
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


def _reference_reads(monkeypatch, storage, index):
    """The same read with every requested frame reached by a seek (the behaviour before the keyframe index)."""
    with monkeypatch.context() as patch:
        patch.setattr(PyAvVideoReader, "_can_decode_forward", lambda self, frame_index: False)
        return storage.get(index)


@pytest.mark.parametrize("seek_mode", ["exact", "approximate"])
@pytest.mark.parametrize("bframes", [True, False])
def test_sparse_reads_match_seeking_every_frame(tmp_path, monkeypatch, seek_mode, bframes):
    storage = _storage(tmp_path, seek_mode=seek_mode, bframes=bframes)
    n = 60
    storage.extend_length(n)
    storage.set(slice(None), _frames(n))
    rng = np.random.default_rng(0)
    indices = [
        np.array([3, 5, 6, 9, 20, 21, 40, 59]),
        np.array([59, 3, 3, 17, 16, 2]),  # unsorted, duplicated
        np.array([0, 7, 8, 15, 16]),  # GOP boundaries
        np.sort(rng.choice(n, 12, replace=False)),
        rng.integers(0, n, 20),
        np.arange(1, n, 4),
    ]
    for index in indices:
        expected = _reference_reads(monkeypatch, storage, index)
        np.testing.assert_array_equal(storage.get(index), expected)
    storage.close()


def test_keyframe_index_is_cached_and_follows_rewrites(tmp_path, monkeypatch):
    storage = _storage(tmp_path, capacity=30)
    first, second = _frames(30, seed=0), _frames(30, seed=1)[::-1].copy()
    storage.set(slice(None), first)
    index = np.array([2, 5, 11, 12, 25])
    reference = _reference_reads(monkeypatch, storage, index)
    np.testing.assert_array_equal(storage.get(index), reference)
    (filename, (stamp, keyframes)), = storage._keyframe_cache.items()
    assert keyframes[0] == 0 and np.all(np.diff(keyframes) > 0) and len(keyframes) >= 30 // 8

    calls = []
    original = PyAvVideoReader.keyframe_indices


    def counting_keyframe_indices(self):
        if self.keyframes is None:
            calls.append(1)  # a demux pass over the file
        return original(self)

    monkeypatch.setattr(PyAvVideoReader, "keyframe_indices", counting_keyframe_indices)
    storage.get(index)
    assert calls == []  # served from the cache

    # Rewriting the same index range reuses the file name: the cache must not serve the old GOP layout
    storage.set(slice(None), second)
    assert storage._keyframe_cache == {}
    np.testing.assert_array_equal(storage.get(index), _reference_reads(monkeypatch, storage, index))
    assert len(calls) == 1
    # ... also when another writer replaced the file behind this storage's back
    storage._keyframe_cache[filename] = ((0, 0), np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(storage.get(index), _reference_reads(monkeypatch, storage, index))
    storage.close()


def test_reads_spanning_many_files(tmp_path, monkeypatch):
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
