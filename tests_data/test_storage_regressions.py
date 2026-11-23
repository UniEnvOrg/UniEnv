"""Regression tests for storage bugs found before 1.0 (one test or group per bug)."""

import os

import h5py
import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import BoxSpace, DictSpace, TextSpace

from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages import DictStorage
from unienv_data.storages.hdf5 import HDF5Storage
from unienv_data.storages.image_storage import ImageStorage
from unienv_data.storages.npz_storage import NPZStorage
from unienv_data.storages.parquet import ParquetStorage

from tests_common.storage_video import LOSSLESS_RGB_KWARGS, clean_video_codec_cache  # noqa: F401  (fixture)


def _box(shape=(2,), dtype=np.float32):
    return BoxSpace(NB, -1e3, 1e3, dtype, shape=shape)


def _img_space():
    return BoxSpace(NB, 0, 255, np.uint8, shape=(8, 8, 3))


def _frame(value):
    return np.full((8, 8, 3), value, np.uint8)


# ---------------------------------------------------------------------------
# Unbounded List storages are not multiprocessing-safe (their length is per process)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("storage_cls,kwargs,space", [
    (NPZStorage, {}, _box((2,))),
    (ImageStorage, {"format": "PNG"}, _img_space()),
])
def test_unbounded_list_storage_is_not_multiprocessing_safe(tmp_path, storage_cls, kwargs, space):
    unbounded = storage_cls.create(space, capacity=None, cache_path=str(tmp_path / "u"), **kwargs)
    assert not unbounded.is_multiprocessing_safe
    with pytest.raises(AssertionError, match="not multiprocessing safe"):
        ReplayBuffer.create(storage_cls, space, cache_path=str(tmp_path / "rb"), capacity=None,
                            multiprocessing=True, **kwargs)
    # Fixed-capacity (every process agrees on the length) and read-only storages still are
    bounded = storage_cls.create(space, capacity=3, cache_path=str(tmp_path / "b"), **kwargs)
    assert bounded.is_multiprocessing_safe
    unbounded.dumps(str(tmp_path / "u"))
    assert storage_cls.load_from(str(tmp_path / "u"), space, read_only=True).is_multiprocessing_safe


def test_list_storage_set_past_length_raises_instead_of_dropping(tmp_path):
    storage = NPZStorage.create(_box((2,)), capacity=None, cache_path=str(tmp_path / "s"))
    storage.extend_length(2)
    with pytest.raises(ValueError):
        storage.set(slice(1, 3), np.ones((2, 2), np.float32))
    assert sorted(os.listdir(tmp_path / "s")) == []


# ---------------------------------------------------------------------------
# One invalid video frame must not drop the open segment
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("clean_video_codec_cache")
def test_invalid_video_frame_keeps_open_segment(tmp_path):
    from unienv_data.storages.video_storage import VideoStorage

    rb = ReplayBuffer.create(VideoStorage, _img_space(), cache_path=str(tmp_path / "rb"), capacity=10,
                             **LOSSLESS_RGB_KWARGS)
    for i in range(3):
        rb.append(_frame(10 * i))
    with pytest.raises(ValueError):  # InvalidVideoFrameError
        rb.append(np.zeros((8, 8, 3), np.float32))  # wrong dtype
    assert rb.storage.has_open_segment
    with pytest.raises(ValueError):
        rb.append(np.zeros((4, 8, 3), np.uint8))  # wrong shape
    assert rb.storage.has_open_segment
    rb.append(_frame(30))
    rb.mark_segment_end()
    assert len(rb) == 4
    np.testing.assert_array_equal(rb[np.arange(4)][:, 0, 0, 0], [0, 10, 20, 30])


@pytest.mark.usefixtures("clean_video_codec_cache")
@pytest.mark.parametrize("bad_key", ["img", "x"])
def test_dict_storage_child_rejecting_a_sample_keeps_children_in_step(tmp_path, bad_key):
    from unienv_data.storages.video_storage import VideoStorage

    space = DictSpace(NB, {"img": _img_space(), "x": _box((1,))})
    rb = ReplayBuffer.create(
        DictStorage, space, cache_path=str(tmp_path / "rb"), capacity=None,
        storage_cls_map={"img": VideoStorage, "*": HDF5Storage}, key_kwargs={"img": LOSSLESS_RGB_KWARGS},
    )
    for i in range(3):
        rb.append({"img": _frame(10 * i), "x": np.array([i], np.float32)})
    bad = {"img": _frame(99), "x": np.array([99], np.float32)}
    bad[bad_key] = np.zeros((8, 8, 3), np.float32) if bad_key == "img" else np.zeros((5, 5), np.float32)
    with pytest.raises(Exception):
        rb.append(bad)
    assert {k: s.has_open_segment for k, s in rb.storage.storage_map.items()} == {"img": True, "*": True}
    rb.append({"img": _frame(30), "x": np.array([3], np.float32)})
    rb.mark_segment_end()
    assert len(rb) == 4
    data = rb[np.arange(4)]
    np.testing.assert_array_equal(data["img"][:, 0, 0, 0], [0, 10, 20, 30])
    np.testing.assert_array_equal(data["x"][:, 0], [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# clear() on an empty unbounded storage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("storage_cls,name", [(HDF5Storage, "s.hdf5"), (ParquetStorage, "s")])
def test_clear_on_empty_unbounded_storage(tmp_path, storage_cls, name):
    rb = ReplayBuffer.create(storage_cls, _box(), cache_path=str(tmp_path / "rb"), capacity=None)
    rb.clear()  # never written
    rb.extend(np.zeros((3, 2), np.float32))
    rb.clear()
    rb.clear()  # already empty
    assert len(rb) == 0 and len(rb.storage) == 0
    rb.extend(np.ones((2, 2), np.float32))
    np.testing.assert_array_equal(rb[np.arange(2)], np.ones((2, 2), np.float32))


# ---------------------------------------------------------------------------
# Episode storages: negative ints and duplicated indices in set()
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("clean_video_codec_cache")
def test_episode_storage_set_rejects_negative_int_and_resolves_duplicates(tmp_path):
    from unienv_data.storages.video_storage import VideoStorage

    path = str(tmp_path / "v")
    storage = VideoStorage.create(_img_space(), capacity=5, cache_path=path, **LOSSLESS_RGB_KWARGS)
    storage.set(slice(None), np.stack([_frame(10 + i) for i in range(5)]))
    with pytest.raises(AssertionError):
        storage.set(-1, _frame(99))
    assert storage.get_segments() == [(0, 4)]
    assert sorted(os.listdir(path)) == ["0_4.mkv"]

    storage.set(np.array([3, 3, 4]), np.stack([_frame(50), _frame(51), _frame(60)]))
    assert storage.get(3)[0, 0, 0] == 51  # the last value written to a row wins
    np.testing.assert_array_equal(storage.get(slice(None))[:, 0, 0, 0], [10, 11, 12, 51, 60])
    storage.close()
