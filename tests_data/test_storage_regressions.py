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
