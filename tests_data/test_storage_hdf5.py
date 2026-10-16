"""HDF5Storage specifics: creation options, dumps, column views and indexing."""

import gc
import os

import h5py
import numpy as np
import pytest

from unienv_data.storages.hdf5 import HDF5Storage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace, DictSpace

from tests_common.trees import assert_tree_equal

B = NumpyComputeBackend


def _box(shape=(2,)) -> BoxSpace:
    return BoxSpace(B, -10.0, 10.0, np.float32, shape=shape)


def _nested_space() -> DictSpace:
    return DictSpace(B, {
        "obs": DictSpace(B, {"pos": _box((2,)), "vel": _box((1,))}),
        "reward": _box(()),
    })


def _nested_batch(n: int, start: float = 0.0) -> dict:
    base = np.arange(n, dtype=np.float32) + start
    return {
        "obs": {"pos": np.stack([base, -base], axis=1), "vel": base[:, None] * 2},
        "reward": base * 10,
    }


def test_create_accepts_h5py_file_kwargs(tmp_path):
    path = str(tmp_path / "kw.hdf5")
    storage = HDF5Storage.create(_box(), capacity=3, cache_path=path, libver="latest")
    storage.set(0, np.array([1.0, 2.0], np.float32))
    np.testing.assert_array_equal(storage.get(0), np.array([1.0, 2.0], np.float32))
    storage.close()


@pytest.mark.parametrize("compression", ["gzip", {"data": "gzip"}], ids=["str", "mapping"])
def test_create_applies_compression_and_chunks_to_non_dict_space(tmp_path, compression):
    path = str(tmp_path / "c.hdf5")
    storage = HDF5Storage.create(_box((4,)), capacity=8, cache_path=path,
                                 compression=compression, compression_level=4, chunks=(2,))
    dataset = storage.root[HDF5Storage.DEFAULT_KEY]
    assert dataset.compression == "gzip"
    assert dataset.chunks == (2, 4)
    storage.close()
