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


@pytest.mark.parametrize("create_kwargs", [{}, {"initial_capacity": 0}], ids=["default", "explicit-zero"])
def test_create_unbounded_storage_starts_empty_and_grows(tmp_path, create_kwargs):
    path = str(tmp_path / "unbounded.hdf5")
    storage = HDF5Storage.create(_nested_space(), capacity=None, cache_path=path, **create_kwargs)
    assert storage.capacity is None and len(storage) == 0
    storage.extend_length(3)
    storage.set(slice(0, 3), _nested_batch(3))
    storage.extend_length(2)
    storage.set(slice(3, 5), _nested_batch(2, start=3.0))
    assert len(storage) == 5
    assert_tree_equal(storage.get(slice(None)), _nested_batch(5))
    storage.shrink_length(4)
    assert len(storage) == 1
    storage.dumps(path)
    storage.close()

    loaded = HDF5Storage.load_from(path, _nested_space(), capacity=None)
    assert len(loaded) == 1
    assert_tree_equal(loaded.get(0), {"obs": {"pos": np.zeros(2, np.float32), "vel": np.zeros(1, np.float32)},
                                      "reward": np.float32(0.0)})
    loaded.close()


def test_unbounded_replay_buffer_with_hdf5_storage(tmp_path):
    from unienv_data.replay_buffer import ReplayBuffer

    cache = str(tmp_path / "rb")
    rb = ReplayBuffer.create(HDF5Storage, _box(), cache_path=cache, capacity=None)
    rb.extend(np.arange(6, dtype=np.float32).reshape(3, 2))
    rb.append(np.array([6.0, 7.0], np.float32))
    rb.mark_segment_end()
    assert len(rb) == 4
    rb.dumps(cache)
    rb.storage.close()
    loaded = ReplayBuffer.load_from(cache, backend=B)
    np.testing.assert_array_equal(loaded.get_at(slice(None)), np.arange(8, dtype=np.float32).reshape(4, 2))
    loaded.storage.close()


@pytest.fixture
def nested_storage(tmp_path):
    storage = HDF5Storage.create(_nested_space(), capacity=4, cache_path=str(tmp_path / "nested.hdf5"))
    storage.set(slice(None), _nested_batch(4))
    yield storage
    storage.close()


@pytest.mark.parametrize("existing", [False, True], ids=["new-file", "existing-file"])
def test_dumps_to_another_path_is_loadable(tmp_path, nested_storage, existing):
    target = tmp_path / "sub" / "copy.hdf5"
    if existing:
        target.parent.mkdir()
        with h5py.File(target, "w") as f:
            f.create_dataset("stale", data=np.zeros(3))
    nested_storage.dumps(str(target))
    loaded = HDF5Storage.load_from(str(target), _nested_space(), capacity=4)
    assert set(loaded.root.keys()) == {"obs", "reward"}
    assert_tree_equal(loaded.get(slice(None)), _nested_batch(4))
    loaded.close()
    # the source storage is untouched and still usable
    assert_tree_equal(nested_storage.get(slice(None)), _nested_batch(4))


def test_dumps_to_own_path_flushes(tmp_path, nested_storage):
    nested_storage.set(1, {"obs": {"pos": np.ones(2, np.float32), "vel": np.ones(1, np.float32)}, "reward": np.float32(7.0)})
    path = nested_storage.cache_filename
    nested_storage.dumps(path)
    assert nested_storage.root is not None  # still open after dumping in place
    nested_storage.close()
    loaded = HDF5Storage.load_from(path, _nested_space(), capacity=4)
    assert loaded.get(1)["reward"] == 7.0
    loaded.close()


def test_dumps_group_column_view_to_another_path(tmp_path, nested_storage):
    view = nested_storage.get_column(["obs"])
    target = str(tmp_path / "obs.hdf5")
    view.dumps(target)
    loaded = HDF5Storage.load_from(target, _nested_space()["obs"], capacity=4)
    assert_tree_equal(loaded.get(slice(None)), _nested_batch(4)["obs"])
    loaded.close()


def test_dumps_column_view_onto_its_own_file_is_rejected(nested_storage):
    view = nested_storage.get_column(["obs"])
    with pytest.raises(ValueError):
        view.dumps(nested_storage.cache_filename)
    assert_tree_equal(nested_storage.get(slice(None)), _nested_batch(4))
