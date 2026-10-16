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


@pytest.mark.parametrize("keys", [["reward"], ["obs", "pos"]], ids=["top-leaf", "nested-leaf"])
def test_leaf_column_view_reads_and_writes_parent(nested_storage, keys):
    view = nested_storage.get_column(keys)
    expected = _nested_batch(4)
    for key in keys:
        expected = expected[key]
    assert len(view) == 4 and view.capacity == 4
    np.testing.assert_array_equal(view.get(slice(None)), expected)
    np.testing.assert_array_equal(view.get(2), expected[2])
    new_value = np.full_like(expected[1], 42.0)
    view.set(1, new_value)
    parent_value = nested_storage.get(1)
    for key in keys:
        parent_value = parent_value[key]
    np.testing.assert_array_equal(parent_value, new_value)


def test_leaf_column_view_dumps_as_standalone_storage(tmp_path, nested_storage):
    view = nested_storage.get_column(["obs", "vel"])
    target = str(tmp_path / "vel.hdf5")
    view.dumps(target)
    loaded = HDF5Storage.load_from(target, _box((1,)), capacity=4)
    np.testing.assert_array_equal(loaded.get(slice(None)), _nested_batch(4)["obs"]["vel"])
    loaded.close()


@pytest.mark.parametrize("how", ["close", "gc"])
def test_releasing_column_view_keeps_parent_file_open(nested_storage, how):
    view = nested_storage.get_column(["obs"])
    view.get(0)
    if how == "close":
        view.close()
    else:
        del view
        gc.collect()
    assert_tree_equal(nested_storage.get(slice(None)), _nested_batch(4))
    nested_storage.set(0, {"obs": {"pos": np.ones(2, np.float32), "vel": np.ones(1, np.float32)}, "reward": np.float32(5.0)})
    assert nested_storage.get(0)["reward"] == 5.0


def test_column_view_of_lazily_opened_storage_owns_its_handle(tmp_path, nested_storage):
    import pickle

    path = nested_storage.cache_filename
    nested_storage.dumps(path)
    reader = HDF5Storage.load_from(path, _nested_space(), capacity=4, read_only=True)
    clone = pickle.loads(pickle.dumps(reader))
    assert clone.root is None  # opened lazily
    view = clone.get_column(["reward"])
    np.testing.assert_array_equal(view.get(slice(None)), _nested_batch(4)["reward"])
    view.close()
    assert view.root is None
    assert_tree_equal(reader.get(slice(None)), _nested_batch(4))
    clone.close()
    reader.close()


def _rows(batch: dict, rows) -> dict:
    rows = np.asarray(rows)
    return {"obs": {k: v[rows] for k, v in batch["obs"].items()}, "reward": batch["reward"][rows]}


@pytest.mark.parametrize("index,rows", [
    (np.array([-1, 0]), [3, 0]),
    (np.array([2, -4, 2]), [2, 0, 2]),
    (np.array([-2]), [2]),
    (np.array([True, False, False, True]), [0, 3]),
    (np.array([], dtype=np.int64), []),
], ids=["negative", "negative-duplicates", "single-negative", "bool-mask", "empty"])
def test_get_with_array_indices(nested_storage, index, rows):
    result = nested_storage.get(index)
    expected = _rows(_nested_batch(4), np.array(rows, dtype=np.int64))
    assert_tree_equal(result, expected)


def test_set_with_negative_array_indices(nested_storage):
    update = _nested_batch(2, start=100.0)
    nested_storage.set(np.array([-1, -3]), update)
    expected = _nested_batch(4)
    for rows_src, row_dst in ((0, 3), (1, 1)):
        expected["obs"]["pos"][row_dst] = update["obs"]["pos"][rows_src]
        expected["obs"]["vel"][row_dst] = update["obs"]["vel"][rows_src]
        expected["reward"][row_dst] = update["reward"][rows_src]
    assert_tree_equal(nested_storage.get(slice(None)), expected)
    nested_storage.set(np.array([], dtype=np.int64), _rows(update, np.array([], dtype=np.int64)))
    assert_tree_equal(nested_storage.get(slice(None)), expected)


def _text_space() -> DictSpace:
    from unienv_interface.space import BinarySpace, TextSpace
    return DictSpace(B, {
        "task": TextSpace(B, max_length=16),
        "obs": _box((2,)),
        "flags": BinarySpace(B, shape=(2,), dtype=np.bool_),
        "step": BoxSpace(B, 0, 100, np.int64, shape=()),
    })


def _text_batch() -> dict:
    return {
        "task": np.array(["pick", "place", "push"], dtype=object),
        "obs": np.arange(6, dtype=np.float32).reshape(3, 2),
        "flags": np.array([[True, False], [False, False], [True, True]]),
        "step": np.array([3, 4, 5], dtype=np.int64),
    }


def test_text_fields_round_trip(tmp_path):
    path = str(tmp_path / "text.hdf5")
    storage = HDF5Storage.create(_text_space(), capacity=3, cache_path=path)
    storage.set(slice(None), _text_batch())
    storage.set(1, {"task": "lift", "obs": np.zeros(2, np.float32), "flags": np.array([True, False]),
                    "step": np.int64(9)})
    expected = _text_batch()
    expected["task"][1] = "lift"
    expected["obs"][1] = 0
    expected["flags"][1] = [True, False]
    expected["step"][1] = 9
    assert_tree_equal(storage.get(slice(None)), expected)
    assert storage.get(1)["task"] == "lift"
    storage.close()


def test_load_replay_buffer_from_raw_hdf5_infers_space(tmp_path):
    path = str(tmp_path / "raw.hdf5")
    with h5py.File(path, "w") as f:
        f.create_dataset("task", data=np.array([b"pick", b"place", b"push"], dtype="S8"), maxshape=(None,))
        f.create_dataset("note", data=np.array(["a", "bc", "def"], dtype=h5py.string_dtype()), maxshape=(None,))
        f.create_dataset("obs", data=np.arange(6, dtype=np.float32).reshape(3, 2), maxshape=(None, 2))
        f.create_dataset("flags", data=np.array([True, False, True]), maxshape=(None,))
        f.create_dataset("_ignored", data=np.zeros(3))
    rb = HDF5Storage.load_replay_buffer_from_raw_hdf5(path)
    assert len(rb) == 3 and rb.capacity is None
    space = rb.single_space
    assert set(space.spaces) == {"task", "note", "obs", "flags"}
    assert space["task"].max_length == 8
    assert space["note"].max_length >= 3
    row = rb.get_at(2)
    assert row["task"] == "push" and row["note"] == "def"
    np.testing.assert_array_equal(row["obs"], np.array([4.0, 5.0], np.float32))
    assert bool(row["flags"]) is True
    rb.storage.close()


def test_load_replay_buffer_from_raw_hdf5_default_key(tmp_path):
    path = str(tmp_path / "raw_box.hdf5")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.arange(8, dtype=np.int32).reshape(4, 2), maxshape=(4, 2))
    rb = HDF5Storage.load_replay_buffer_from_raw_hdf5(path)
    assert rb.capacity == 4 and len(rb) == 4
    assert isinstance(rb.single_space, BoxSpace)
    np.testing.assert_array_equal(rb.get_at(slice(None)), np.arange(8, dtype=np.int32).reshape(4, 2))
    rb.storage.close()


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------

from unienv_data.storages.hdf5 import (  # noqa: E402
    fancy_indexing_to_supported_indexing,
    fancy_indexing_to_supported_set_indexing,
    io_reduced_indexing,
    is_fancy_index,
)


def test_is_fancy_index():
    assert is_fancy_index(np.array([1, 2]))
    assert is_fancy_index(np.array([3], dtype=np.uint8)) is False
    assert not is_fancy_index(np.array([True, False]))
    assert not is_fancy_index(slice(0, 2))
    assert not is_fancy_index(3)


def test_fancy_indexing_helpers_reconstruct_original_order():
    index = np.array([5, 1, 5, 3, 1])
    unique, reverse = fancy_indexing_to_supported_indexing(index)
    np.testing.assert_array_equal(unique, [1, 3, 5])
    np.testing.assert_array_equal(unique[reverse], index)
    unique, first = fancy_indexing_to_supported_set_indexing(index)
    np.testing.assert_array_equal(unique, [1, 3, 5])
    np.testing.assert_array_equal(index[first], unique)


@pytest.mark.parametrize("index,expected", [
    (np.array([], dtype=np.int64), []),
    (np.array([4]), [4]),
    (np.array([0, 1, 2, 3]), [slice(0, 4, 1)]),
    (np.array([0, 2, 4, 30, 31]), [slice(0, 6, 2), slice(30, 32, 1)]),
    (np.array([0, 20, 40]), [np.array([0, 20, 40])]),
    (np.array([0, 1, 50]), [slice(0, 2, 1), 50]),
])
def test_io_reduced_indexing(index, expected):
    result = io_reduced_indexing(index)
    assert len(result) == len(expected)
    for got, want in zip(result, expected):
        if isinstance(want, np.ndarray):
            np.testing.assert_array_equal(got, want)
        else:
            assert got == want
    covered = np.concatenate([np.arange(1000)[r] if not isinstance(r, int) else np.array([r]) for r in result]) \
        if result else np.array([], dtype=np.int64)
    np.testing.assert_array_equal(covered, index)


def test_io_reduced_indexing_rejects_bad_input():
    with pytest.raises(ValueError):
        io_reduced_indexing(np.array([[1, 2]]))
    with pytest.raises(ValueError):
        io_reduced_indexing(np.array([3, 1]))


@pytest.mark.parametrize("reduce_io", [True, False])
def test_fancy_reads_with_and_without_io_reduction(tmp_path, reduce_io):
    storage = HDF5Storage.create(_nested_space(), capacity=40, cache_path=str(tmp_path / "f.hdf5"),
                                 reduce_io=reduce_io)
    storage.set(slice(None), _nested_batch(40))
    index = np.array([39, 0, 1, 2, 20, 22, 24, 0])
    assert_tree_equal(storage.get(index), _rows(_nested_batch(40), index))
    storage.set(np.array([5, 3, 5]), _nested_batch(3, start=100.0))
    # duplicated targets: the first occurrence wins
    assert storage.get(5)["reward"] == 1000.0
    assert storage.get(3)["reward"] == 1010.0
    storage.close()


# ---------------------------------------------------------------------------
# Loading, validation and multiprocessing
# ---------------------------------------------------------------------------

def test_load_validates_file_layout(tmp_path, nested_storage):
    path = nested_storage.cache_filename
    nested_storage.dumps(path)
    with pytest.raises(AssertionError):
        HDF5Storage.load_from(path, _nested_space(), capacity=5)
    with pytest.raises(AssertionError):
        HDF5Storage.load_from(path, DictSpace(B, {"missing": _box()}), capacity=4)
    with pytest.raises(AssertionError):
        HDF5Storage.load_from(path, DictSpace(B, {"reward": _box((3,))}), capacity=4)
    with pytest.raises(AssertionError):
        HDF5Storage.load_from(str(tmp_path / "missing.hdf5"), _nested_space(), capacity=4)
    with pytest.raises(AssertionError):
        HDF5Storage.load_from(path, _nested_space(), capacity=4, read_only=False, multiprocessing=True)


def test_multiprocessing_load_opens_lazily_and_rejects_writes(tmp_path, nested_storage):
    import pickle

    path = nested_storage.cache_filename
    nested_storage.dumps(path)
    nested_storage.close()  # HDF5 refuses mixed locking flags for one file within a process
    loaded = HDF5Storage.load_from(path, _nested_space(), capacity=4, multiprocessing=True)
    assert loaded.root is None and loaded.is_multiprocessing_safe
    assert loaded.cache_filename == path
    assert_tree_equal(loaded.get(slice(None)), _nested_batch(4))
    with pytest.raises(RuntimeError):
        loaded.set(0, _rows(_nested_batch(1), 0))
    view = loaded.get_column(["obs", "pos"])
    np.testing.assert_array_equal(view.get(slice(None)), _nested_batch(4)["obs"]["pos"])
    clone = pickle.loads(pickle.dumps(view))
    np.testing.assert_array_equal(clone.get(1), _nested_batch(4)["obs"]["pos"][1])
    for item in (clone, view, loaded):
        item.close()


def test_create_rejects_multiprocessing_and_missing_path(tmp_path):
    with pytest.raises(AssertionError):
        HDF5Storage.create(_box(), capacity=2, cache_path=None)
    with pytest.raises(AssertionError):
        HDF5Storage.create(_box(), capacity=2, cache_path=str(tmp_path / "x.hdf5"), multiprocessing=True)


def test_mapping_options_for_nested_dicts(tmp_path):
    storage = HDF5Storage.create(
        _nested_space(), capacity=6, cache_path=str(tmp_path / "m.hdf5"),
        compression={"obs": "gzip", "reward": "lzf"}, compression_level={"obs": 3},
        chunks={"obs": (3,), "*": True},
    )
    assert storage.root["obs"]["pos"].compression == "gzip"
    assert storage.root["obs"]["pos"].chunks == (3, 2)
    assert storage.root["reward"].compression == "lzf"
    storage.close()


def test_mutable_storage_pickles_by_filename(tmp_path, nested_storage):
    import pickle

    path = nested_storage.cache_filename
    nested_storage.dumps(path)
    state = nested_storage.__getstate__()
    assert "root" not in state and state["_filename"] == path and state["_mode"] == "r+"
    nested_storage.close()
    clone = pickle.loads(pickle.dumps(HDF5Storage.load_from(path, _nested_space(), capacity=4)))
    assert_tree_equal(clone.get(slice(None)), _nested_batch(4))
    clone.close()
