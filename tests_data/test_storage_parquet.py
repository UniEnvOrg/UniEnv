"""ParquetStorage specifics not covered by the generic storage contract."""

import numpy as np
import pytest

from unienv_data.storages.parquet import ParquetStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace, DictSpace

from tests_common.trees import assert_tree_equal

B = NumpyComputeBackend


def _space() -> DictSpace:
    return DictSpace(B, {
        "obs": BoxSpace(B, -10.0, 10.0, np.float32, shape=(2,)),
        "step": BoxSpace(B, 0, 100, np.int64, shape=()),
    })


def _batch(n: int) -> dict:
    return {"obs": np.arange(2 * n, dtype=np.float32).reshape(n, 2), "step": np.arange(n, dtype=np.int64)}


@pytest.fixture(params=[False, True], ids=["mutable", "read-only"])
def storage(request, tmp_path):
    path = str(tmp_path / "pq")
    writer = ParquetStorage.create(_space(), capacity=5, cache_path=path, piece_size=2)
    writer.set(slice(None), _batch(5))
    writer.dumps(path)
    if not request.param:
        yield writer
    else:
        writer.close()
        reader = ParquetStorage.load_from(path, _space(), capacity=5, read_only=True)
        yield reader
        reader.close()


def test_get_with_boolean_mask(storage):
    mask = np.array([True, False, False, True, True])
    result = storage.get(mask)
    expected = {k: v[mask] for k, v in _batch(5).items()}
    assert_tree_equal(result, expected)


def test_set_with_boolean_mask(tmp_path):
    storage = ParquetStorage.create(_space(), capacity=5, cache_path=str(tmp_path / "pq"), piece_size=2)
    storage.set(slice(None), _batch(5))
    mask = np.array([False, True, False, False, True])
    storage.set(mask, {"obs": np.full((2, 2), -1.0, np.float32), "step": np.array([70, 80])})
    np.testing.assert_array_equal(storage.get(Ellipsis)["step"], np.array([0, 70, 2, 3, 80]))
    storage.close()


def _text_space() -> DictSpace:
    from unienv_interface.space import BinarySpace, TextSpace
    return DictSpace(B, {
        "task": TextSpace(B, max_length=32),
        "obs": BoxSpace(B, -10.0, 10.0, np.float32, shape=(2,)),
        "flags": BinarySpace(B, shape=(2,), dtype=np.bool_),
        "flag": BinarySpace(B, shape=(), dtype=np.bool_),
        "step": BoxSpace(B, 0, 100, np.int64, shape=()),
        "one": BoxSpace(B, 0.0, 1.0, np.float64, shape=(1,)),
    })


def _text_batch() -> dict:
    return {
        "task": np.array(["pick", "place", "push"], dtype=object),
        "obs": np.arange(6, dtype=np.float32).reshape(3, 2),
        "flags": np.array([[True, False], [False, False], [True, True]]),
        "flag": np.array([True, False, True]),
        "step": np.array([3, 4, 5], dtype=np.int64),
        "one": np.array([[0.5], [0.25], [1.0]]),
    }


@pytest.mark.parametrize("piece_files", [True, False], ids=["from-pieces", "from-metadata"])
def test_load_replay_buffer_from_raw_parquet_infers_space(tmp_path, piece_files):
    import os
    path = str(tmp_path / "raw")
    storage = ParquetStorage.create(_text_space(), capacity=None if not piece_files else 3, cache_path=path,
                                    piece_size=2)
    if piece_files:
        storage.set(slice(None), _text_batch())
    storage.dumps(path)
    storage.close()
    count, capacity, space = ParquetStorage.build_space_from_parquet_file(path)
    assert set(space.spaces) == set(_text_space().spaces)
    assert space["obs"].shape == (2,) and space["one"].shape == (1,)
    rb = ParquetStorage.load_replay_buffer_from_raw_parquet(path)
    if piece_files:
        assert len(rb) == 3
        assert_tree_equal(rb.get_at(slice(None)), _text_batch())
        assert rb.get_at(1)["task"] == "place"
    else:
        assert len(rb) == 0
        assert os.listdir(path) == ["_metadata.json"]
    rb.storage.close()


def test_raw_parquet_box_space_round_trip(tmp_path):
    path = str(tmp_path / "box")
    storage = ParquetStorage.create(BoxSpace(B, 0, 10, np.uint8, shape=(2, 2)), capacity=3, cache_path=path)
    storage.set(slice(None), np.arange(12, dtype=np.uint8).reshape(3, 2, 2))
    storage.dumps(path)
    storage.close()
    rb = ParquetStorage.load_replay_buffer_from_raw_parquet(path)
    assert isinstance(rb.single_space, BoxSpace) and rb.single_space.shape == (2, 2)
    np.testing.assert_array_equal(rb.get_at(slice(None)), np.arange(12, dtype=np.uint8).reshape(3, 2, 2))
    rb.storage.close()


def test_column_view_reads_writes_and_dumps_through_parent(tmp_path):
    path = str(tmp_path / "pq")
    storage = ParquetStorage.create(_space(), capacity=5, cache_path=path, piece_size=2)
    storage.set(slice(None), _batch(5))
    column = storage.get_column(["step"])
    assert len(column) == 5 and column.capacity == 5
    assert column.cache_filename == path and column.is_mutable and not column.is_multiprocessing_safe
    np.testing.assert_array_equal(column.get(slice(None)), np.arange(5))
    column.set(np.array([0, 4]), np.array([50, 40]))
    np.testing.assert_array_equal(storage.get(slice(None))["step"], np.array([50, 1, 2, 3, 40]))
    np.testing.assert_array_equal(storage.get(1)["obs"], np.array([2.0, 3.0], np.float32))
    column.dumps(path)
    column.close()
    storage.close()
    reloaded = ParquetStorage.load_from(path, _space(), capacity=5)
    np.testing.assert_array_equal(reloaded.get(slice(None))["step"], np.array([50, 1, 2, 3, 40]))
    with pytest.raises(AssertionError):
        reloaded.get_column(["missing"])
    with pytest.raises(AssertionError):
        reloaded.get_column(["obs", "x"])
    reloaded.close()


def test_column_view_of_non_dict_storage_is_rejected(tmp_path):
    storage = ParquetStorage.create(BoxSpace(B, 0.0, 1.0, np.float32, shape=(2,)), capacity=2,
                                    cache_path=str(tmp_path / "pq"))
    with pytest.raises(AssertionError):
        storage.get_column(["data"])


def test_nested_dict_space_is_rejected(tmp_path):
    nested = DictSpace(B, {"a": DictSpace(B, {"b": BoxSpace(B, 0.0, 1.0, np.float32, shape=())})})
    with pytest.raises(AssertionError, match="Nested DictSpace"):
        ParquetStorage.create(nested, capacity=2, cache_path=str(tmp_path / "pq"))


def test_unbounded_growth_across_pieces_and_truncation(tmp_path):
    path = str(tmp_path / "pq")
    storage = ParquetStorage.create(_space(), capacity=None, cache_path=path, piece_size=3)
    storage.extend_length(2)
    storage.set(slice(None), _batch(2))
    storage.extend_length(5)  # fills the partial piece, then adds new pieces
    assert len(storage) == 7 and storage._num_pieces == 3
    storage.set(slice(2, 7), {"obs": _batch(7)["obs"][2:], "step": np.arange(2, 7)})
    assert_tree_equal(storage.get(slice(None)), _batch(7))
    storage.shrink_length(3)  # cached last piece is truncated
    assert len(storage) == 4
    storage.close()
    reopened = ParquetStorage.load_from(path, _space(), read_only=False)
    reopened.shrink_length(2)  # last piece is loaded from disk and truncated
    reopened.close()
    final = ParquetStorage.load_from(path, _space())
    assert len(final) == 2
    assert_tree_equal(final.get(slice(None)), _batch(2))
    with pytest.raises(AssertionError):
        final.extend_length(1)
    final.close()


def test_index_validation(storage):
    with pytest.raises(AssertionError):
        storage.get(5)
    with pytest.raises(AssertionError):
        storage.get(np.array([0, 9]))
    with pytest.raises(AssertionError):
        storage.get(np.array([True, False]))
    with pytest.raises(ValueError):
        storage.get("0")
    assert_tree_equal(storage.get([4, -5]), {k: v[[4, 0]] for k, v in _batch(5).items()})
    assert_tree_equal(storage.get(np.int64(2)), {k: v[2] for k, v in _batch(5).items()})


@pytest.mark.parametrize("compression", [None, "gzip", {"obs": "zstd", "step": "snappy"}],
                         ids=["none", "gzip", "per-column"])
def test_compression_options_round_trip(tmp_path, compression):
    path = str(tmp_path / "pq")
    storage = ParquetStorage.create(_space(), capacity=4, cache_path=path, piece_size=4, compression=compression)
    storage.set(slice(None), _batch(4))
    storage.dumps(path)
    storage.close()
    loaded = ParquetStorage.load_from(path, _space(), capacity=4)
    assert_tree_equal(loaded.get(slice(None)), _batch(4))
    loaded.close()


def test_pickling_flushes_dirty_pieces(tmp_path):
    import pickle
    path = str(tmp_path / "pq")
    storage = ParquetStorage.create(_space(), capacity=4, cache_path=path, piece_size=2)
    storage.set(slice(None), _batch(4))
    clone = pickle.loads(pickle.dumps(storage))
    assert not storage._dirty_pieces
    assert_tree_equal(clone.get(slice(None)), _batch(4))
    reader = ParquetStorage.load_from(path, _space(), capacity=4)
    assert_tree_equal(reader.get(slice(None)), _batch(4))
    reader.close()
    clone.close()
    storage.close()


def test_fresh_pieces_are_allocated_not_read(tmp_path, monkeypatch):
    # A just-created fixed-capacity storage knows its pieces hold zeros: no piece file is decoded
    import pyarrow.parquet as pq
    path = str(tmp_path / "pq")
    storage = ParquetStorage.create(_space(), capacity=5, cache_path=path, piece_size=2)

    def no_read(*args, **kwargs):
        raise AssertionError("read a freshly created piece from disk")

    monkeypatch.setattr(pq, "read_table", no_read)
    storage.set(slice(3, 5), {"obs": np.ones((2, 2), np.float32), "step": np.array([7, 8])})
    np.testing.assert_array_equal(storage.get(Ellipsis)["step"], np.array([0, 0, 0, 7, 8]))
    assert_tree_equal(storage.get(1), {"obs": np.zeros(2, np.float32), "step": np.int64(0)})
    monkeypatch.undo()
    storage.dumps(path)
    storage.close()
    reader = ParquetStorage.load_from(path, _space(), capacity=5, read_only=True)
    np.testing.assert_array_equal(reader.get(Ellipsis)["step"], np.array([0, 0, 0, 7, 8]))
    np.testing.assert_array_equal(reader.get(Ellipsis)["obs"][3:], np.ones((2, 2), np.float32))
    reader.close()


def test_fresh_text_pieces_match_the_written_defaults(tmp_path):
    path = str(tmp_path / "pq")
    created = ParquetStorage.create(_text_space(), capacity=3, cache_path=path, piece_size=2)
    fresh = created.get(Ellipsis)
    created.close()
    reread = ParquetStorage.load_from(path, _text_space(), capacity=3)  # decodes the files create() wrote
    assert_tree_equal(fresh, reread.get(Ellipsis))
    reread.close()


def test_unpickled_storage_reads_pieces_written_since_create(tmp_path):
    import pickle
    path = str(tmp_path / "pq")
    storage = ParquetStorage.create(_space(), capacity=4, cache_path=path, piece_size=2)
    storage.set(slice(0, 4), _batch(4))
    clone = pickle.loads(pickle.dumps(storage))  # flushes the dirty pieces
    assert_tree_equal(clone.get(Ellipsis), _batch(4))
    storage.close()
    clone.close()


# --- array_encoding ----------------------------------------------------------------------

def _mixed_space() -> DictSpace:
    from unienv_interface.space import BinarySpace, TextSpace
    return DictSpace(B, {
        "image": BoxSpace(B, 0, 255, np.uint8, shape=(4, 5, 3)),
        "vec": BoxSpace(B, -10.0, 10.0, np.float32, shape=(7,)),
        "one": BoxSpace(B, -10.0, 10.0, np.float64, shape=(1,)),
        "mask": BinarySpace(B, shape=(3,)),
        "step": BoxSpace(B, 0, 100, np.int64, shape=()),
        "task": TextSpace(B, max_length=16),
    })


def _mixed_batch(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "image": rng.integers(0, 256, (n, 4, 5, 3), dtype=np.uint8),
        "vec": rng.standard_normal((n, 7)).astype(np.float32),
        "one": rng.standard_normal((n, 1)),
        "mask": rng.random((n, 3)) > 0.5,
        "step": np.arange(n, dtype=np.int64),
        "task": np.array([f"task {i}" for i in range(n)], dtype=object),
    }


def _arrow_types(path) -> dict:
    import os
    import pyarrow.parquet as pq
    schema = pq.read_schema(os.path.join(path, "part-000000.parquet"))
    return {field.name: field.type for field in schema}


@pytest.mark.parametrize("capacity", [7, None], ids=["fixed", "unbounded"])
@pytest.mark.parametrize("array_encoding", ["binary", {"image": "binary", "mask": "binary", "*": "list"}],
                         ids=["all", "per-column"])
def test_binary_array_encoding_round_trip(tmp_path, capacity, array_encoding):
    import pyarrow as pa
    path = str(tmp_path / "pq")
    data = _mixed_batch(7)
    storage = ParquetStorage.create(_mixed_space(), capacity=capacity, cache_path=path, piece_size=3,
                                    array_encoding=array_encoding)
    if capacity is None:
        storage.extend_length(7)
    storage.set(slice(None), data)
    assert_tree_equal(storage.get(Ellipsis), data)
    storage.dumps(path)
    storage.close()

    types = _arrow_types(path)
    binary = {"image", "vec", "one", "mask"} if array_encoding == "binary" else {"image", "mask"}
    for name in ("image", "vec", "one", "mask"):
        assert pa.types.is_fixed_size_binary(types[name]) == (name in binary), (name, types[name])
    assert types["step"] == pa.int64() and pa.types.is_string(types["task"])

    for read_only in (True, False):
        loaded = ParquetStorage.load_from(path, _mixed_space(), capacity=capacity, read_only=read_only)
        assert_tree_equal(loaded.get(Ellipsis), data)
        index = np.array([5, 0, 6, 0])  # fancy, unsorted, repeated: the read-only `take` path
        assert_tree_equal(loaded.get(index), {k: v[index] for k, v in data.items()})
        if not read_only:
            loaded.set(2, {k: v[0] for k, v in data.items()})
            np.testing.assert_array_equal(loaded.get(2)["image"], data["image"][0])
        loaded.close()


def test_default_array_encoding_writes_lists_and_no_metadata_key(tmp_path):
    import json
    import os
    import pyarrow as pa
    path = str(tmp_path / "pq")
    storage = ParquetStorage.create(_mixed_space(), capacity=3, cache_path=path, piece_size=3)
    storage.set(slice(None), _mixed_batch(3))
    storage.dumps(path)
    storage.close()
    assert pa.types.is_fixed_size_list(_arrow_types(path)["image"])
    with open(os.path.join(path, "_metadata.json")) as f:
        assert "array_encoding" not in json.load(f)


def test_binary_array_encoding_is_recorded_and_kept_by_dumps_elsewhere(tmp_path):
    import json
    import os
    import pyarrow as pa
    src, dst = str(tmp_path / "a"), str(tmp_path / "b")
    storage = ParquetStorage.create(_mixed_space(), capacity=None, cache_path=src, piece_size=2,
                                    array_encoding={"image": "binary"})
    storage.extend_length(5)
    storage.set(slice(None), _mixed_batch(5))
    storage.dumps(dst)
    storage.close()
    with open(os.path.join(dst, "_metadata.json")) as f:
        assert json.load(f)["array_encoding"] == {"image": "binary", "vec": "list", "one": "list", "mask": "list"}
    reopened = ParquetStorage.load_from(dst, _mixed_space(), capacity=None, read_only=False)
    reopened.extend_length(1)
    reopened.set(5, {k: v[0] for k, v in _mixed_batch(1, seed=1).items()})
    reopened.close()  # writes the grown storage with the recorded encodings
    assert pa.types.is_fixed_size_binary(_arrow_types(dst)["image"])
    assert pa.types.is_fixed_size_list(_arrow_types(dst)["vec"])
    final = ParquetStorage.load_from(dst, _mixed_space(), capacity=None)
    assert len(final) == 6
    np.testing.assert_array_equal(final.get(5)["image"], _mixed_batch(1, seed=1)["image"][0])
    final.close()


def test_binary_columns_infer_their_space_from_metadata(tmp_path):
    from unienv_interface.space import BinarySpace
    path = str(tmp_path / "pq")
    storage = ParquetStorage.create(_mixed_space(), capacity=3, cache_path=path, array_encoding="binary")
    storage.set(slice(None), _mixed_batch(3))
    storage.dumps(path)
    storage.close()
    _, _, space = ParquetStorage.build_space_from_parquet_file(path)
    assert space["image"].shape == (4, 5, 3) and space["image"].dtype == np.uint8
    assert space["vec"].shape == (7,) and space["vec"].dtype == np.float32
    assert isinstance(space["mask"], BinarySpace) and space["mask"].shape == (3,)
    rb = ParquetStorage.load_replay_buffer_from_raw_parquet(path)
    assert_tree_equal(rb.get_at(slice(None)), _mixed_batch(3))


def test_binary_array_encoding_with_per_column_compression(tmp_path):
    path = str(tmp_path / "pq")
    storage = ParquetStorage.create(_mixed_space(), capacity=4, cache_path=path, array_encoding={"image": "binary"},
                                    compression={"image": "zstd", "vec": "gzip"})
    storage.set(slice(None), _mixed_batch(4))
    storage.dumps(path)
    storage.close()
    reader = ParquetStorage.load_from(path, _mixed_space(), capacity=4)
    assert_tree_equal(reader.get(Ellipsis), _mixed_batch(4))
    reader.close()


@pytest.mark.parametrize("array_encoding", ["bytes", {"image": "binray"}, {"nope": "binary"}])
def test_invalid_array_encoding_raises(tmp_path, array_encoding):
    with pytest.raises(ValueError):
        ParquetStorage.create(_mixed_space(), capacity=2, cache_path=str(tmp_path / "pq"), array_encoding=array_encoding)


def test_replay_buffer_passes_array_encoding_through(tmp_path):
    from unienv_data.replay_buffer import ReplayBuffer
    import pyarrow as pa
    path = str(tmp_path / "rb")
    rb = ReplayBuffer.create(ParquetStorage, _mixed_space(), cache_path=path, capacity=None, array_encoding="binary")
    data = _mixed_batch(4)
    rb.extend(data)
    rb.dumps(path)
    rb.close()
    loaded = ReplayBuffer.load_from(path, backend=B)
    assert_tree_equal(loaded.get_at(slice(None)), data)
    loaded.close()
