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
