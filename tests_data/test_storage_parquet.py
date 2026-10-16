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
