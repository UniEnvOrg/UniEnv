"""ReplayBuffer.get_column views over column-capable storages."""

import numpy as np
import pytest

from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.dict_storage import DictStorage
from unienv_data.storages.hdf5 import HDF5Storage
from unienv_data.storages.parquet import ParquetStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace, DictSpace

B = NumpyComputeBackend


def _space() -> DictSpace:
    return DictSpace(B, {
        "obs": BoxSpace(B, -100.0, 100.0, np.float32, shape=(2,)),
        "rew": BoxSpace(B, -100.0, 100.0, np.float32, shape=()),
    })


def _batch(start: int, n: int) -> dict:
    base = np.arange(start, start + n, dtype=np.float32)
    return {"obs": np.stack([base, base + 0.5], axis=1), "rew": base * 10}


STORAGES = {
    "hdf5": (HDF5Storage, {}),
    "parquet": (ParquetStorage, {"piece_size": 3}),
    "dict-hdf5": (DictStorage, {"storage_cls_map": {"obs": HDF5Storage, "rew": HDF5Storage}}),
}


@pytest.fixture(params=list(STORAGES))
def rb(request, tmp_path):
    cls, kwargs = STORAGES[request.param]
    buffer = ReplayBuffer.create(cls, _space(), cache_path=str(tmp_path / "rb"), capacity=5,
                                 maintain_segment_metadata=True, **kwargs)
    yield buffer
    storage = buffer.storage
    buffer.close()
    storage.close()


def test_column_view_tracks_parent_after_extend_and_wraparound(rb):
    rb.extend(_batch(0, 3))
    column = rb.get_column(["rew"])
    assert len(column) == 3
    np.testing.assert_array_equal(column.get_at(slice(None)), np.array([0.0, 10.0, 20.0], np.float32))

    rb.extend(_batch(3, 4))  # wraps: logical rows are now 2..6, offset 2
    assert rb.offset == 2
    assert len(column) == 5
    np.testing.assert_array_equal(column.get_at(slice(None)), np.arange(2, 7, dtype=np.float32) * 10)
    np.testing.assert_array_equal(column.get_at(-1), np.float32(60.0))
    assert column.get_segments() == rb.get_segments()


def test_column_view_set_writes_through_to_parent(rb):
    rb.extend(_batch(0, 7))
    column = rb.get_column(["obs"])
    column.set_at(0, np.array([-1.0, -2.0], np.float32))
    np.testing.assert_array_equal(rb.get_at(0)["obs"], np.array([-1.0, -2.0], np.float32))


def test_column_view_rejects_structural_writes(rb):
    rb.extend(_batch(0, 2))
    column = rb.get_column(["obs"])
    with pytest.raises(RuntimeError):
        column.extend(np.zeros((1, 2), np.float32))
    with pytest.raises(RuntimeError):
        column.append(np.zeros((2,), np.float32))
    with pytest.raises(RuntimeError):
        column.clear()
    assert len(rb) == 2
