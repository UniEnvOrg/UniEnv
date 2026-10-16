"""Default implementations of BatchBase / BatchSampler in unienv_data.base.common."""

import numpy as np
import pytest

from unienv_data.base import BatchBase, BatchSampler
from unienv_data.base.common import convert_index_to_backendarray
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.space.space_utils import flatten_utils as sfu

from tests_common.trees import assert_tree_equal

B = NumpyComputeBackend


def _space() -> DictSpace:
    return DictSpace(B, {
        "x": BoxSpace(B, -100.0, 100.0, np.float32, shape=(2,)),
        "y": BoxSpace(B, -100.0, 100.0, np.float32, shape=()),
    })


class FlatArrayBatch(BatchBase):
    """Implements only the flattened primitives; everything else uses the BatchBase defaults."""

    def __init__(self, space):
        super().__init__(space)
        self.data = np.zeros((0, sfu.flatten_space(space).shape[0]), np.float32)

    def __len__(self):
        return self.data.shape[0]

    def get_flattened_at(self, idx):
        return self.data[idx].copy()

    def set_flattened_at(self, idx, value):
        self.data[idx] = value

    def extend_flattened(self, value):
        self.data = np.concatenate([self.data, value], axis=0)


class StructuredListBatch(BatchBase):
    """Implements only the structured primitives."""

    def __init__(self, space, rows):
        super().__init__(space)
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def get_at(self, idx):
        if isinstance(idx, int):
            return self.rows[idx]
        idx = np.arange(len(self.rows))[idx]
        return {
            "x": np.stack([self.rows[i]["x"] for i in idx]),
            "y": np.stack([self.rows[i]["y"] for i in idx]),
        }


def _row(v: float) -> dict:
    return {"x": np.array([v, -v], np.float32), "y": np.float32(v * 10)}


def _rows(values) -> dict:
    return {"x": np.stack([_row(v)["x"] for v in values]), "y": np.stack([_row(v)["y"] for v in values])}


@pytest.fixture
def flat_batch():
    batch = FlatArrayBatch(_space())
    batch.extend(_rows([0.0, 1.0, 2.0]))
    batch.append(_row(3.0))
    return batch


def test_flat_primitives_drive_structured_defaults(flat_batch):
    assert len(flat_batch) == 4
    assert_tree_equal(flat_batch.get_at(1), _row(1.0))
    assert_tree_equal(flat_batch[slice(1, 3)], _rows([1.0, 2.0]))
    assert_tree_equal(flat_batch.get_at(np.array([3, 0])), _rows([3.0, 0.0]))
    flat_batch.set_at(0, _row(9.0))
    flat_batch[slice(1, 3)] = _rows([7.0, 8.0])
    assert_tree_equal(flat_batch.get_at(slice(None)), _rows([9.0, 7.0, 8.0, 3.0]))
    flat_batch.append_flattened(np.array([1.0, 2.0, 3.0], np.float32))
    assert_tree_equal(flat_batch.get_at(-1), {"x": np.array([1.0, 2.0], np.float32), "y": np.float32(3.0)})


def test_with_metadata_defaults_do_not_recurse(flat_batch):
    value, metadata = flat_batch.get_at_with_metadata(2)
    assert_tree_equal(value, _row(2.0))
    assert metadata is None
    flat, metadata = flat_batch.get_flattened_at_with_metadata(slice(0, 2))
    np.testing.assert_array_equal(flat, flat_batch.get_flattened_at(slice(0, 2)))
    assert metadata is None

    structured = StructuredListBatch(_space(), [_row(1.0), _row(2.0)])
    value, metadata = structured.get_at_with_metadata(slice(None))
    assert_tree_equal(value, _rows([1.0, 2.0]))
    assert metadata is None
    flat, metadata = structured.get_flattened_at_with_metadata(1)
    np.testing.assert_array_equal(flat, np.array([2.0, -2.0, 20.0], np.float32))
    assert metadata is None
    np.testing.assert_array_equal(structured.get_flattened_at(slice(None)),
                                  np.array([[1.0, -1.0, 10.0], [2.0, -2.0, 20.0]], np.float32))
