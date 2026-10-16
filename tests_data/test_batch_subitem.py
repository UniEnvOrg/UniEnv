"""SubItemBatch (``BatchBase.get_column``) views."""

import numpy as np
import pytest
import torch

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import BoxSpace, DictSpace, TupleSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_data.batches import CombinedBatch, SubItemBatch

from tests_common.batch_helpers import ArrayBatch, make_rb, numpy_array_batch
from tests_common.trees import assert_tree_equal


def _nested_batch(n=6):
    space = DictSpace(NB, {
        "obs": DictSpace(NB, {
            "pos": BoxSpace(NB, -1e6, 1e6, np.float32, shape=(2,)),
            "vel": BoxSpace(NB, -1e6, 1e6, np.float32, shape=(3,)),
        }),
        "pair": TupleSpace(NB, (
            BoxSpace(NB, -1e6, 1e6, np.float32, shape=()),
            BoxSpace(NB, -1e6, 1e6, np.int64, shape=(2,)),
        )),
    })
    i = np.arange(n)
    data = {
        "obs": {"pos": np.stack([i, -i], -1).astype(np.float32),
                "vel": np.stack([i, i + 1, i + 2], -1).astype(np.float32)},
        "pair": (i.astype(np.float32) / 2, np.stack([i, 2 * i], -1).astype(np.int64)),
    }
    meta_space = DictSpace(NB, {"step": BoxSpace(NB, 0, 1000, np.int64, shape=())})
    return ArrayBatch(space, data, meta_space, {"step": (10 * i).astype(np.int64)})


@pytest.mark.parametrize("keys", [["obs"], ["obs", "pos"], ["pair", 1]], ids=["dict", "nested", "tuple-item"])
@pytest.mark.parametrize("idx", [3, np.array([0, 4]), slice(1, 4), ...], ids=["int", "array", "slice", "ellipsis"])
def test_get_at_matches_parent_item(keys, idx):
    parent = _nested_batch()
    col = parent.get_column(keys)
    assert isinstance(col, SubItemBatch)
    expected = parent.get_at(idx)
    for k in keys:
        expected = expected[k]
    assert_tree_equal(col.get_at(idx), expected)
    assert len(col) == len(parent)
    assert not col.is_mutable


@pytest.mark.parametrize("keys", [["obs"], ["obs", "vel"], ["pair", 0]], ids=["dict", "nested", "tuple-item"])
def test_flattened_access_is_flattened_column(keys):
    # Regression: get_flattened_at indexed the parent's flat array with a key.
    parent = _nested_batch()
    col = parent.get_column(keys)
    idx = np.array([1, 5])
    expected = sfu.flatten_data(col._batched_space, col.get_at(idx), start_dim=1)
    assert_tree_equal(col.get_flattened_at(idx), expected)
    flat, meta = col.get_flattened_at_with_metadata(idx)
    assert_tree_equal(flat, expected)
    assert_tree_equal(meta, {"step": np.array([10, 50])})
    assert_tree_equal(col.get_flattened_at(2), sfu.flatten_data(col.single_space, col.get_at(2)))


def test_with_metadata_returns_column_and_parent_metadata():
    # Regression: get_at_with_metadata indexed the (data, metadata) tuple with a key.
    parent = _nested_batch()
    col = parent.get_column(["obs", "pos"])
    data, meta = col.get_at_with_metadata(np.array([2, 3]))
    assert_tree_equal(data, parent.get_at(np.array([2, 3]))["obs"]["pos"])
    assert_tree_equal(meta, {"step": np.array([20, 30])})
    data, meta = col.get_at_with_metadata(4)
    assert_tree_equal(data, np.array([4, -4], dtype=np.float32))
    assert int(meta["step"]) == 40


def test_metadata_space_is_parent_metadata_space():
    # Metadata is per sample, not per field: the column keeps the parent's metadata space.
    parent = numpy_array_batch(4)
    combined = CombinedBatch([parent, numpy_array_batch(3)])
    col = combined.get_column(["obs"])
    assert set(col.single_metadata_space.keys()) == {"step", "batch_index"}
    data, meta = col.get_at_with_metadata(np.array([0, 5]))
    assert_tree_equal(data, combined.get_at(np.array([0, 5]))["obs"])
    assert_tree_equal(meta["batch_index"], np.array([0, 1]))


def test_replay_buffer_column():
    rb = make_rb(8)
    col = rb.get_column(["obs"])
    assert_tree_equal(col.get_at(torch.tensor([1, 6])), rb.get_at(torch.tensor([1, 6]))["obs"])
    assert col.single_space == rb.single_space["obs"]
    assert col.single_metadata_space is None
    assert col.backend is rb.backend and col.device == rb.device
