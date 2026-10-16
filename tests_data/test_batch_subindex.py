"""SubIndexedBatch (``BatchBase.get_slice``) views."""

import numpy as np
import pytest
import torch

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_data.batches import SubIndexedBatch

from tests_common.batch_helpers import make_rb, numpy_array_batch, rb_data
from tests_common.trees import assert_tree_equal

SUB = [1, 3, 4, 7]


def _bool_mask(n, picks):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[picks] = True
    return mask


@pytest.mark.parametrize("sub_index,expected", [
    (slice(2, 8, 2), [2, 4, 6]),
    (torch.tensor(SUB), SUB),
    (_bool_mask(10, SUB), SUB),
    (..., list(range(10))),
], ids=["slice", "int-array", "bool-mask", "ellipsis"])
def test_get_slice_matches_parent(sub_index, expected):
    rb = make_rb(10)
    view = rb.get_slice(sub_index)
    assert isinstance(view, SubIndexedBatch)
    assert len(view) == len(expected)
    assert_tree_equal(view.get_at(...), rb.get_at(torch.tensor(expected)))
    assert_tree_equal(view.get_at(slice(1, None)), rb.get_at(torch.tensor(expected[1:])))
    assert_tree_equal(view.get_flattened_at(torch.tensor([0, 1])),
                      rb.get_flattened_at(torch.tensor(expected[:2])))


def test_int_index_returns_single_sample():
    # Regression: an int index became a 0-d array, which the replay buffer rejects.
    rb = make_rb(10)
    view = SubIndexedBatch(rb, torch.tensor(SUB))
    for i, parent_i in enumerate(SUB):
        assert_tree_equal(view.get_at(i), rb.get_at(parent_i))
        assert_tree_equal(view[i], rb.get_at(parent_i))
    assert_tree_equal(view.get_at(-1), rb.get_at(SUB[-1]))
    assert_tree_equal(view.get_flattened_at(2), rb.get_flattened_at(SUB[2]))
    data, meta = view.get_at_with_metadata(0)
    assert_tree_equal(data, rb.get_at(SUB[0]))
    flat, _ = view.get_flattened_at_with_metadata(0)
    assert_tree_equal(flat, sfu.flatten_data(rb.single_space, rb.get_at(SUB[0])))


def test_int_set_at_writes_through():
    rb = make_rb(10)
    view = SubIndexedBatch(rb, torch.tensor(SUB))
    value = sbu.get_at(sbu.batch_space(rb.single_space, 1), rb_data(1, start=100), 0)
    view.set_at(1, value)
    assert_tree_equal(rb.get_at(SUB[1]), value)
    flat = sfu.flatten_data(rb.single_space, rb.get_at(0))
    view.set_flattened_at(2, flat)
    assert_tree_equal(rb.get_flattened_at(SUB[2]), flat)


def test_array_set_at_writes_through():
    rb = make_rb(10)
    view = rb.get_slice(slice(5, 9))
    view.set_at(torch.tensor([0, 2]), rb_data(2, start=100))
    assert_tree_equal(rb.get_at(torch.tensor([5, 7])), rb_data(2, start=100))


@pytest.mark.parametrize("sub_index,parent_index", [(3, 3), (-1, 9), (-10, 0)])
def test_int_sub_index_selects_one_sample(sub_index, parent_index):
    rb = make_rb(10)
    view = SubIndexedBatch(rb, sub_index)
    assert len(view) == 1
    assert_tree_equal(view.get_at(0), rb.get_at(parent_index))


def test_extend_requires_ellipsis():
    rb = make_rb(6, capacity=10)
    with pytest.raises(NotImplementedError):
        rb.get_slice(slice(0, 3)).extend(rb_data(1))
    with pytest.raises(NotImplementedError):
        rb.get_slice(slice(0, 3)).extend_flattened(rb.get_flattened_at(torch.tensor([0])))
    full = SubIndexedBatch(rb, ...)
    full.extend(rb_data(1, start=50))
    assert len(rb) == 7 and len(full) == 7
    full.extend_flattened(rb.get_flattened_at(torch.tensor([0])))
    assert len(rb) == 8
    assert_tree_equal(full.get_at(6), rb.get_at(6))
    full.set_at(0, rb.get_at(1))
    full.set_flattened_at(torch.tensor([2]), rb.get_flattened_at(torch.tensor([1])))
    assert_tree_equal(rb.get_at(torch.tensor([0, 2])), rb.get_at(torch.tensor([1, 1])))
    assert_tree_equal(full.get_at_with_metadata(3)[0], rb.get_at(3))
    assert_tree_equal(full.get_flattened_at_with_metadata(torch.tensor([3]))[0],
                      rb.get_flattened_at(torch.tensor([3])))


def test_metadata_is_forwarded():
    src = numpy_array_batch(10)
    view = src.get_slice(np.array(SUB))
    data, meta = view.get_at_with_metadata(np.array([0, 3]))
    assert_tree_equal(meta, {"step": np.array([10, 70])})
    assert view.single_metadata_space is src.single_metadata_space


def test_mutability_follows_parent_and_bad_indices_rejected():
    rb = make_rb(4)
    assert rb.get_slice(slice(0, 2)).is_mutable == rb.is_mutable
    with pytest.raises(AssertionError):
        SubIndexedBatch(rb, torch.zeros((2, 2), dtype=torch.int64))
    with pytest.raises(AssertionError):
        SubIndexedBatch(rb, torch.tensor([0.5]))


@pytest.mark.xfail(strict=True, reason="cross-scope: unienv_data/replay_buffer/replay_buffer.py:498 "
                   "ReplayBuffer.get_flattened_at ignores the ring offset once the buffer wraps")
def test_flattened_view_over_wrapped_replay_buffer():
    rb = make_rb(5, capacity=5)
    rb.extend(rb_data(2, start=5))  # wraps: logical index 0 is sample 2
    view = rb.get_slice(slice(0, 3))
    assert_tree_equal(view.get_flattened_at(torch.tensor([0, 2])),
                      sfu.flatten_data(sbu.batch_space(rb.single_space, 2), view.get_at(torch.tensor([0, 2])), start_dim=1))
