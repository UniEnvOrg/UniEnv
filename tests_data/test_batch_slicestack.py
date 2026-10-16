"""SliceStackedBatch / FrameStackedBatch views."""

import gc

import numpy as np
import pytest
import torch

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_data.batches import FrameStackedBatch, SliceStackedBatch

from tests_common.batch_helpers import make_rb, numpy_array_batch
from tests_common.trees import assert_tree_equal

EPISODE_ID_MASK = SliceStackedBatch.get_valid_mask_function_with_episodeid_key("episode_id")
EPISODE_END_MASK = SliceStackedBatch.get_valid_mask_function_with_episode_end_key("episode_end")


def _obs_rows(frames):
    """Expected ``obs`` for a list of source frame indices (see ``rb_data``)."""
    return torch.tensor([[2 * f, 2 * f + 1] for f in frames], dtype=torch.float32)


def test_framestack_shapes_and_edge_clipping():
    rb = make_rb(10)
    fs = FrameStackedBatch(rb, prefetch_horizon=2)
    assert len(fs) == 10
    assert not fs.is_mutable
    out = fs.get_at(torch.tensor([0, 5, 9]))
    assert out["obs"].shape == (3, 3, 2)
    assert out["episode_id"].shape == (3, 3)
    # Offsets reaching before index 0 are clipped to the first frame.
    assert_tree_equal(out["obs"][0], _obs_rows([0, 0, 0]))
    assert_tree_equal(out["obs"][1], _obs_rows([3, 4, 5]))
    assert_tree_equal(out["obs"][2], _obs_rows([7, 8, 9]))
    assert fs.single_space.contains(fs.get_at(4))


def test_postfetch_clipped_at_end():
    rb = make_rb(10)
    fs = FrameStackedBatch(rb, prefetch_horizon=1, postfetch_horizon=2)
    out = fs.get_at(slice(8, 10))
    assert_tree_equal(out["obs"][0], _obs_rows([7, 8, 9, 9]))
    assert_tree_equal(out["obs"][1], _obs_rows([8, 9, 9, 9]))


@pytest.mark.parametrize("idx", [slice(0, 10), ..., torch.arange(10), torch.ones(10, dtype=torch.bool)],
                         ids=["slice", "ellipsis", "int-array", "bool-mask"])
def test_index_kinds_agree(idx):
    rb = make_rb(10)
    fs = FrameStackedBatch(rb, prefetch_horizon=1)
    expected = fs.get_at(torch.arange(10))
    assert_tree_equal(fs.get_at(idx), expected)


@pytest.mark.parametrize("mask_fn", [EPISODE_ID_MASK, EPISODE_END_MASK], ids=["episode_id", "episode_end"])
def test_episode_boundary_mask_fills_invalid_frames(mask_fn):
    rb = make_rb(10)
    fs = FrameStackedBatch(rb, prefetch_horizon=2, get_valid_mask_function=mask_fn)
    out, meta = fs.get_at_with_metadata(torch.tensor([5, 2]))
    # Index 5 is the 2nd step of episode 1 (steps 4..7): frame 3 belongs to episode 0.
    assert_tree_equal(meta["slice_valid_mask"], torch.tensor([[False, True, True], [True, True, True]]))
    assert_tree_equal(out["obs"][0], _obs_rows([4, 4, 5]))
    assert_tree_equal(out["obs"][1], _obs_rows([0, 1, 2]))


@pytest.mark.parametrize("mask_fn", [EPISODE_ID_MASK, EPISODE_END_MASK], ids=["episode_id", "episode_end"])
def test_episode_boundary_mask_postfetch(mask_fn):
    rb = make_rb(10)
    fs = FrameStackedBatch(rb, prefetch_horizon=1, postfetch_horizon=2, get_valid_mask_function=mask_fn)
    out, meta = fs.get_at_with_metadata(torch.tensor([2]))
    assert_tree_equal(meta["slice_valid_mask"], torch.tensor([[True, True, True, False]]))
    assert_tree_equal(out["obs"][0], _obs_rows([1, 2, 3, 3]))


def test_unfilled_invalid_frames_are_left_untouched():
    rb = make_rb(10)
    fs = FrameStackedBatch(rb, prefetch_horizon=2, get_valid_mask_function=EPISODE_ID_MASK,
                           fill_invalid_data=False)
    out, meta = fs.get_at_with_metadata(torch.tensor([5]))
    assert_tree_equal(meta["slice_valid_mask"], torch.tensor([[False, True, True]]))
    assert_tree_equal(out["obs"][0], _obs_rows([3, 4, 5]))


def test_custom_offsets():
    rb = make_rb(10)
    ss = SliceStackedBatch(rb, torch.tensor([-3, 0, 2]))
    out = ss.get_at(torch.tensor([4]))
    assert_tree_equal(out["obs"][0], _obs_rows([1, 4, 6]))


@pytest.mark.parametrize("offsets", [[0, -1], [1, 2], [-1, 0, 0.5]], ids=["unsorted", "no-zero", "float"])
def test_constructor_rejects_bad_offsets(offsets):
    rb = make_rb(4)
    with pytest.raises(AssertionError):
        SliceStackedBatch(rb, torch.tensor(offsets))


def test_framestack_rejects_zero_horizons():
    with pytest.raises(AssertionError):
        FrameStackedBatch(make_rb(4))


def test_int_index_matches_batched_index():
    rb = make_rb(10)
    fs = FrameStackedBatch(rb, prefetch_horizon=2, get_valid_mask_function=EPISODE_ID_MASK)
    single, single_meta = fs.get_at_with_metadata(5)
    batched, batched_meta = fs.get_at_with_metadata(torch.tensor([5]))
    assert_tree_equal(single, sbu.get_at(fs._batched_space, batched, 0))
    assert_tree_equal(single_meta["slice_valid_mask"], batched_meta["slice_valid_mask"][0])


def test_get_flattened_at_without_source_metadata():
    # Regression: ReplayBuffer metadata is None, so the flattened path crashed
    # assigning ``metadata['slice_valid_mask']``.
    rb = make_rb(10)
    fs = FrameStackedBatch(rb, prefetch_horizon=2, get_valid_mask_function=EPISODE_ID_MASK)
    idx = torch.tensor([0, 5, 9])
    flat, meta = fs.get_flattened_at_with_metadata(idx)
    data, data_meta = fs.get_at_with_metadata(idx)
    # Flattened frame stacks keep the time axis: (B, T, flat_dim_of_one_frame).
    assert_tree_equal(flat, sfu.flatten_data(fs._batched_space, data, start_dim=2))
    assert_tree_equal(meta["slice_valid_mask"], data_meta["slice_valid_mask"])
    assert_tree_equal(fs.get_flattened_at(idx), flat)
    single = fs.get_flattened_at(5)
    assert_tree_equal(single, flat[1])


@pytest.mark.parametrize("flat", [False, True], ids=["structured", "flattened"])
def test_non_stacked_metadata_is_taken_at_offset_zero(flat):
    # Regression: the non-stacked metadata path referenced a never-set
    # ``self.prefetch_horizon`` attribute.
    src = numpy_array_batch(10)
    fs = FrameStackedBatch(src, prefetch_horizon=2, postfetch_horizon=1)
    idx = np.array([1, 5])
    getter = fs.get_flattened_at_with_metadata if flat else fs.get_at_with_metadata
    _, meta = getter(idx)
    assert_tree_equal(meta["step"], np.array([10, 50]))
    assert meta["slice_valid_mask"].shape == (2, 4)
    _, single_meta = (fs.get_flattened_at_with_metadata if flat else fs.get_at_with_metadata)(5)
    assert int(single_meta["step"]) == 50


@pytest.mark.parametrize("flat", [False, True], ids=["structured", "flattened"])
def test_stacked_metadata_is_filled_with_mask(flat):
    src = numpy_array_batch(10)
    fs = FrameStackedBatch(src, prefetch_horizon=2, stack_metadata=True,
                           get_valid_mask_function=EPISODE_ID_MASK)
    getter = fs.get_flattened_at_with_metadata if flat else fs.get_at_with_metadata
    _, meta = getter(np.array([5]))
    assert_tree_equal(meta["step"], np.array([[40, 40, 50]]))
    assert fs.single_metadata_space["step"].shape == (3,)


def test_valid_mask_from_metadata():
    src = numpy_array_batch(10)
    mask_fn = SliceStackedBatch.get_valid_mask_function_with_episodeid_key("step", is_in_metadata=True)
    fs = FrameStackedBatch(src, prefetch_horizon=1, get_valid_mask_function=mask_fn)
    _, meta = fs.get_at_with_metadata(np.array([3]))
    # Every step value differs, so only the anchor frame is "in the same episode".
    assert_tree_equal(meta["slice_valid_mask"], np.array([[False, True]]))


def test_negative_indices_address_from_the_end():
    rb = make_rb(10)
    fs = FrameStackedBatch(rb, prefetch_horizon=1)
    assert_tree_equal(fs.get_at(torch.tensor([-1, -2])), fs.get_at(torch.tensor([9, 8])))
    assert_tree_equal(fs.get_at(-1), fs.get_at(9))


def test_view_over_numpy_source():
    src = numpy_array_batch(10, with_metadata=False)
    fs = FrameStackedBatch(src, prefetch_horizon=1, get_valid_mask_function=EPISODE_END_MASK)
    out, meta = fs.get_at_with_metadata(np.array([4, 6]))
    assert_tree_equal(meta["slice_valid_mask"], np.array([[False, True], [True, True]]))
    assert_tree_equal(out["obs"][0], np.array([[8, 9], [8, 9]], dtype=np.float32))
