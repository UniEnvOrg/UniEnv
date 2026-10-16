"""get_column of SliceStackedBatch / FrameStackedBatch / CombinedBatch.

These views build the column view from their children's column views (so a replay buffer
only gathers that column) instead of wrapping themselves in a ``SubItemBatch``; the result is
another class, but must be indistinguishable from ``SubItemBatch(view, keys)``.
"""

import functools
import itertools

import numpy as np
import pytest
import torch

from unienv_data.batches import CombinedBatch, FrameStackedBatch, SliceStackedBatch, SubItemBatch

from tests_common.batch_helpers import make_rb, numpy_array_batch, rb_data, rb_space
from tests_common.trees import assert_tree_equal

EPISODE_ID_MASK = SliceStackedBatch.get_valid_mask_function_with_episodeid_key("episode_id")
EPISODE_END_MASK = SliceStackedBatch.get_valid_mask_function_with_episode_end_key("episode_end")


def custom_mask(batch, expanded_idx, data, metadata):
    return data["episode_id"] == data["episode_id"][:, batch._idx_zero][:, None]


def dict_storage_rb(n=40):
    from unienv_data import ReplayBuffer
    from unienv_data.storages.dict_storage import DictStorage
    from unienv_data.storages.pytorch import PytorchTensorStorage
    rb = ReplayBuffer.create(DictStorage, rb_space(), {k: PytorchTensorStorage for k in ("obs", "episode_id", "episode_end")},
                             capacity=n, is_memmap=False)
    rb.extend(rb_data(n))
    return rb


SOURCES = {
    "flattened rb": lambda: make_rb(40),
    "dict rb": dict_storage_rb,
    "numpy+metadata": lambda: numpy_array_batch(40),
}
MASKS = {"none": None, "episode_id": EPISODE_ID_MASK, "episode_end": EPISODE_END_MASK}


def _indices(be, n):
    rng = np.random.default_rng(0)
    return [
        be.from_numpy(rng.integers(-n, n, 17)),
        be.from_numpy(rng.random(n) < 0.4),
        slice(3, 30, 4),
        7,
        -1,
    ]


def assert_same_view(view, expected):
    assert view.single_space == expected.single_space
    assert view.single_metadata_space == expected.single_metadata_space
    assert len(view) == len(expected)
    assert not view.is_mutable
    for idx in _indices(view.backend, len(view)):
        assert_tree_equal(view.get_at(idx), expected.get_at(idx))
        assert_tree_equal(view.get_at_with_metadata(idx), expected.get_at_with_metadata(idx))
        assert_tree_equal(view.get_flattened_at(idx), expected.get_flattened_at(idx))
        assert_tree_equal(view.get_flattened_at_with_metadata(idx), expected.get_flattened_at_with_metadata(idx))


@pytest.mark.parametrize("source,mask,offsets,stack_metadata,fill,keys", list(itertools.product(
    SOURCES, MASKS, ["framestack", "sparse"], [False, True], [True, False], [("obs",), ("episode_end",)])))
def test_slicestack_column_matches_subitem_view(source, mask, offsets, stack_metadata, fill, keys):
    src = SOURCES[source]()
    kwargs = dict(get_valid_mask_function=MASKS[mask], stack_metadata=stack_metadata, fill_invalid_data=fill)
    if offsets == "framestack":
        view = FrameStackedBatch(src, prefetch_horizon=2, postfetch_horizon=1, **kwargs)
    else:
        view = SliceStackedBatch(src, src.backend.from_numpy(np.array([-3, 0, 2, 5])), **kwargs)
    column = view.get_column(list(keys))
    # Approved behaviour change: a dedicated column view rather than a SubItemBatch.
    assert not isinstance(column, SubItemBatch)
    assert_same_view(column, SubItemBatch(view, list(keys)))


@pytest.mark.parametrize("mask_fn,split", [
    (custom_mask, False),
    (SliceStackedBatch.get_valid_mask_function_with_episodeid_key("step", is_in_metadata=True), False),
    (functools.partial(SliceStackedBatch._valid_mask_function_episodeid_key, episode_id_key="episode_id"), True),
], ids=["custom", "in-metadata", "hand-made-partial"])
def test_slicestack_column_with_other_masks(mask_fn, split):
    # Only the built-in episode helpers reading a data field can be split off the column.
    view = FrameStackedBatch(numpy_array_batch(40), prefetch_horizon=2, get_valid_mask_function=mask_fn)
    column = view.get_column(["obs"])
    assert isinstance(column, SubItemBatch) != split
    assert_same_view(column, SubItemBatch(view, ["obs"]))


def test_slicestack_column_only_gathers_the_column_and_episode_ids():
    rb = dict_storage_rb()
    view = FrameStackedBatch(rb, prefetch_horizon=2, get_valid_mask_function=EPISODE_ID_MASK)
    column = view.get_column(["obs"])
    rb.get_at = rb.get_at_with_metadata = None  # whole samples are never fetched
    out = column.get_at(torch.tensor([0, 5, 9]))
    assert out.shape == (3, 3, 2)
    assert_tree_equal(out[1, :, 0], torch.tensor([8.0, 8.0, 10.0]))  # frames 3 (episode 0) -> filled with 4


def test_combined_column_matches_subitem_view_and_uses_children_columns():
    children = [FrameStackedBatch(make_rb(20, start=s), prefetch_horizon=1, get_valid_mask_function=EPISODE_ID_MASK)
                for s in (0, 20)]
    for use_thread_pool in (False, True):
        combined = CombinedBatch(children, use_thread_pool=use_thread_pool)
        column = combined.get_column(["obs"])
        assert isinstance(column, CombinedBatch) and not column.is_mutable
        assert all(not isinstance(c, SubItemBatch) for c in column.batches)
        assert_same_view(column, SubItemBatch(combined, ["obs"]))
        with pytest.raises(AssertionError):
            column.set_at(torch.tensor([0]), torch.zeros(1, 2, 2))


def test_combined_column_of_combined_children():
    inner = CombinedBatch([make_rb(10, start=0), make_rb(10, start=10)])
    combined = CombinedBatch([inner, CombinedBatch([make_rb(10, start=20), make_rb(10, start=30)])])
    assert_same_view(combined.get_column(["episode_id"]), SubItemBatch(combined, ["episode_id"]))


def test_combined_column_falls_back_to_a_subitem_view():
    # Children without a dedicated column view.
    combined = CombinedBatch([numpy_array_batch(10), numpy_array_batch(10, start=10)])
    column = combined.get_column(["obs"])
    assert isinstance(column, SubItemBatch) and column.batch is combined
    # Children with metadata, stacking columns of sources without a dedicated column view.
    combined = CombinedBatch([FrameStackedBatch(numpy_array_batch(10), 1), FrameStackedBatch(numpy_array_batch(10, start=10), 1)])
    assert_same_view(combined.get_column(["obs"]), SubItemBatch(combined, ["obs"]))


def test_column_views_fall_back_when_child_columns_change_metadata():
    class OtherMetadataColumn(SubItemBatch):
        """A column view that drops the metadata of its batch."""
        def __init__(self, batch, keys):
            super().__init__(batch, keys)
            self.single_metadata_space = None

        def get_at_with_metadata(self, idx):
            return self.get_at(idx), None

    class Source(type(numpy_array_batch(1))):
        def get_column(self, nested_keys):
            return OtherMetadataColumn(self, nested_keys)

    src = numpy_array_batch(20)
    src.__class__ = Source
    view = FrameStackedBatch(src, 1)
    column = view.get_column(["obs"])
    assert isinstance(column, SubItemBatch) and column.batch is view
    combined = CombinedBatch([src, Source(src.single_space, src.data, src.single_metadata_space, src.metadata)])
    column = combined.get_column(["obs"])
    assert isinstance(column, SubItemBatch) and column.batch is combined
