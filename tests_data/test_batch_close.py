"""Ownership semantics of batch views: a view never closes the batch it wraps.

Views are cheap and created freely (``get_slice``, ``get_column``, samplers,
frame stacks...), and ``BatchBase.__del__`` calls ``close()``; if a view closed
its source, garbage-collecting any view would tear down the shared source.
"""

import gc

import pytest
import torch

from unienv_interface.transformations.identity import IdentityTransformation
from unienv_data.batches import (
    CombinedBatch,
    FrameStackedBatch,
    SliceStackedBatch,
    SubIndexedBatch,
    SubItemBatch,
    ToBackendOrDeviceBatch,
    TransformedBatch,
)

from tests_common.batch_helpers import make_rb, numpy_array_batch
from tests_common.trees import assert_tree_equal


class CountingIdentity(IdentityTransformation):
    def __init__(self):
        super().__init__()
        self.closed = 0

    def close(self):
        self.closed += 1


VIEW_FACTORIES = {
    "framestack": lambda b: FrameStackedBatch(b, prefetch_horizon=1),
    "slicestack": lambda b: SliceStackedBatch(b, torch.tensor([0, 1])),
    "subindex": lambda b: SubIndexedBatch(b, slice(0, 2)),
    "subitem": lambda b: SubItemBatch(b, ["obs"]),
    "to-device": lambda b: ToBackendOrDeviceBatch(b, device="cpu"),
    "transformed": lambda b: TransformedBatch(b, CountingIdentity()),
    "combined": lambda b: CombinedBatch([b, b]),
}


@pytest.mark.parametrize("kind", list(VIEW_FACTORIES))
def test_garbage_collected_view_keeps_replay_buffer_usable(kind):
    rb = make_rb(6)
    expected = rb.get_at(torch.tensor([0, 1]))
    view = VIEW_FACTORIES[kind](rb)
    del view
    gc.collect()
    assert rb.storage is not None
    assert_tree_equal(rb.get_at(torch.tensor([0, 1])), expected)


@pytest.mark.parametrize("kind", list(VIEW_FACTORIES))
def test_explicit_close_does_not_close_source(kind):
    src = numpy_array_batch(6, with_metadata=False)
    if kind == "slicestack":
        import numpy as np
        view = SliceStackedBatch(src, np.array([0, 1]))
    else:
        view = VIEW_FACTORIES[kind](src)
    view.close()
    assert src.closed == 0
    del view
    gc.collect()
    assert src.closed == 0


def test_transformed_batch_does_not_close_shared_transformation():
    src = numpy_array_batch(4, with_metadata=False)
    transformation = CountingIdentity()
    first = TransformedBatch(src, transformation)
    second = TransformedBatch(src, transformation)
    first.close()
    del first
    gc.collect()
    assert transformation.closed == 0
    assert_tree_equal(second.get_at(0), src.get_at(0))


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_failed_view_construction_does_not_raise_in_finaliser():
    rb = make_rb(4)
    with pytest.raises(AssertionError):
        SliceStackedBatch(rb, torch.tensor([1, 2]))
    gc.collect()
