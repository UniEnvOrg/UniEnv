"""TransformedBatch: lazily transformed views (with optional inverse for writes)."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.space.space_utils import flatten_utils as sfu
from unienv_interface.transformations.dict_transform import DictTransformation
from unienv_interface.transformations.filter_dict import DictExcludeKeyTransformation
from unienv_interface.transformations.rescale import RescaleTransformation
from unienv_data.batches import TransformedBatch

from tests_common.batch_helpers import ArrayBatch, numpy_array_batch
from tests_common.trees import assert_tree_equal


def _box_batch(n=5):
    space = BoxSpace(NB, 0.0, 10.0, np.float32, shape=(2,))
    data = np.stack([np.arange(n), np.arange(n) + 5], -1).astype(np.float32)
    meta_space = DictSpace(NB, {"step": BoxSpace(NB, 0, 100, np.int64, shape=())})
    return ArrayBatch(space, data, meta_space, {"step": np.arange(n, dtype=np.int64)})


def test_rescale_view_reads():
    src = _box_batch()
    tb = TransformedBatch(src, RescaleTransformation(-1.0, 1.0))
    assert len(tb) == 5 and tb.is_mutable
    assert tb.single_space.contains(tb.get_at(0))
    np.testing.assert_allclose(tb.get_at(np.array([0, 4])), np.array([[-1.0, 0.0], [-0.2, 0.8]]), atol=1e-6)
    np.testing.assert_allclose(tb.get_at(1), np.array([-0.8, 0.2]), atol=1e-6)
    data, meta = tb.get_at_with_metadata(np.array([2]))
    np.testing.assert_allclose(data, np.array([[-0.6, 0.4]]), atol=1e-6)
    assert_tree_equal(meta, {"step": np.array([2])})
    flat, meta = tb.get_flattened_at_with_metadata(3)
    assert_tree_equal(flat, sfu.flatten_data(tb.single_space, tb.get_at(3)))
    assert int(meta["step"]) == 3
    assert_tree_equal(tb.get_flattened_at(np.array([1, 2])), tb.get_at(np.array([1, 2])))


def test_rescale_view_writes_through_inverse():
    src = _box_batch()
    tb = TransformedBatch(src, RescaleTransformation(-1.0, 1.0))
    tb.set_at(np.array([0]), np.array([[1.0, -1.0]], np.float32))
    np.testing.assert_allclose(src.get_at(0), np.array([10.0, 0.0]), atol=1e-5)
    tb.set_at(1, np.array([0.0, 0.0], np.float32))
    np.testing.assert_allclose(src.get_at(1), np.array([5.0, 5.0]), atol=1e-5)
    tb.set_flattened_at(np.array([2]), np.array([[-1.0, 1.0]], np.float32))
    np.testing.assert_allclose(src.get_at(2), np.array([0.0, 10.0]), atol=1e-5)
    tb.set_flattened_at(3, np.array([0.5, 0.5], np.float32))
    np.testing.assert_allclose(src.get_at(3), np.array([7.5, 7.5]), atol=1e-5)


def test_extend_applies_inverse_with_target_space():
    # Regression: extend() ran the inverse with the *source* batched space, so the
    # rescale inverse used the wrong bounds and stored target-space values verbatim.
    src = _box_batch()
    tb = TransformedBatch(src, RescaleTransformation(-1.0, 1.0))
    tb.extend(np.array([[1.0, -1.0], [0.0, 0.5]], np.float32))
    assert len(src) == 7
    np.testing.assert_allclose(src.get_at(np.array([5, 6])), np.array([[10.0, 0.0], [5.0, 7.5]]), atol=1e-5)
    tb.extend_flattened(np.array([[0.0, 0.0]], np.float32))
    np.testing.assert_allclose(src.get_at(7), np.array([5.0, 5.0]), atol=1e-5)


def test_metadata_transformation():
    src = numpy_array_batch(4)
    meta_tf = DictTransformation({"step": RescaleTransformation(0.0, 1.0)})
    src.single_metadata_space = DictSpace(NB, {"step": BoxSpace(NB, 0, 100, np.float32, shape=())})
    src.metadata = {"step": src.metadata["step"].astype(np.float32)}
    tb = TransformedBatch(src, DictExcludeKeyTransformation(["episode_end"]), metadata_transformation=meta_tf)
    assert set(tb.single_space.keys()) == {"obs", "episode_id"}
    data, meta = tb.get_at_with_metadata(np.array([2, 3]))
    assert set(data.keys()) == {"obs", "episode_id"}
    np.testing.assert_allclose(meta["step"], np.array([0.2, 0.3]), atol=1e-6)
    _, meta = tb.get_at_with_metadata(1)
    np.testing.assert_allclose(meta["step"], 0.1, atol=1e-6)


def test_without_inverse_is_read_only():
    src = numpy_array_batch(4, with_metadata=False)
    tb = TransformedBatch(src, DictExcludeKeyTransformation(["episode_end"]))
    assert not tb.is_mutable
    with pytest.raises(AssertionError):
        tb.set_at(0, tb.get_at(0))
    with pytest.raises(AssertionError):
        tb.extend(tb.get_at(np.array([0])))
    # Metadata transformations are ignored when the source has no metadata.
    tb2 = TransformedBatch(src, DictExcludeKeyTransformation(["episode_end"]),
                           metadata_transformation=DictExcludeKeyTransformation(["x"]))
    assert tb2.metadata_transformation is None and tb2.single_metadata_space is None


def test_remove_at_forwards():
    src = _box_batch()
    tb = TransformedBatch(src, RescaleTransformation(-1.0, 1.0))
    tb.remove_at(np.array([0, 1]))
    assert len(src) == 3 and len(tb) == 3
