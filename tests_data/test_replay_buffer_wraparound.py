"""Round-robin (offset != 0) consistency of every ReplayBuffer access path."""

import numpy as np
import pytest
import torch

from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.flattened import FlattenedStorage
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace


def _wrapped_flattened_rb(capacity: int = 4) -> ReplayBuffer:
    """Capacity-4 buffer holding rows 2..5 of ``arange(12).reshape(6, 2)`` with offset 2."""
    space = BoxSpace(PyTorchComputeBackend, -100.0, 100.0, torch.float32, shape=(2,), device="cpu")
    rb = ReplayBuffer.create(
        FlattenedStorage,
        space,
        inner_storage_cls=PytorchTensorStorage,
        capacity=capacity,
        is_memmap=False,
    )
    rb.extend(torch.arange(6, dtype=torch.float32).reshape(3, 2))
    rb.extend(torch.arange(6, 12, dtype=torch.float32).reshape(3, 2))
    assert rb.offset == 2 and rb.count == 4
    return rb


def test_get_flattened_at_respects_offset_after_wraparound():
    rb = _wrapped_flattened_rb()
    expected = torch.arange(4, 12, dtype=torch.float32).reshape(4, 2)
    for i in range(-4, 4):
        torch.testing.assert_close(rb.get_flattened_at(i), expected[i])
        torch.testing.assert_close(rb.get_flattened_at(i), rb.get_at(i))
        data, metadata = rb.get_flattened_at_with_metadata(i)
        torch.testing.assert_close(data, expected[i])
        assert metadata is None
    torch.testing.assert_close(rb.get_flattened_at(slice(None)), expected)
    torch.testing.assert_close(rb.get_flattened_at(Ellipsis), expected)
    torch.testing.assert_close(rb.get_flattened_at(torch.tensor([3, 0, -1])), expected[[3, 0, 3]])


# ---------------------------------------------------------------------------
# index_with_offset and segment helpers
# ---------------------------------------------------------------------------

import json  # noqa: E402
import pickle  # noqa: E402

from unienv_data.replay_buffer import replay_buffer as rb_module  # noqa: E402
from unienv_data.storages.npz_storage import NPZStorage  # noqa: E402
from unienv_interface.backends.numpy import NumpyComputeBackend  # noqa: E402

NB = NumpyComputeBackend


@pytest.mark.parametrize("index,expected", [
    (0, 2), (3, 1), (-1, 1), (-4, 2),
    (Ellipsis, [2, 3, 0, 1]),
    (slice(1, 3), [3, 0]),
    (slice(None, None, -1), [1, 0, 3, 2]),
    (np.array([0, -1, 2]), [2, 1, 0]),
    (np.array([True, False, False, True]), [2, 1]),
])
def test_index_with_offset(index, expected):
    result = rb_module.index_with_offset(NB, index, 4, 2)
    np.testing.assert_array_equal(result, np.asarray(expected))


@pytest.mark.parametrize("bad", [4, -5, np.array([0, 4]), np.array([[0]]), np.array([0.5]), np.array([True, False])])
def test_index_with_offset_rejects_invalid(bad):
    with pytest.raises(AssertionError):
        rb_module.index_with_offset(NB, bad, 4, 2)


def test_segment_helpers():
    norm = rb_module._normalize_physical_segment
    assert norm(None, capacity=4) is None
    assert norm([1], capacity=4) is None
    assert norm([3, 1], capacity=None) is None
    assert norm([1, 3], capacity=None) == (1, 3)
    assert norm([3, 1], capacity=4) == (3, 1)
    assert norm([0, 4], capacity=4) is None
    assert rb_module._normalize_physical_segments(None, capacity=4) == []
    assert rb_module._normalize_physical_segments([[0, 1], [5, 6], [2]], capacity=4) == [(0, 1)]
    assert rb_module._load_segment_metadata({}, capacity=4) is None
    assert rb_module._load_segment_metadata({"physical_segments": None}, capacity=4) is None
    assert rb_module._expand_physical_segment((3, 1), 4) == [3, 0, 1]
    assert rb_module._compress_physical_indices([], 4) == []
    assert rb_module._compress_physical_indices([2, 3, 0, 1], 4) == [(2, 1)]
    assert rb_module._compress_physical_indices([0, 2, 3], None) == [(0, 0), (2, 3)]
    assert rb_module._segment_length_from_inclusive((3, 1), 4) == 3
    assert rb_module._physical_segments_to_logical_half_open([(4, 5), (0, 3)], offset=0, count=6, capacity=None) \
        == [(0, 4), (4, 6)]
    # a segment that is entirely outside of the logical window disappears; a split one is reported in pieces
    assert rb_module._physical_segments_to_logical_half_open([(0, 0)], offset=2, count=1, capacity=4) == []
    assert rb_module._physical_segments_to_logical_half_open([(1, 3)], offset=2, count=4, capacity=4) \
        == [(0, 2), (3, 4)]


def _npz_rb(tmp_path, **kwargs):
    space = BoxSpace(NB, -100.0, 100.0, np.float32, shape=(2,))
    return ReplayBuffer.create(NPZStorage, space, cache_path=str(tmp_path / "rb"), compressed=False, **kwargs)


def test_metadata_helpers_on_missing_or_foreign_paths(tmp_path):
    assert not ReplayBuffer.is_loadable_from(str(tmp_path))
    assert ReplayBuffer.get_length_from_path(str(tmp_path)) is None
    assert ReplayBuffer.get_capacity_from_path(str(tmp_path)) is None
    assert ReplayBuffer.get_space_from_path(str(tmp_path), backend=NB) is None
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump({"type": "SomethingElse"}, f)
    assert not ReplayBuffer.is_loadable_from(str(tmp_path))
    assert ReplayBuffer.get_length_from_path(str(tmp_path)) is None
    assert ReplayBuffer.get_capacity_from_path(str(tmp_path)) is None
    assert ReplayBuffer.get_space_from_path(str(tmp_path), backend=NB) is None


def test_load_from_validates_and_reuses_provided_space(tmp_path):
    rb = _npz_rb(tmp_path, capacity=3, maintain_segment_metadata=True)
    rb.extend(np.ones((2, 2), np.float32))
    rb.dumps(rb.cache_path)
    space = BoxSpace(NB, -100.0, 100.0, np.float32, shape=(2,))
    loaded = ReplayBuffer.load_from(rb.cache_path, backend=NB, single_instance_space=space)
    assert loaded.single_space is space
    with pytest.raises(AssertionError):
        ReplayBuffer.load_from(rb.cache_path, backend=NB,
                               single_instance_space=BoxSpace(NB, -1.0, 1.0, np.float32, shape=(2,)))
    with pytest.raises(RuntimeError):
        ReplayBuffer.load_from(rb.cache_path, backend=NB, read_only=False, multiprocessing=True)


def test_constructor_rejects_mutable_multiprocessing_with_segments(tmp_path):
    rb = _npz_rb(tmp_path, capacity=3)
    with pytest.raises(RuntimeError):
        ReplayBuffer(rb.storage, "storage", multiprocessing=True, maintain_segment_metadata=True)


def test_constructor_restores_active_segment(tmp_path):
    rb = _npz_rb(tmp_path, capacity=3)
    restored = ReplayBuffer(rb.storage, "storage", 2, 0, maintain_segment_metadata=True,
                            physical_segments=[(0, 0)], active_segment_start_physical=4, active_segment_length=1)
    assert restored._active_segment_start_physical == 1
    restored.mark_segment_end()
    assert restored.get_segments() == [(0, 1), (1, 2)]


def test_extend_flattened_and_set_flattened_without_storage_support(tmp_path):
    rb = _npz_rb(tmp_path, capacity=3)
    rb.extend_flattened(np.arange(8, dtype=np.float32).reshape(4, 2))
    np.testing.assert_array_equal(rb.get_at(slice(None)), np.arange(2, 8, dtype=np.float32).reshape(3, 2))
    rb.set_flattened_at(0, np.array([9.0, 9.0], np.float32))
    rb.set_flattened_at(slice(1, 3), np.zeros((2, 2), np.float32))
    np.testing.assert_array_equal(rb.get_flattened_at(slice(None)), np.array([[9, 9], [0, 0], [0, 0]], np.float32))
    np.testing.assert_array_equal(rb.get_flattened_at(0), np.array([9.0, 9.0], np.float32))


def test_extend_with_empty_batch_is_a_no_op(tmp_path):
    rb = _npz_rb(tmp_path, capacity=3)
    rb.extend(np.zeros((0, 2), np.float32))
    assert len(rb) == 0


def test_unbounded_segments_and_eviction_bookkeeping(tmp_path):
    rb = _npz_rb(tmp_path, capacity=None, maintain_segment_metadata=True)
    rb.extend(np.zeros((2, 2), np.float32))
    rb.append(np.ones(2, np.float32))
    rb.append(np.ones(2, np.float32))
    rb.mark_segment_end()
    assert rb.get_segments() == [(0, 2), (2, 4)]
    rb._remove_physical_range_from_metadata(1, 2)
    assert rb.get_segments() == [(0, 1), (3, 4)]


def test_open_segment_is_clipped_by_eviction(tmp_path):
    rb = _npz_rb(tmp_path, capacity=3, maintain_segment_metadata=True)
    for v in range(5):
        rb.append(np.full(2, v, np.float32))
    # the open segment has been running for longer than the capacity
    assert rb._active_segment_length == 3
    rb.mark_segment_end()
    assert rb.get_segments() == [(0, 3)]
    np.testing.assert_array_equal(rb.get_at(slice(None))[:, 0], np.array([2.0, 3.0, 4.0], np.float32))


class _FailingAppendStorage(NPZStorage):
    def append(self, value):
        raise ValueError("boom")


@pytest.mark.parametrize("capacity", [None, 3])
def test_failed_append_rolls_back(tmp_path, capacity):
    space = BoxSpace(NB, -100.0, 100.0, np.float32, shape=(2,))
    (tmp_path / "rb").mkdir()
    storage = _FailingAppendStorage(space, compressed=False, cache_filename=str(tmp_path / "rb"), capacity=capacity)
    rb = ReplayBuffer(storage, "storage")
    with pytest.raises(ValueError, match="boom"):
        rb.append(np.zeros(2, np.float32))
    assert len(rb) == 0
    assert not rb.storage.has_open_segment
    assert len(rb.storage) == (0 if capacity is None else capacity)


def test_get_column_falls_back_to_sub_item_batch(tmp_path):
    from unienv_data.batches.subitem_batch import SubItemBatch
    from unienv_interface.space import DictSpace

    space = DictSpace(PyTorchComputeBackend, {
        "a": BoxSpace(PyTorchComputeBackend, -1.0, 1.0, torch.float32, shape=(2,), device="cpu"),
    }, device="cpu")
    rb = ReplayBuffer.create(FlattenedStorage, space, inner_storage_cls=PytorchTensorStorage, capacity=2,
                             is_memmap=False)
    rb.extend({"a": torch.ones(1, 2)})
    column = rb.get_column(["a"])
    assert isinstance(column, SubItemBatch)
    torch.testing.assert_close(column.get_at(slice(None)), torch.ones(1, 2))


def test_column_view_counters_survive_pickling(tmp_path):
    from unienv_data.storages.hdf5 import HDF5Storage
    from unienv_interface.space import DictSpace

    space = DictSpace(NB, {"a": BoxSpace(NB, -100.0, 100.0, np.float32, shape=())})
    rb = ReplayBuffer.create(HDF5Storage, space, capacity=3, cache_path=str(tmp_path / "rb"))
    rb.extend({"a": np.arange(4, dtype=np.float32)})
    column = rb.get_column(["a"])
    rb.dumps(rb.cache_path)
    state = pickle.loads(pickle.dumps(rb._count_value))
    assert state.value == 3
    np.testing.assert_array_equal(column.get_at(slice(None)), np.array([1.0, 2.0, 3.0], np.float32))
