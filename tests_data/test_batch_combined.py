"""CombinedBatch: concatenation of several batches."""

import numpy as np
import pytest
import torch

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_data.batches import CombinedBatch
from unienv_data.batches.combined_batch import convert_index_to_batch, convert_single_index_to_batch

from tests_common.batch_helpers import make_rb, numpy_array_batch, rb_data
from tests_common.trees import assert_tree_equal

SIZES = (4, 3, 5)


def _combined(sizes=SIZES, **kwargs):
    starts = np.cumsum((0,) + tuple(sizes))[:-1]
    children = [make_rb(n, capacity=max(n, 1) + 4, start=int(s)) for n, s in zip(sizes, starts)]
    return CombinedBatch(children, **kwargs), children


def _reference(n):
    """What the concatenation of ``rb_data`` chunks looks like."""
    return rb_data(n)


def _ref_at(n, idx):
    return sbu.get_at(sbu.batch_space(make_rb(1).single_space, n), _reference(n), idx)


@pytest.mark.parametrize("use_thread_pool", [False, True], ids=["serial", "threads"])
@pytest.mark.parametrize("idx", [
    torch.tensor([0, 3, 4, 6, 7, 11]),
    torch.tensor([11, 0, 7, 4]),
    torch.tensor([-1, -12, -5]),
    slice(2, 10, 3),
    ...,
    torch.tensor([i % 3 == 0 for i in range(12)]),
], ids=["sorted", "unsorted", "negative", "slice", "ellipsis", "bool-mask"])
def test_get_at_across_children(idx, use_thread_pool):
    cb, _ = _combined(use_thread_pool=use_thread_pool)
    total = sum(SIZES)
    assert len(cb) == total and isinstance(len(cb), int)
    expected = _ref_at(total, idx)
    assert_tree_equal(cb.get_at(idx), expected)
    assert_tree_equal(cb.get_flattened_at(idx),
                      sfu.flatten_data(sbu.batch_space(cb.single_space, 1), expected, start_dim=1))
    data, meta = cb.get_at_with_metadata(idx)
    flat, flat_meta = cb.get_flattened_at_with_metadata(idx)
    assert_tree_equal(data, expected)
    assert_tree_equal(flat, cb.get_flattened_at(idx))
    child_of = torch.repeat_interleave(torch.arange(len(SIZES)), torch.tensor(SIZES))
    assert_tree_equal(meta["batch_index"], child_of[idx], check_dtype=False)
    assert_tree_equal(flat_meta["batch_index"], meta["batch_index"])


@pytest.mark.parametrize("i", [0, 3, 4, 6, 7, 11, -1, -12])
def test_int_index_and_batch_index_metadata(i):
    cb, _ = _combined()
    expected = _ref_at(sum(SIZES), i)
    assert_tree_equal(cb.get_at(i), expected)
    data, meta = cb.get_at_with_metadata(i)
    assert_tree_equal(data, expected)
    child = [0] * 4 + [1] * 3 + [2] * 5
    assert int(meta["batch_index"]) == child[i]
    flat, flat_meta = cb.get_flattened_at_with_metadata(i)
    assert_tree_equal(flat, sfu.flatten_data(cb.single_space, expected))
    assert int(flat_meta["batch_index"]) == child[i]
    assert_tree_equal(cb.get_flattened_at(i), flat)


@pytest.mark.parametrize("i", [12, -13])
def test_out_of_range_int_raises(i):
    cb, _ = _combined()
    with pytest.raises(AssertionError):
        cb.get_at(i)
    with pytest.raises(AssertionError):
        cb.get_at(torch.tensor([i]))


@pytest.mark.parametrize("sizes", [(0, 3, 2), (3, 0, 2), (2, 3, 0)], ids=["first-empty", "middle-empty", "last-empty"])
def test_empty_children(sizes):
    cb, _ = _combined(sizes)
    total = sum(sizes)
    assert len(cb) == total
    assert_tree_equal(cb.get_at(...), _ref_at(total, ...))
    for i in range(total):
        assert_tree_equal(cb.get_at(i), _ref_at(total, i))


def test_thread_pool_matches_serial_with_metadata_children():
    children = [numpy_array_batch(4), numpy_array_batch(5, start=4), numpy_array_batch(2, start=9)]
    serial = CombinedBatch(children)
    threaded = CombinedBatch(children, use_thread_pool=True)
    idx = np.array([10, 0, 5, 3, 8])
    for getter in ("get_at", "get_flattened_at", "get_at_with_metadata", "get_flattened_at_with_metadata"):
        assert_tree_equal(getattr(threaded, getter)(idx), getattr(serial, getter)(idx))
    data, meta = serial.get_at_with_metadata(idx)
    assert_tree_equal(meta["step"], idx * 10)
    assert_tree_equal(meta["batch_index"], np.array([2, 0, 1, 0, 1]), check_dtype=False)
    _, meta = serial.get_at_with_metadata(5)
    assert int(meta["step"]) == 50 and int(meta["batch_index"]) == 1
    assert set(serial.single_metadata_space.keys()) == {"step", "batch_index"}


def test_extend_goes_to_last_child():
    cb, children = _combined()
    cb.extend(rb_data(2, start=12))
    assert [len(c) for c in children] == [4, 3, 7]
    assert len(cb) == 14
    assert_tree_equal(cb.get_at(slice(12, 14)), rb_data(2, start=12))
    cb.extend_flattened(cb.get_flattened_at(torch.tensor([0])))
    assert len(children[-1]) == 8 and len(cb) == 15
    assert_tree_equal(cb.get_at(14), cb.get_at(0))


def test_set_at_with_structured_value():
    # Regression: the size check used len(value), i.e. the number of dict keys.
    cb, children = _combined()
    idx = torch.tensor([0, 5, 11, 6])
    value = rb_data(4, start=100)
    cb.set_at(idx, value)
    assert_tree_equal(cb.get_at(idx), value)
    assert_tree_equal(children[2].get_at(4), sbu.get_at(sbu.batch_space(cb.single_space, 4), value, 2))
    with pytest.raises(AssertionError):
        cb.set_at(idx, rb_data(3))


def test_set_int_and_flattened():
    cb, children = _combined()
    value = _ref_at(sum(SIZES), 0)
    cb.set_at(9, value)
    assert_tree_equal(children[2].get_at(2), value)
    flat = cb.get_flattened_at(torch.tensor([1, 2]))
    cb.set_flattened_at(torch.tensor([5, 8]), flat)
    assert_tree_equal(cb.get_flattened_at(torch.tensor([5, 8])), flat)
    cb.set_flattened_at(3, flat[0])
    assert_tree_equal(cb.get_flattened_at(3), flat[0])
    with pytest.raises(AssertionError):
        cb.set_flattened_at(torch.tensor([5, 8, 9]), flat)


def test_remove_at_updates_index():
    children = [numpy_array_batch(4), numpy_array_batch(3, start=4)]
    cb = CombinedBatch(children)
    cb.remove_at(np.array([1, 5]))
    assert [len(c) for c in children] == [3, 2] and len(cb) == 5
    assert_tree_equal(cb.get_at(...)["obs"][:, 0], np.array([0, 4, 6, 8, 12], dtype=np.float32))
    cb.remove_at(0)
    assert len(cb) == 4
    assert_tree_equal(cb.get_at(0)["obs"], np.array([4, 5], dtype=np.float32))


def test_immutable_child_makes_combined_immutable():
    a, b = numpy_array_batch(2), numpy_array_batch(2)
    b.is_mutable = False
    cb = CombinedBatch([a, b])
    assert not cb.is_mutable
    with pytest.raises(AssertionError):
        cb.extend(a.get_at(slice(0, 1)))
    with pytest.raises(AssertionError):
        cb.set_at(0, a.get_at(0))


def test_constructor_validation():
    with pytest.raises(AssertionError):
        CombinedBatch([make_rb(2)])
    with pytest.raises(AssertionError):
        CombinedBatch([make_rb(2), numpy_array_batch(2, with_metadata=False)])
    with pytest.raises(AssertionError):
        CombinedBatch([numpy_array_batch(2), numpy_array_batch(2, with_metadata=False)])


def test_index_conversion_helpers():
    from unienv_interface.backends.numpy import NumpyComputeBackend as NB
    starts = np.array([0, 4, 4, 7])
    assert convert_single_index_to_batch(NB, starts, 9, 4) == (2, 0)
    assert convert_single_index_to_batch(NB, starts, 9, -1) == (3, 1)
    n, groups = convert_index_to_batch(NB, starts, 9, np.array([8, 0, 5]))
    assert n == 3
    assert [g[0] for g in groups] == [0, 2, 3]
    with pytest.raises(ValueError):
        convert_index_to_batch(NB, starts, 9, 3)


def _all_reads(combined, idx):
    return (combined.get_at(idx), combined.get_at_with_metadata(idx),
            combined.get_flattened_at(idx), combined.get_flattened_at_with_metadata(idx))


def test_thread_pool_is_reused_and_shut_down_on_close():
    combined, _ = _combined(use_thread_pool=True)
    serial, _ = _combined(use_thread_pool=False)
    idx = torch.tensor([0, 5, 11, 3, 7])
    for _ in range(3):
        assert_tree_equal(_all_reads(combined, idx), _all_reads(serial, idx))
    executor = combined._executor
    assert executor is not None and executor._max_workers == len(SIZES)
    combined.get_at(idx)
    assert combined._executor is executor
    combined.close()
    assert combined._executor is None and executor._shutdown
    assert_tree_equal(combined.get_at(idx), serial.get_at(idx))  # a closed view still reads
    combined.close()


def test_thread_pool_is_not_pickled_and_is_recreated_after_fork():
    import multiprocessing as mp
    import pickle
    combined, _ = _combined(use_thread_pool=True)
    idx = torch.tensor([0, 5, 11])
    expected = combined.get_at(idx)
    assert combined._executor is not None
    clone = pickle.loads(pickle.dumps(combined))
    assert clone._executor is None
    assert_tree_equal(clone.get_at(idx), expected)
    clone.close()

    def child(queue):
        # The inherited pool has no threads in this process: work submitted to it never runs.
        queue.put({k: v.numpy() for k, v in combined.get_at(idx).items()})

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=child, args=(queue,), daemon=True)
    proc.start()
    try:
        assert_tree_equal(queue.get(timeout=30), expected)
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
    combined.close()


def test_batched_spaces_are_memoised_per_batch_size():
    combined, _ = _combined()
    idx = torch.tensor([0, 5, 11])
    assert_tree_equal(combined.get_at(idx), _ref_at(12, idx))
    space = combined._cached_batch_space(combined.single_space, 3)
    assert space == sbu.batch_space(combined.single_space, 3)
    assert combined._cached_batch_space(combined.single_space, 3) is space
    assert combined._cached_batch_space(combined.single_space, 4) == sbu.batch_space(combined.single_space, 4)
    for n in range(1, 40):
        combined._cached_batch_space(combined.single_space, n)
    assert len(combined._batch_space_cache) <= 16


def test_view_pickled_without_pool_attributes_still_reads():
    import pickle
    combined, _ = _combined(use_thread_pool=True)
    state = combined.__getstate__()
    for name in ("_executor", "_executor_pid", "_batch_space_cache"):
        del state[name]
    old = CombinedBatch.__new__(CombinedBatch)
    old.__setstate__(pickle.loads(pickle.dumps(state)))
    idx = torch.tensor([0, 5, 11])
    assert_tree_equal(old.get_at(idx), combined.get_at(idx))
    old.close()
    combined.close()
