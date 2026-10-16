"""
ReplayBuffer segment bookkeeping (maintain_segment_metadata=True) works on intervals; these tests check it
against the original per-index implementation (kept below as the reference) on randomized operation sequences.
"""

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch

from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.replay_buffer import replay_buffer as rb_module
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace


# ---------------------------------------------------------------------------
# Reference: the per-index implementation the interval arithmetic replaced
# ---------------------------------------------------------------------------

_expand = rb_module._expand_physical_segment
_compress = rb_module._compress_physical_indices


def _reference_logical_half_open(
    physical_segments: Sequence[Tuple[int, int]], *, offset: int, count: int, capacity: Optional[int],
) -> List[Tuple[int, int]]:
    if capacity is None:
        return sorted(((int(start), int(end) + 1) for start, end in physical_segments), key=lambda item: item[0])
    logical_segments = []
    for segment in physical_segments:
        logical_positions = []
        for physical_index in _expand(segment, capacity):
            logical_index = int((physical_index - offset) % capacity)
            if logical_index < count:
                logical_positions.append(logical_index)
        if not logical_positions:
            continue
        logical_positions.sort()
        run_start = run_end = logical_positions[0]
        for logical_index in logical_positions[1:]:
            if logical_index != run_end + 1:
                logical_segments.append((run_start, run_end + 1))
                run_start = logical_index
            run_end = logical_index
        logical_segments.append((run_start, run_end + 1))
    logical_segments.sort(key=lambda item: item[0])
    return logical_segments


class _ReferenceReplayBuffer(ReplayBuffer):
    """ReplayBuffer with the original per-index metadata removal and logical mapping."""

    def _remove_physical_range_from_metadata(self, start_physical, length):
        if not self._maintain_segment_metadata or length <= 0:
            return
        if self.capacity is None:
            victim_indices = set(range(int(start_physical), int(start_physical) + int(length)))
        else:
            end_physical = (int(start_physical) + int(length) - 1) % self.capacity
            victim_indices = set(_expand((int(start_physical), end_physical), self.capacity))
        updated_segments = []
        for segment in self._physical_segments:
            surviving = [index for index in _expand(segment, self.capacity) if index not in victim_indices]
            updated_segments.extend(_compress(surviving, self.capacity))
        self._physical_segments = updated_segments

        active_indices = []
        if self._active_segment_start_physical is not None and self._active_segment_length > 0:
            start, length = int(self._active_segment_start_physical), int(self._active_segment_length)
            if self.capacity is None:
                active_indices = list(range(start, start + length))
            else:
                active_indices = _expand((start, (start + length - 1) % self.capacity), self.capacity)
        active_indices = [index for index in active_indices if index not in victim_indices]
        if not active_indices:
            self._active_segment_start_physical = None
            self._active_segment_length = 0
            return
        active_run = _compress(active_indices, self.capacity)[-1]
        self._active_segment_start_physical = int(active_run[0])
        self._active_segment_length = rb_module._segment_length_from_inclusive(active_run, self.capacity)

    def get_segments(self):
        count, offset = self._ring_state.read()
        return _reference_logical_half_open(self._physical_segments, offset=offset, count=count, capacity=self.capacity)


class _RecordingStorage:
    """Minimal storage: records the indices of `set()` and tracks an open append-built segment."""
    backend = NumpyComputeBackend
    device = None
    is_mutable = True
    is_multiprocessing_safe = False
    pending_segment_length = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.single_instance_space = BoxSpace(NumpyComputeBackend, -1.0, 1.0, np.float32, shape=(1,))
        self.has_open_segment = False
        self.set_indices = []

    def mark_segment_start(self, index):
        self.has_open_segment = True

    def append(self, value):
        pass

    def mark_segment_end(self):
        self.has_open_segment = False

    def abort_segment(self):
        self.has_open_segment = False

    def set(self, index, value):
        self.set_indices.append(index)

    def extend_length(self, length):
        pass

    def clear(self):
        self.has_open_segment = False

    def get_segments(self):
        return None


def _state(rb):
    return (
        list(rb._physical_segments), rb._active_segment_start_physical, rb._active_segment_length,
        rb._ring_state.read(), rb.get_segments(),
    )


@pytest.mark.parametrize("capacity", [None, 1, 2, 3, 5, 7, 16, 50])
def test_interval_bookkeeping_matches_per_index_reference(capacity):
    rng = random.Random(capacity or 0)
    checks = 0
    for _ in range(60):
        new = ReplayBuffer(_RecordingStorage(capacity), "s", maintain_segment_metadata=True)
        ref = _ReferenceReplayBuffer(_RecordingStorage(capacity), "s", maintain_segment_metadata=True)
        for _ in range(rng.randint(1, 150)):
            op = rng.random()
            if op < 0.55:
                ops = [lambda rb: rb.append(np.zeros(1, np.float32))]
            elif op < 0.75:
                ops = [lambda rb: rb.mark_segment_end()]
            elif op < 0.97:
                size = rng.randint(1, (capacity or 10) + 3)
                ops = [lambda rb: rb.mark_segment_end(), lambda rb, size=size: rb.extend(np.zeros((size, 1), np.float32))]
            else:
                ops = [lambda rb: rb.clear()]
            for fn in ops:
                fn(new)
                fn(ref)
            assert _state(new) == _state(ref)
            checks += 1
    assert checks > 1000


def test_logical_mapping_matches_per_index_reference():
    rng = random.Random(0)
    for _ in range(5000):
        capacity = rng.randint(1, 30)
        segments = [(rng.randrange(capacity), rng.randrange(capacity)) for _ in range(rng.randint(0, 6))]
        offset, count = rng.randrange(capacity), rng.randint(0, capacity)
        assert rb_module._physical_segments_to_logical_half_open(
            segments, offset=offset, count=count, capacity=capacity,
        ) == _reference_logical_half_open(segments, offset=offset, count=count, capacity=capacity)


def test_subtract_physical_range_matches_per_index_reference():
    rng = random.Random(1)
    for _ in range(5000):
        capacity = rng.choice([None, rng.randint(1, 25)])
        limit = capacity or 40
        start = rng.randrange(limit)
        if capacity is None:
            segment = (start, rng.randrange(start, limit))
        else:
            segment = (start, rng.randrange(limit))
        victim_start, victim_length = rng.randrange(limit), rng.randint(1, limit)
        if capacity is None:
            victim_set = set(range(victim_start, victim_start + victim_length))
            victims = [(victim_start, victim_start + victim_length - 1)]
        else:
            victim_end = (victim_start + victim_length - 1) % capacity
            victim_set = set(_expand((victim_start, victim_end), capacity))
            victims = rb_module._linear_pieces((victim_start, victim_end), capacity)
        expected = _compress([i for i in _expand(segment, capacity) if i not in victim_set], capacity)
        assert rb_module._subtract_physical_range(segment, victims, capacity) == expected


def test_extend_writes_a_slice_unless_it_wraps():
    storage = _RecordingStorage(8)
    rb = ReplayBuffer(storage, "s")
    rb.extend(np.zeros((5, 1), np.float32))
    rb.extend(np.zeros((3, 1), np.float32))
    assert storage.set_indices[:2] == [slice(0, 5), slice(5, 8)]
    rb.extend(np.zeros((2, 1), np.float32))
    assert storage.set_indices[2] == slice(0, 2)
    rb.extend(np.zeros((7, 1), np.float32))
    wrapped = storage.set_indices[3]
    np.testing.assert_array_equal(wrapped, np.array([2, 3, 4, 5, 6, 7, 0]))
    assert (rb.count, rb.offset) == (8, 1)


@pytest.mark.parametrize("backend, make", [
    (NumpyComputeBackend, lambda values: np.asarray(values, dtype=np.int64)),
    (PyTorchComputeBackend, lambda values: torch.tensor(values, dtype=torch.int64)),
])
def test_index_with_offset_bounds_check(backend, make):
    mapped = rb_module.index_with_offset(backend, make([-4, -1, 0, 3]), 4, 1)
    assert [int(x) for x in mapped] == [1, 0, 1, 0]
    for bad in ([4], [-5], [0, 4], [-5, 3]):
        with pytest.raises(AssertionError, match="out of bounds"):
            rb_module.index_with_offset(backend, make(bad), 4, 1)
