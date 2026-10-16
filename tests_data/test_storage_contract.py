"""Generic SpaceStorage contract (see ``unienv_data/base/storage.py``) run against every backend.

Backends and their legitimate differences are listed in ``tests_common/storage_cases.py``.
"""

import pickle

import numpy as np
import pytest

from unienv_interface.space.space_utils import batch_utils as sbu

from tests_common.storage_cases import (
    STORAGE_CASES,
    StorageCase,
    get_rows,
    random_batch,
    storage_case_params,
    to_backend_index,
)
from tests_common.storage_video import clean_video_codec_cache  # noqa: F401  (fixture)
from tests_common.trees import assert_tree_equal

pytestmark = pytest.mark.usefixtures("clean_video_codec_cache")

N = 5  # rows used by every test (== fixed capacity of the cases)

READ_ONLY_ERRORS = (AssertionError, RuntimeError, OSError, ValueError, KeyError, TypeError)


def _assert_rows(case: StorageCase, actual, expected) -> None:
    assert_tree_equal(actual, expected, atol=case.atol, check_dtype=case.check_dtype)


def _make(case: StorageCase, tmp_path, capacity, **kwargs):
    storage = case.create(tmp_path, capacity, **kwargs)
    if capacity is None:
        storage.extend_length(N)
    assert len(storage) == N
    return storage


def _filled(case: StorageCase, tmp_path, capacity, seed: int = 0):
    storage = _make(case, tmp_path, capacity)
    data = random_batch(storage.single_instance_space, N, seed)
    storage.set(slice(None), data)
    return storage, data


@pytest.fixture
def cleanup():
    """Collects storages and closes them at teardown (also when a test fails)."""
    storages = []
    yield storages.append
    for storage in storages:
        try:
            storage.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Creation, length and attributes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_create_length_and_attributes(case, capacity, tmp_path, cleanup):
    storage = case.create(tmp_path, capacity)
    cleanup(storage)
    assert storage.capacity == capacity
    assert len(storage) == (0 if capacity is None else capacity)
    assert storage.is_mutable
    assert not storage.has_open_segment
    assert storage.pending_segment_length == 0
    assert storage.single_instance_space == case.make_space()
    assert storage.backend is case.make_space().backend
    if case.uses_cache_path:
        assert storage.cache_filename is not None
    else:
        assert storage.cache_filename is None


@pytest.mark.parametrize("case,capacity", [p for p in storage_case_params() if p.values[1] is None])
def test_extend_and_shrink_length(case, capacity, tmp_path, cleanup):
    storage = case.create(tmp_path, capacity)
    cleanup(storage)
    space = storage.single_instance_space
    data = random_batch(space, 5, seed=3)
    storage.extend_length(3)
    storage.set(slice(0, 3), get_rows(space, data, slice(0, 3)))
    storage.extend_length(2)
    storage.set(slice(3, 5), get_rows(space, data, slice(3, 5)))
    assert len(storage) == 5
    _assert_rows(case, storage.get(slice(None)), data)
    storage.shrink_length(2)
    assert len(storage) == 3
    _assert_rows(case, storage.get(slice(None)), get_rows(space, data, slice(0, 3)))
    storage.shrink_length(3)
    assert len(storage) == 0


@pytest.mark.parametrize("case,capacity", [p for p in storage_case_params() if p.values[1] is not None])
def test_fixed_capacity_ignores_length_changes(case, capacity, tmp_path, cleanup):
    storage = case.create(tmp_path, capacity)
    cleanup(storage)
    if getattr(storage, "extend_length").__qualname__.startswith("SpaceStorage."):
        storage.extend_length(2)
        storage.shrink_length(2)
    else:
        for op in (storage.extend_length, storage.shrink_length):
            try:
                op(2)
            except AssertionError:
                pass  # rejecting the call is also acceptable
    assert len(storage) == capacity


# ---------------------------------------------------------------------------
# get / set indexing semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_get_single_index_is_unbatched(case, capacity, tmp_path, cleanup):
    storage, data = _filled(case, tmp_path, capacity)
    cleanup(storage)
    space = storage.single_instance_space
    for i in range(N):
        _assert_rows(case, storage.get(i), get_rows(space, data, i))


@pytest.mark.parametrize("index", [
    slice(1, 4), slice(None), slice(None, None, 2), slice(4, 0, -1), Ellipsis,
    np.array([4, 0, 2]), np.array([3, 3, 1, 3]), np.array([2]),
    np.array([True, False, True, True, False]),
], ids=["slice", "full-slice", "step-slice", "reverse-slice", "ellipsis",
        "int-array", "int-array-duplicates", "int-array-single", "bool-mask"])
@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_get_batched_index_adds_batch_dim(case, capacity, index, tmp_path, cleanup):
    if isinstance(index, slice) and (index.step or 1) < 0 and not case.negative_step_slices:
        pytest.skip(f"{case.id} does not support negative slice steps")
    storage, data = _filled(case, tmp_path, capacity)
    cleanup(storage)
    space = storage.single_instance_space
    if index is Ellipsis:
        rows = np.arange(N)
    elif isinstance(index, slice):
        rows = np.arange(N)[index]
    elif index.dtype == bool:
        rows = np.nonzero(index)[0]
    else:
        rows = index
    result = storage.get(to_backend_index(space, index))
    assert sbu.batch_size_data(result) == len(rows)
    _assert_rows(case, result, get_rows(space, data, rows))


@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_negative_indices(case, capacity, tmp_path, cleanup):
    storage, data = _filled(case, tmp_path, capacity)
    cleanup(storage)
    space = storage.single_instance_space
    if not case.negative_indices:
        with pytest.raises((AssertionError, IndexError)):
            storage.get(-1)
        return
    _assert_rows(case, storage.get(-1), get_rows(space, data, N - 1))
    _assert_rows(case, storage.get(-N), get_rows(space, data, 0))
    _assert_rows(case, storage.get(to_backend_index(space, np.array([-1, 0, -2]))),
                 get_rows(space, data, np.array([N - 1, 0, N - 2])))
    replacement = random_batch(space, 1, seed=11)
    storage.set(-2, get_rows(space, replacement, 0))
    _assert_rows(case, storage.get(N - 2), get_rows(space, replacement, 0))


@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_set_mirrors_get(case, capacity, tmp_path, cleanup):
    storage, data = _filled(case, tmp_path, capacity)
    cleanup(storage)
    space = storage.single_instance_space
    expected = data
    new = random_batch(space, N, seed=7)

    def write(index, rows):
        nonlocal expected
        storage.set(to_backend_index(space, index), get_rows(space, new, rows))
        expected = _replace_rows(space, expected, new, rows)
        _assert_rows(case, storage.get(slice(None)), expected)

    write(1, 1)  # single, unbatched value
    write(slice(2, 4), np.array([2, 3]))
    if case.scattered_set:
        write(np.array([4, 0]), np.array([4, 0]))
        write(np.array([False, True, False, True, False]), np.array([1, 3]))
    elif capacity is not None:
        write(np.array([4, 0]), np.array([4, 0]))  # one run wrapping around the capacity


def _replace_rows(space, base, new, rows):
    """Return ``base`` with ``rows`` (int or int array) replaced by the same rows of ``new``."""
    rows = np.atleast_1d(np.asarray(rows))
    batched = sbu.batch_space(space, N)
    out = base
    for r in rows:
        r = int(r)
        out = sbu.set_at(batched, out, slice(r, r + 1), sbu.get_at(batched, new, slice(r, r + 1)))
    return out


# ---------------------------------------------------------------------------
# Segment API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_segment_append_round_trip(case, capacity, tmp_path, cleanup):
    storage, data = _filled(case, tmp_path, capacity)
    cleanup(storage)
    space = storage.single_instance_space
    new = random_batch(space, 3, seed=5)
    # Fixed capacity: start at the last slot so the segment wraps around.
    start = N - 1 if capacity is not None else 1
    rows = np.array([(start + k) % N for k in range(3)])

    with pytest.raises(RuntimeError):
        storage.append(get_rows(space, new, 0))

    storage.mark_segment_start(start)
    assert storage.has_open_segment
    for k in range(3):
        storage.append(get_rows(space, new, k))
        assert storage.pending_segment_length == (k + 1 if case.buffered_segments else 0)
    storage.mark_segment_end()
    assert not storage.has_open_segment
    assert storage.pending_segment_length == 0
    storage.mark_segment_end()  # closing twice is a no-op

    _assert_rows(case, storage.get(to_backend_index(space, rows)), new)
    segments = storage.get_segments()
    if case.reports_segments:
        assert (int(start), int(rows[-1])) in [tuple(s) for s in segments]
    else:
        assert segments is None


@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_abort_segment(case, capacity, tmp_path, cleanup):
    storage, data = _filled(case, tmp_path, capacity)
    cleanup(storage)
    space = storage.single_instance_space
    new = random_batch(space, 1, seed=9)
    storage.abort_segment()  # no open segment: no-op
    storage.mark_segment_start(2)
    with pytest.raises(AssertionError):
        storage.mark_segment_start(3)
    storage.append(get_rows(space, new, 0))
    storage.abort_segment()
    assert not storage.has_open_segment
    assert storage.pending_segment_length == 0
    if case.buffered_segments:
        # buffered appends are discarded on abort
        _assert_rows(case, storage.get(2), get_rows(space, data, 2))
    # storage stays usable
    storage.mark_segment_start(2)
    storage.append(get_rows(space, new, 0))
    storage.mark_segment_end()
    _assert_rows(case, storage.get(2), get_rows(space, new, 0))


# ---------------------------------------------------------------------------
# clear / dumps / load_from / close
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_clear(case, capacity, tmp_path, cleanup):
    storage, _ = _filled(case, tmp_path, capacity)
    cleanup(storage)
    storage.mark_segment_start(0)
    storage.clear()
    assert not storage.has_open_segment
    assert len(storage) == (0 if capacity is None else capacity)
    if capacity is None:
        storage.extend_length(2)
        data = random_batch(storage.single_instance_space, 2, seed=4)
        storage.set(slice(None), data)
        _assert_rows(case, storage.get(slice(None)), data)


def _dump_path(case: StorageCase, storage, tmp_path) -> str:
    if storage.cache_filename is not None:
        return str(storage.cache_filename)
    return case.cache_path_for(tmp_path, "dumped")


@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_dumps_close_and_reload_read_only(case, capacity, tmp_path, cleanup):
    storage, data = _filled(case, tmp_path, capacity)
    space = storage.single_instance_space
    path = _dump_path(case, storage, tmp_path)
    storage.dumps(path)
    storage.close()

    loaded = case.load(path, capacity, read_only=True)
    cleanup(loaded)
    assert len(loaded) == N
    assert loaded.capacity == capacity
    assert not loaded.is_mutable
    _assert_rows(case, loaded.get(slice(None)), data)
    _assert_rows(case, loaded.get(N - 1), get_rows(space, data, N - 1))
    with pytest.raises(READ_ONLY_ERRORS):
        loaded.set(0, get_rows(space, data, 1))
    _assert_rows(case, loaded.get(0), get_rows(space, data, 0))


@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_reload_writable(case, capacity, tmp_path, cleanup):
    storage, data = _filled(case, tmp_path, capacity)
    space = storage.single_instance_space
    path = _dump_path(case, storage, tmp_path)
    storage.dumps(path)
    storage.close()

    loaded = case.load(path, capacity, read_only=False)
    cleanup(loaded)
    assert loaded.is_mutable
    new = random_batch(space, 1, seed=21)
    loaded.set(3, get_rows(space, new, 0))
    loaded.dumps(path)
    loaded.close()

    again = case.load(path, capacity, read_only=True)
    cleanup(again)
    _assert_rows(case, again.get(3), get_rows(space, new, 0))
    _assert_rows(case, again.get(0), get_rows(space, data, 0))


@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_dumps_to_another_path(case, capacity, tmp_path, cleanup):
    storage, data = _filled(case, tmp_path, capacity)
    cleanup(storage)
    target = case.cache_path_for(tmp_path, "copy")
    if case.dumps_only_to_own_path:
        with pytest.raises(AssertionError):
            storage.dumps(target)
        return
    storage.dumps(target)
    loaded = case.load(target, capacity, read_only=True)
    cleanup(loaded)
    _assert_rows(case, loaded.get(slice(None)), data)
    # the source keeps working after the copy
    _assert_rows(case, storage.get(slice(None)), data)


# ---------------------------------------------------------------------------
# Multiprocessing / pickling
# ---------------------------------------------------------------------------

def _assert_pickle_round_trip(case, storage, data):
    assert storage.is_multiprocessing_safe
    clone = pickle.loads(pickle.dumps(storage))
    try:
        _assert_rows(case, clone.get(slice(None)), data)
    finally:
        clone.close()


@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_create_with_multiprocessing(case, capacity, tmp_path, cleanup):
    if not case.multiprocessing_create:
        pytest.skip(f"{case.id} does not support multiprocessing-safe mutable creation")
    storage = case.create(tmp_path, capacity, multiprocessing=True)
    cleanup(storage)
    if capacity is None:
        storage.extend_length(N)
    data = random_batch(storage.single_instance_space, N, seed=2)
    storage.set(slice(None), data)
    _assert_pickle_round_trip(case, storage, data)


@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_load_read_only_multiprocessing_is_picklable(case, capacity, tmp_path, cleanup):
    if not case.multiprocessing_load:
        pytest.skip(f"{case.id} does not support multiprocessing loading")
    storage, data = _filled(case, tmp_path, capacity)
    path = _dump_path(case, storage, tmp_path)
    storage.dumps(path)
    storage.close()
    loaded = case.load(path, capacity, read_only=True, multiprocessing=True)
    cleanup(loaded)
    _assert_rows(case, loaded.get(slice(None)), data)
    _assert_pickle_round_trip(case, loaded, data)


def test_every_case_has_unique_id():
    ids = [case.id for case in STORAGE_CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case,capacity", storage_case_params())
def test_close_is_idempotent(case, capacity, tmp_path):
    storage, _ = _filled(case, tmp_path, capacity)
    storage.close()
    storage.close()


@pytest.mark.parametrize("case", storage_case_params(capacities=False))
def test_create_and_load_from_honour_subclasses(case, tmp_path, monkeypatch):
    base_cls = case.storage_cls()
    subclass = type(f"Custom{base_cls.__name__}", (base_cls,), {})
    monkeypatch.setattr(case, "storage_cls", lambda: subclass)
    capacity = case.capacities[-1]
    storage = case.create(tmp_path, capacity)
    assert type(storage) is subclass
    if capacity is None:
        storage.extend_length(N)
    data = random_batch(storage.single_instance_space, len(storage), seed=1)
    storage.set(slice(None), data)
    path = _dump_path(case, storage, tmp_path)
    storage.dumps(path)
    storage.close()
    loaded = case.load(path, capacity)
    try:
        assert type(loaded) is subclass
        _assert_rows(case, loaded.get(slice(None)), data)
    finally:
        loaded.close()
