"""ReplayBuffer behaviour on top of every storage backend of the contract suite."""

import multiprocessing as mp
import pickle

import numpy as np
import pytest

from unienv_data.replay_buffer import ReplayBuffer
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu

from tests_common.storage_cases import STORAGE_CASES, get_rows, random_batch, to_backend_index
from tests_common.storage_video import clean_video_codec_cache  # noqa: F401  (fixture)
from tests_common.trees import assert_tree_equal

pytestmark = pytest.mark.usefixtures("clean_video_codec_cache")

CAPACITY = 5


def _params(capacity):
    return [pytest.param(case, id=case.id, marks=list(case.marks))
            for case in STORAGE_CASES if capacity in case.capacities]


def _create_rb(case, tmp_path, capacity, **kwargs):
    create_kwargs = dict(case.create_kwargs() if callable(case.create_kwargs) else case.create_kwargs)
    if capacity is None:
        create_kwargs.update(case.unbounded_create_kwargs)
    create_kwargs.update(kwargs)
    return ReplayBuffer.create(
        case.storage_cls(),
        case.make_space(),
        cache_path=str(tmp_path / "rb") if case.uses_cache_path else None,
        capacity=capacity,
        **create_kwargs,
    )


def _close(rb):
    storage = rb.storage
    rb.close()
    if storage is not None:
        storage.close()


def _check(case, actual, expected):
    assert_tree_equal(actual, expected, atol=case.atol, check_dtype=case.check_dtype)


def _flat(space, value, batched: bool):
    if batched:
        n = sbu.batch_size_data(value)
        return sfu.flatten_data(sbu.batch_space(space, n), value, start_dim=1)
    return sfu.flatten_data(space, value)


@pytest.mark.parametrize("case", _params(CAPACITY))
def test_wrapped_buffer_access_paths_agree(case, tmp_path):
    rb = _create_rb(case, tmp_path, CAPACITY)
    space = rb.single_space
    data = random_batch(space, 7, seed=1)
    rb.extend(get_rows(space, data, slice(0, 3)))
    rb.extend(get_rows(space, data, slice(3, 7)))
    assert (rb.count, rb.offset) == (CAPACITY, 2)
    expected = get_rows(space, data, slice(2, 7))

    for i in range(-CAPACITY, CAPACITY):
        row = get_rows(space, expected, i % CAPACITY)
        _check(case, rb.get_at(i), row)
        _check(case, rb[i], row)
        value, metadata = rb.get_at_with_metadata(i)
        _check(case, value, row)
        assert metadata is None
        _check(case, rb.get_flattened_at(i), _flat(space, row, False))
        flat, metadata = rb.get_flattened_at_with_metadata(i)
        _check(case, flat, _flat(space, row, False))

    for index, rows in [
        (slice(None), np.arange(5)),
        (Ellipsis, np.arange(5)),
        (slice(1, 4), np.arange(1, 4)),
        (np.array([4, 0, -1, 2]), np.array([4, 0, 4, 2])),
        (np.array([True, False, False, True, True]), np.array([0, 3, 4])),
    ]:
        batch = get_rows(space, expected, rows)
        backend_index = to_backend_index(space, index)
        _check(case, rb.get_at(backend_index), batch)
        _check(case, rb.get_flattened_at(backend_index), _flat(space, batch, True))

    new = random_batch(space, 2, seed=2)
    rb.set_at(0, get_rows(space, new, 0))
    _check(case, rb.get_at(0), get_rows(space, new, 0))
    rb.set_flattened_at(-1, _flat(space, get_rows(space, new, 1), False))
    _check(case, rb.get_at(CAPACITY - 1), get_rows(space, new, 1))
    rb.set_at(slice(1, 3), get_rows(space, new, slice(0, 2)))
    _check(case, rb.get_at(slice(1, 3)), new)
    rb.set_flattened_at(slice(2, 4), _flat(space, new, True))
    _check(case, rb.get_at(slice(2, 4)), new)
    _close(rb)


@pytest.mark.parametrize("case", _params(CAPACITY))
def test_wrapped_buffer_dumps_and_load_from(case, tmp_path):
    rb = _create_rb(case, tmp_path, CAPACITY, maintain_segment_metadata=True)
    space = rb.single_space
    data = random_batch(space, 7, seed=4)
    for k in range(3):
        rb.append(get_rows(space, data, k))
    rb.mark_segment_end()
    for k in range(3, 7):
        rb.append(get_rows(space, data, k))
    rb.mark_segment_end()
    # The second segment wrapped around and evicted the first two rows.
    assert (rb.count, rb.offset) == (CAPACITY, 2)
    assert rb.get_segments() == [(0, 1), (1, 5)]
    _check(case, rb.get_at(slice(None)), get_rows(space, data, slice(2, 7)))

    dump_dir = rb.cache_path or str(tmp_path / "dump")
    rb.dumps(dump_dir)
    _close(rb)
    assert ReplayBuffer.is_loadable_from(dump_dir)
    assert ReplayBuffer.get_length_from_path(dump_dir) == CAPACITY
    assert ReplayBuffer.get_capacity_from_path(dump_dir) == CAPACITY
    assert ReplayBuffer.get_space_from_path(dump_dir, backend=space.backend, device=space.device) == space

    loaded = ReplayBuffer.load_from(dump_dir, backend=space.backend, device=space.device,
                                    single_instance_space=case.make_space(), **case.load_kwargs)
    assert (loaded.count, loaded.offset, loaded.capacity) == (CAPACITY, 2, CAPACITY)
    assert loaded.get_segments() == [(0, 1), (1, 5)]
    assert not loaded.is_mutable
    _check(case, loaded.get_at(slice(None)), get_rows(space, data, slice(2, 7)))
    _check(case, loaded.get_flattened_at(-1), _flat(space, get_rows(space, data, 6), False))
    _close(loaded)


@pytest.mark.parametrize("case", _params(None))
def test_unbounded_buffer_extend_append_dump_load(case, tmp_path):
    rb = _create_rb(case, tmp_path, None)
    space = rb.single_space
    data = random_batch(space, 6, seed=6)
    rb.extend(get_rows(space, data, slice(0, 2)))
    for k in range(2, 5):
        rb.append(get_rows(space, data, k))
    with pytest.raises(RuntimeError):
        rb.extend(get_rows(space, data, slice(5, 6)))  # append-built segment still open
    rb.mark_segment_end()
    rb.extend(get_rows(space, data, slice(5, 6)))
    assert len(rb) == 6 and rb.offset == 0
    _check(case, rb.get_at(slice(None)), data)
    _check(case, rb.get_at(-1), get_rows(space, data, 5))
    dump_dir = rb.cache_path or str(tmp_path / "dump")
    rb.dumps(dump_dir)
    _close(rb)
    loaded = ReplayBuffer.load_from(dump_dir, backend=space.backend, device=space.device, **case.load_kwargs)
    assert loaded.capacity is None and len(loaded) == 6
    _check(case, loaded.get_at(slice(None)), data)
    _close(loaded)


@pytest.mark.parametrize("case", _params(CAPACITY))
def test_clear_resets_buffer(case, tmp_path):
    rb = _create_rb(case, tmp_path, CAPACITY, maintain_segment_metadata=True)
    space = rb.single_space
    data = random_batch(space, 7, seed=8)
    rb.extend(data)
    rb.clear()
    assert (rb.count, rb.offset, len(rb)) == (0, 0, 0)
    assert rb.get_segments() == []
    rb.extend(get_rows(space, data, slice(0, 2)))
    _check(case, rb.get_at(slice(None)), get_rows(space, data, slice(0, 2)))
    _close(rb)


@pytest.mark.parametrize("case", [c for c in _params(CAPACITY) if c.values[0].multiprocessing_load])
def test_read_only_buffer_pickles(case, tmp_path):
    rb = _create_rb(case, tmp_path, CAPACITY)
    space = rb.single_space
    data = random_batch(space, 6, seed=10)
    rb.extend(data)
    dump_dir = rb.cache_path or str(tmp_path / "dump")
    rb.dumps(dump_dir)
    _close(rb)
    loaded = ReplayBuffer.load_from(dump_dir, backend=space.backend, device=space.device,
                                    multiprocessing=True, **case.load_kwargs)
    assert loaded.is_multiprocessing_safe
    clone = pickle.loads(pickle.dumps(loaded))
    _check(case, clone.get_at(slice(None)), get_rows(space, data, slice(1, 6)))
    _close(clone)
    _close(loaded)


def _child_extend(rb, values):
    rb.extend(values)


@pytest.mark.slow
def test_multiprocessing_buffer_shares_counters_and_data_with_child():
    import torch
    from unienv_data.storages.pytorch import PytorchTensorStorage
    from unienv_interface.backends.pytorch import PyTorchComputeBackend
    from unienv_interface.space import BoxSpace

    space = BoxSpace(PyTorchComputeBackend, -10.0, 10.0, torch.float32, shape=(2,), device="cpu")
    rb = ReplayBuffer.create(PytorchTensorStorage, space, capacity=4, is_memmap=False, multiprocessing=True)
    assert rb.is_multiprocessing_safe and rb.storage.is_multiprocessing_safe
    rb.extend(torch.zeros(3, 2))
    values = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    ctx = mp.get_context("fork")
    process = ctx.Process(target=_child_extend, args=(rb, values))
    process.start()
    process.join(timeout=60)
    assert process.exitcode == 0
    assert (rb.count, rb.offset) == (4, 2)
    torch.testing.assert_close(rb.get_at(slice(1, 4)), values)
    torch.testing.assert_close(rb.get_flattened_at(-1), values[-1])
