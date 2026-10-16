"""Readers of a ``multiprocessing=True`` ReplayBuffer in other processes see consistent (count, offset) pairs."""

import multiprocessing as mp

import pytest
import torch

from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace

from tests_common.storage_ring_hammer import HAMMER_CAPACITY, hammer_reader, hammer_writer


def _shared_rb() -> ReplayBuffer:
    space = BoxSpace(PyTorchComputeBackend, 0, HAMMER_CAPACITY, torch.int64, shape=(), device="cpu")
    return ReplayBuffer.create(
        PytorchTensorStorage, space, capacity=HAMMER_CAPACITY, is_memmap=False, multiprocessing=True,
    )


@pytest.mark.slow
@pytest.mark.timeout(120)
@pytest.mark.parametrize("ctx_name", ["fork", "spawn"])
def test_readers_never_see_torn_count_offset_pairs(ctx_name):
    """
    The writer moves the buffer through the "becomes full" transition (count and offset both change) and
    clear() thousands of times while reader processes read it without locking. Reading count and offset as
    two separate shared values let a reader combine the new offset with the old count and read the ring with
    a wrong index mapping.
    """
    ctx = mp.get_context(ctx_name)
    rb = _shared_rb()
    stop_event = ctx.Event()
    result_queue = ctx.Queue()
    readers = [ctx.Process(target=hammer_reader, args=(rb, stop_event, result_queue)) for _ in range(3)]
    writer = ctx.Process(target=hammer_writer, args=(rb, stop_event, 3.0))
    for reader in readers:
        reader.start()
    writer.start()
    try:
        results = [result_queue.get(timeout=60) for _ in readers]
    finally:
        stop_event.set()
        writer.join(timeout=30)
        for reader in readers:
            reader.join(timeout=30)
    assert writer.exitcode == 0
    assert all(reader.exitcode == 0 for reader in readers)
    total_reads = sum(reads for reads, _ in results)
    torn = [physical for _, bad in results for physical in bad]
    assert total_reads > 1000
    assert torn == [], f"{len(torn)}+ reads used an inconsistent (count, offset) pair, e.g. {torn[:3]}"


def test_shared_ring_state_round_trips_count_and_offset():
    rb = _shared_rb()
    rb.extend(torch.arange(HAMMER_CAPACITY - 1))
    rb.extend(torch.arange(HAMMER_CAPACITY - 1, HAMMER_CAPACITY + 3))
    assert (rb.count, rb.offset) == (HAMMER_CAPACITY, 3)
    assert rb.get_at(slice(None)).tolist() == list(range(3, HAMMER_CAPACITY + 3))
    rb.clear()
    assert (rb.count, rb.offset) == (0, 0)
    rb.count, rb.offset = 5, 0
    assert len(rb) == 5
