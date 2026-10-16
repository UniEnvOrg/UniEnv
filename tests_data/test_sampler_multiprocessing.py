"""MultiprocessingSampler: background workers producing sampler batches."""

import gc
import multiprocessing as mp

import pytest
import torch

from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
from unienv_interface.space import BoxSpace, DictSpace
from unienv_data import ReplayBuffer
from unienv_data.samplers import MultiprocessingSampler, StepSampler
from unienv_data.storages.pytorch import PytorchTensorStorage

from tests_common.batch_helpers import make_rb

pytestmark = [pytest.mark.slow, pytest.mark.timeout(120)]

N = 1000
BATCH = 8


def _int_rb(n=N):
    space = BoxSpace(TB, 0, n, torch.int64, shape=(), device="cpu")
    rb = ReplayBuffer.create(PytorchTensorStorage, space, capacity=n, is_memmap=False)
    rb.extend(torch.arange(n))
    return rb


def _sampler(ctx="fork", n=N, batch=BATCH, n_workers=3, **kwargs):
    return MultiprocessingSampler(
        StepSampler(_int_rb(n), batch_size=batch, seed=0),
        n_workers=n_workers, n_buffers=4, ctx=mp.get_context(ctx), **kwargs,
    )


def _workers(fn):
    manager = getattr(fn, "sample_manager", None)
    return [] if manager is None else list(manager.workers)


@pytest.mark.parametrize("ctx", ["fork", "spawn"])
def test_workers_produce_distinct_batches(ctx):
    # Regression: every worker inherited the same RNG state, so the workers
    # returned identical batches (3 unique batches out of 12 with 4 workers),
    # and spawn contexts failed because the worker targets were local closures.
    sampler = _sampler(ctx)
    try:
        batches = [tuple(sampler.sample().tolist()) for _ in range(12)]
    finally:
        workers = _workers(sampler.sample)
        sampler.close()
    assert len(set(batches)) == len(batches)
    assert all(not w.is_alive() for w in workers) and len(workers) == 3


def test_samples_are_in_sampled_space():
    sampler = MultiprocessingSampler(StepSampler(make_rb(20), batch_size=4, seed=1),
                                     n_workers=2, n_buffers=2, ctx=mp.get_context("fork"))
    try:
        space = sampler.sampled_space
        assert space == sampler.sampler.sampled_space
        assert sampler.sampled_metadata_space is None
        assert len(sampler) == 20 and sampler.backend is TB and sampler.data is sampler.sampler.data
        for _ in range(4):
            assert space.contains(sampler.sample())
        flat = sampler.sample_flat()
        assert flat.shape[0] == 4
        data, meta = sampler.sample_with_metadata()
        assert space.contains(data) and meta is None
        assert space.contains(sampler.get_at(torch.tensor([0, 1, 2, 3])))
        assert sampler.get_flattened_at(torch.tensor([0, 1, 2, 3])).shape[0] == 4
        assert sampler.get_at_with_metadata(torch.tensor([0, 1, 2, 3]))[1] is None
        assert sampler.get_flattened_at_with_metadata(torch.tensor([0, 1, 2, 3]))[1] is None
    finally:
        sampler.close()


def test_epoch_iter_is_a_permutation():
    sampler = _sampler(n=50, batch=8)
    try:
        batches = list(sampler.epoch_iter())
        # Batches arrive in completion order, so the short batch is not necessarily last.
        assert sorted(len(b) for b in batches) == [2] + [8] * 6
        assert sorted(torch.cat(batches).tolist()) == list(range(50))
        flat = list(sampler.epoch_flat_iter())
        assert sorted(torch.cat([f.reshape(-1) for f in flat]).tolist()) == list(range(50))
        with_meta = list(sampler.epoch_iter_with_metadata())
        assert sorted(torch.cat([b for b, _ in with_meta]).tolist()) == list(range(50))
    finally:
        sampler.close()


def test_breaking_out_of_an_epoch_stops_its_workers():
    sampler = _sampler(n=200, batch=4)
    try:
        it = sampler.epoch_iter()
        next(it)
        workers = _workers(sampler.epoch_iter)
        assert workers and all(w.is_alive() for w in workers)
        it.close()
        for w in workers:
            w.join(timeout=10)
        assert all(not w.is_alive() for w in workers)
    finally:
        sampler.close()


def test_close_terminates_workers_and_is_idempotent():
    sampler = _sampler()
    sampler.sample()
    sampler.sample_flat()
    workers = _workers(sampler.sample) + _workers(sampler.sample_flat)
    assert len(workers) == 6 and all(w.is_alive() for w in workers)
    sampler.close()
    sampler.close()
    assert all(not w.is_alive() for w in workers)


def test_manual_seed_and_rng_passthrough():
    sampler = _sampler()
    try:
        sampler.manual_seed(5)
        rng = sampler.data_rng
        assert rng is sampler.sampler.data_rng
        sampler.data_rng = rng
        sampler.rng = None
        assert sampler.rng is None
    finally:
        sampler.close()


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_invalid_arguments_raise_cleanly():
    with pytest.raises(AssertionError):
        MultiprocessingSampler(StepSampler(make_rb(4), batch_size=2), n_workers=0)
    gc.collect()


_FORK_AFTER_PARALLEL_OP = """
import multiprocessing as mp, torch
from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
from unienv_interface.space import BoxSpace
from unienv_data import ReplayBuffer
from unienv_data.samplers import MultiprocessingSampler, StepSampler
from unienv_data.storages.pytorch import PytorchTensorStorage

space = BoxSpace(TB, 0, 255, torch.uint8, shape=(3, 64, 64), device="cpu")
rb = ReplayBuffer.create(PytorchTensorStorage, space, capacity=1000, is_memmap=False)
rb.extend(torch.randint(0, 256, (1000, 3, 64, 64), dtype=torch.uint8))
# 256 images of 3*64*64 bytes: well above the intra-op parallel grain size.
StepSampler(rb, batch_size=256, seed=0).sample()  # the parent runs a parallel gather
sampler = MultiprocessingSampler(
    StepSampler(rb, batch_size=256, seed=0), n_workers=2, n_buffers=1, ctx=mp.get_context("fork"),
)
assert sampler.sample().shape == (256, 3, 64, 64)
assert sampler.sample().shape == (256, 3, 64, 64)
sampler.close()
print("ok")
"""


def test_fork_workers_do_not_deadlock_after_parent_used_intraop_pool():
    # Regression: a fork-ed worker inherits torch's OpenMP thread-pool state but
    # not its threads, so once the parent had run a parallel op (here a large
    # image gather), the worker's first parallel op waited forever on threads
    # that do not exist and sample() never returned. Workers now run their
    # tensor ops single-threaded, as torch's DataLoader workers do.
    # Runs in a fresh interpreter: the pool state belongs to the thread that used
    # it, so the parent must use it, and fork, from its main thread, and a hang
    # must be killable (whole process group) without leaving stuck workers behind.
    import os, signal, subprocess, sys
    if torch.get_num_threads() < 2:
        pytest.skip("needs a multi-threaded intra-op pool")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [sys.executable, "-c", _FORK_AFTER_PARALLEL_OP], cwd=root, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        out, err = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        pytest.fail("fork-ed sampler workers deadlocked")
    assert proc.returncode == 0 and out.strip().endswith("ok"), err

def test_worker_threads_are_limited_only_if_torch_is_imported(monkeypatch):
    import sys
    from unienv_data.samplers import multiprocessing_sampler as mps
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    mps._limit_worker_intraop_threads()
    assert calls == [1]
    monkeypatch.delitem(sys.modules, "torch")
    mps._limit_worker_intraop_threads()
    assert calls == [1]


def _assert_rb_rows_consistent(batch):
    # make_rb rows: obs[i] = [2i, 2i + 1], episode_id = i // 4, episode_end = i % 4 == 3.
    i = (batch["obs"][:, 0] / 2).long()
    assert torch.equal(batch["obs"][:, 1], batch["obs"][:, 0] + 1)
    assert torch.equal(batch["episode_id"], i // 4)
    assert torch.equal(batch["episode_end"], i % 4 == 3)


@pytest.mark.parametrize("ctx", ["fork", "spawn"])
def test_shared_memory_batches_match_and_are_not_overwritten(ctx):
    sampler = MultiprocessingSampler(
        StepSampler(make_rb(200), batch_size=16, seed=0),
        n_workers=2, n_buffers=3, ctx=mp.get_context(ctx), shared_memory=True,
    )
    try:
        batches = [sampler.sample() for _ in range(6)]
        snapshots = [{k: v.clone() for k, v in b.items()} for b in batches]
        batches += [sampler.sample() for _ in range(12)]  # every slot is reused several times
        assert sampler.sample.slots is not None and len(sampler.sample.slots) == 4
        for b in batches:
            assert sampler.sampled_space.contains(b)
            _assert_rb_rows_consistent(b)
        # Returned batches are copies: refilling their slot does not change them.
        for b, snap in zip(batches, snapshots):
            assert all(torch.equal(b[k], snap[k]) for k in snap)
        assert len({tuple(b["obs"][:, 0].tolist()) for b in batches}) == len(batches)
        data, meta = sampler.sample_with_metadata()
        _assert_rb_rows_consistent(data)
        assert meta is None and sampler.sample_with_metadata.slots is not None
        assert sampler.sample_flat().shape[0] == 16
        assert sampler.sample_flat.slots is None  # only sample / sample_with_metadata use slots
    finally:
        sampler.close()


def test_shared_memory_is_off_by_default():
    sampler = _sampler()
    try:
        sampler.sample()
        assert sampler.sample.slots is None
    finally:
        sampler.close()


def test_shared_memory_without_torch_tensors_falls_back():
    from tests_common.batch_helpers import numpy_array_batch
    sampler = MultiprocessingSampler(
        StepSampler(numpy_array_batch(40, with_metadata=False), batch_size=4, seed=0),
        n_workers=2, n_buffers=2, ctx=mp.get_context("fork"), shared_memory=True,
    )
    try:
        batch = sampler.sample()
        assert sampler.sample.slots is None
        assert sampler.sampled_space.contains(batch)
    finally:
        sampler.close()


def test_shared_slot_task_sends_mismatching_leaves_as_is():
    from unienv_data.samplers.multiprocessing_sampler import _SharedSlotTask, _allocate_shared_slots, _InSlot
    exemplar = {"a": torch.zeros(4, 2), "b": torch.zeros(4, dtype=torch.int64), "c": None}
    slots = _allocate_shared_slots(exemplar, 2)
    assert slots[0]["a"].is_shared() and slots[0]["c"] is None
    result = {"a": torch.ones(4, 2), "b": torch.ones(3, dtype=torch.int64), "c": "x"}
    task = _SharedSlotTask(lambda work: result, slots)
    slot_id, packed, skeleton = task(1)
    assert slot_id == 1 and packed
    assert isinstance(skeleton["a"], _InSlot) and skeleton["b"] is result["b"] and skeleton["c"] == "x"
    out = _SharedSlotTask.unpack(packed, skeleton, slots[1])
    assert torch.equal(out["a"], result["a"]) and out["a"].data_ptr() != slots[1]["a"].data_ptr()
    # A result whose structure differs from the slots is sent whole.
    other = {"z": torch.ones(4, 2)}
    _, packed, skeleton = _SharedSlotTask(lambda work: other, slots)(0)
    assert not packed and _SharedSlotTask.unpack(packed, skeleton, slots[0]) is other
    assert _allocate_shared_slots({"x": 1}, 2) is None
