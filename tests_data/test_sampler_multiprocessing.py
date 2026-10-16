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
        assert [len(b) for b in batches] == [8] * 6 + [2]
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
