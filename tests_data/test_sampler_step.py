"""StepSampler: uniform mini-batch sampling over a batch."""

import numpy as np
import pytest
import torch

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_data.samplers import StepSampler

from tests_common.batch_helpers import make_rb, numpy_array_batch
from tests_common.trees import assert_tree_equal, to_numpy


def _steps(batch):
    """Sample ids recovered from ``obs[:, 0] == 2 * i``."""
    return (to_numpy(batch["obs"])[:, 0] // 2).astype(int).tolist()


def test_same_seed_same_batches():
    a = StepSampler(make_rb(50), batch_size=6, seed=3)
    b = StepSampler(make_rb(50), batch_size=6, seed=3)
    c = StepSampler(make_rb(50), batch_size=6, seed=4)
    seq_a = [_steps(a.sample()) for _ in range(5)]
    assert seq_a == [_steps(b.sample()) for _ in range(5)]
    assert seq_a != [_steps(c.sample()) for _ in range(5)]
    a.manual_seed(3)
    assert [_steps(a.sample()) for _ in range(5)] == seq_a


@pytest.mark.parametrize("make", [lambda: make_rb(23), lambda: numpy_array_batch(23)], ids=["torch-rb", "numpy"])
def test_epoch_iter_is_a_permutation_with_short_last_batch(make):
    sampler = StepSampler(make(), batch_size=5, seed=0)
    batches = list(sampler.epoch_iter())
    assert [len(b["obs"]) for b in batches] == [5, 5, 5, 5, 3]
    assert sorted(sum((_steps(b) for b in batches), [])) == list(range(23))
    assert len(list(iter(sampler))) == 5

    flat = list(sampler.epoch_flat_iter())
    assert [f.shape[0] for f in flat] == [5, 5, 5, 5, 3]
    with_meta = list(sampler.epoch_iter_with_metadata())
    assert sorted(sum((_steps(b) for b, _ in with_meta), [])) == list(range(23))
    flat_meta = list(sampler.epoch_flat_iter_with_metadata())
    assert sum(f.shape[0] for f, _ in flat_meta) == 23


def test_exact_multiple_has_no_short_batch():
    sampler = StepSampler(make_rb(12), batch_size=4, seed=0)
    assert [len(b["obs"]) for b in sampler.epoch_iter()] == [4, 4, 4]


def test_samples_are_in_sampled_space():
    rb = make_rb(30)
    sampler = StepSampler(rb, batch_size=7, seed=1)
    assert len(sampler) == 30
    space = sampler.sampled_space
    assert sbu.batch_size(space) == 7
    for _ in range(5):
        assert space.contains(sampler.sample())
    idx = sampler.sample_index()
    assert idx.shape == (7,) and int(idx.min()) >= 0 and int(idx.max()) < 30
    flat = sampler.sample_flat()
    assert flat.shape == (7, sfu.flat_dim(rb.single_space))
    data, meta = sampler.sample_with_metadata()
    assert space.contains(data) and meta is None
    flat, meta = sampler.sample_flat_with_metadata()
    assert flat.shape[0] == 7 and meta is None


def test_metadata_is_sampled_with_data():
    src = numpy_array_batch(40)
    sampler = StepSampler(src, batch_size=8, seed=2)
    assert sampler.sampled_metadata_space == sbu.batch_space(src.single_metadata_space, 8)
    data, meta = sampler.sample_with_metadata()
    assert_tree_equal(meta["step"], np.asarray(_steps(data)) * 10, check_dtype=False)
    flat, meta = sampler.sample_flat_with_metadata()
    assert flat.shape[0] == 8 and meta["step"].shape == (8,)


def test_explicit_device_moves_outputs():
    rb = make_rb(10)
    sampler = StepSampler(rb, batch_size=3, seed=0, device="cpu")
    assert sampler.device == "cpu"
    out = sampler.sample()
    assert all(v.device == torch.device("cpu") for v in out.values())
    assert sampler.sample_flat().device == torch.device("cpu")
    assert sampler.sample_with_metadata()[1] is None
    assert sampler.sample_flat_with_metadata()[0].shape[0] == 3


def test_explicit_device_moves_metadata():
    from tests_common.batch_helpers import ArrayBatch
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    from unienv_interface.space import BoxSpace, DictSpace
    rb = make_rb(6)
    meta_space = DictSpace(TB, {"step": BoxSpace(TB, 0, 100, torch.int64, shape=(), device="cpu")}, device="cpu")
    src = ArrayBatch(rb.single_space, rb.get_at(...), meta_space, {"step": torch.arange(6)})
    sampler = StepSampler(src, batch_size=4, seed=0, device="cpu")
    data, meta = sampler.sample_with_metadata()
    assert meta["step"].device == torch.device("cpu")
    assert_tree_equal(meta["step"], torch.tensor(_steps(data)))
    _, meta = sampler.sample_flat_with_metadata()
    assert meta["step"].shape == (4,)


def test_invalid_batch_size():
    with pytest.raises(AssertionError):
        StepSampler(make_rb(4), batch_size=0)
