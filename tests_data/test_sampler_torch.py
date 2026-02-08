from typing import Generic, TypeVar, Generic, Optional, Any, Dict as DictT, Tuple as TupleT, Sequence as SequenceT, Union as UnionT, List
import torch
import numpy as np
import pytest
import tempfile

from unienv_data import *
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_data.storages.flattened import FlattenedStorage
from unienv_data.samplers import *
from unienv_interface.space import *
from unienv_interface.space.space_utils import flatten_utils as sfu, batch_utils as sbu
from unienv_interface.backends.pytorch import PyTorchComputeBackend

from test_replay_buffer import check_fixed_capacity_replay_buffer

def construct_torch_rb(
    space : Space,
    capacity : int,
    use_mmap : bool,
    seed : Optional[int] = None
) -> ReplayBuffer:
    rb = ReplayBuffer.create(
        FlattenedStorage,
        space,
        inner_storage_cls=PytorchTensorStorage,
        cache_path=tempfile.mkdtemp() if use_mmap else None,
        capacity=capacity,
        is_memmap=use_mmap,
    )
    return rb

@pytest.mark.parametrize("capacity", [10, 50])
@pytest.mark.parametrize("seed", [0, 1024])
def test_step_sampler(
    capacity : int,
    seed : int
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    space = BoxSpace(
        PyTorchComputeBackend,
        0.0,
        1.0,
        dtype=torch.float32,
        device=device,
        shape=(3, 5, 2)
    )
    rb = construct_torch_rb(space, capacity, False, seed)
    rb = check_fixed_capacity_replay_buffer(
        rb,
        seed=seed,
        load_kwargs={"is_memmap": False}
    )
    sampler = StepSampler(
        rb,
        batch_size=capacity//2,
        seed=seed
    )
    sample_space = sampler.sampled_space
    for i in range(10):
        sample = sampler.sample()
        assert sample_space.contains(sample)

@pytest.mark.parametrize("capacity", [50, 100])
@pytest.mark.parametrize("seed", [0, 1024])
@pytest.mark.parametrize("use_mmap", [True, False])
def test_multiprocessing_sampler(
    capacity : int,
    seed : int,
    use_mmap : bool
):
    device = torch.device("cpu")
    space = BoxSpace(
        PyTorchComputeBackend,
        0.0,
        1.0,
        dtype=torch.float32,
        device=device,
        shape=(3, 5, 2)
    )
    rb = construct_torch_rb(space, capacity, use_mmap, seed)
    rb = check_fixed_capacity_replay_buffer(
        rb,
        seed=seed,
        load_kwargs={
            "is_memmap": use_mmap
        }
    )
    sampler = StepSampler(
        rb,
        batch_size=capacity//5,
        seed=seed,
    )
    sampler = MultiprocessingSampler(
        sampler,
        n_workers=4,
        n_buffers=8,
        ctx=torch.multiprocessing.get_context("spawn") if device.type == "cuda" else torch.multiprocessing.get_context("fork")
    )
    sample_space = sampler.sampled_space
    for i in range(10):
        sample = sampler.sample()
        assert sample_space.contains(sample)
    for sample in sampler.epoch_iter():
        assert sample_space.contains(sample)

