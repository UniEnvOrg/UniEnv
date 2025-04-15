from typing import Generic, TypeVar, Generic, Optional, Any, Dict as DictT, Tuple as TupleT, Sequence as SequenceT, Union as UnionT, List
from unienv_data import *
from unienv_data.storages.common import ListStorage
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_data.samplers import *
from unienv_interface.space import *
from unienv_interface.space import flatten_utils as sfu, batch_utils as sbu
from unienv_interface.backends.pytorch import PyTorchComputeBackend
import torch
import numpy as np
import pytest
from test_replay_buffer import perform_torch_replay_buffer_with_space_test


@pytest.mark.parametrize("capacity", [10, 50])
@pytest.mark.parametrize("seed", [0, 1024])
def test_step_sampler(
    capacity : int,
    seed : int
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    space = Box(
        PyTorchComputeBackend,
        0.0,
        1.0,
        dtype=torch.float32,
        device=device,
        shape=(3, 5, 2)
    )
    rb = perform_torch_replay_buffer_with_space_test(
        space,
        capacity,
        False,
        seed
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

@pytest.mark.parametrize("capacity", [10, 50])
@pytest.mark.parametrize("seed", [0, 1024])
def test_multiprocessing_sampler(
    capacity : int,
    seed : int
):
    device = torch.device("cpu")
    space = Box(
        PyTorchComputeBackend,
        0.0,
        1.0,
        dtype=torch.float32,
        device=device,
        shape=(3, 5, 2)
    )
    rb = perform_torch_replay_buffer_with_space_test(
        space,
        capacity,
        False,
        seed
    )
    sampler = StepSampler(
        rb,
        batch_size=capacity//2,
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

@pytest.mark.parametrize("capacity", [10, 50])
@pytest.mark.parametrize("seed", [0, 1024])
@pytest.mark.parametrize("prefetch_horizon", [0, 1, 2])
@pytest.mark.parametrize("postfetch_horizon", [0, 1, 2])
def test_slice_sampler(
    capacity : int,
    seed : int,
    prefetch_horizon : int,
    postfetch_horizon : int
):
    if prefetch_horizon == 0 and postfetch_horizon == 0:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dat_space = Box(
        PyTorchComputeBackend,
        0.0,
        1.0,
        dtype=torch.float32,
        device=device,
        shape=(3, 5, 2)
    )
    space = Dict(
        PyTorchComputeBackend,
        spaces={
            "dat": dat_space,
            "episode_id": Box(
                PyTorchComputeBackend,
                0,
                2,
                dtype=torch.int64,
                device=device,
                shape=()
            )
        },
        device=device
    )
    rb = perform_torch_replay_buffer_with_space_test(
        space,
        capacity,
        False,
        seed
    )
    
    sampler = SliceSampler(
        rb,
        batch_size=capacity//2,
        prefetch_horizon=prefetch_horizon,
        postfetch_horizon=postfetch_horizon,
        get_episode_id_fn=lambda x: x["episode_id"],
        seed=seed
    )
    sample_space = sampler.sampled_space
    for i in range(100):
        unfiltered_flat, metadata = sampler._get_unfiltered_flat_with_metadata(sampler.sample_index())
        unfiltered = sfu.unflatten_data(sample_space, unfiltered_flat, start_dim=2)
        filtered_flat, _, _ = sampler.unfiltered_to_filtered_flat(unfiltered_flat)
        filtered = sfu.unflatten_data(sample_space, filtered_flat, start_dim=2)
        
        ok_mask = unfiltered['episode_id'] == unfiltered['episode_id'][:, prefetch_horizon:prefetch_horizon+1]
        assert torch.allclose(unfiltered_flat[ok_mask], filtered_flat[ok_mask])

        sample = sampler.sample()
        # print("Sampled")
        # for key, value in sample.items():
        #     assert sample_space.spaces[key].contains(value)
        assert sample_space.contains(sample)

