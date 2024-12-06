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
        torch.float32,
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


