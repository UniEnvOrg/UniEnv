import typing
import pytest
from xbarray import ComputeBackend
from xbarray import numpy as NumpyComputeBackend
from xbarray import jax as JaxComputeBackend
from xbarray import pytorch as PyTorchComputeBackend

from unienv_interface.space.spaces import *
from unienv_interface.space.space_utils import gym_utils as space_gym_utils

import jax
import torch
import numpy as np

ALL_BACKENDS_AND_DEVICES : typing.Dict[ComputeBackend, typing.List[typing.Any]] = {
    NumpyComputeBackend: [],
    JaxComputeBackend: jax.devices(),
    PyTorchComputeBackend: ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda'],
}
# ALL_BACKENDS_AND_DTYPES : typing.Dict[ComputeBackend, typing.List[typing.Any]] = {
#     NumpyComputeBackend: NumpyComputeBackend.list_real_floating_dtypes() + NumpyComputeBackend.list_real_integer_dtypes(),
#     JaxComputeBackend: JaxComputeBackend.list_real_floating_dtypes() + JaxComputeBackend.list_real_integer_dtypes(),
#     PyTorchComputeBackend: PyTorchComputeBackend.list_real_floating_dtypes() + PyTorchComputeBackend.list_real_integer_dtypes(),
# }

SEEDS = [0, 1024, 2048]
NUM_SAMPLES_TEST = 20
NUM_SPACE_INSTANCES = 50

def perform_single_space_cross_backend_device_test(space : Space, rng : typing.Any) -> None:
    gym_space = space.to_gym_space()
    converted_back_space = space_gym_utils.from_gym_space(gym_space, space.backend, dtype=space.dtype, device=space.device)
    assert space == converted_back_space
    for _ in range(NUM_SAMPLES_TEST):
        rng, data = space.sample(rng)
        assert space.contains(data)
        assert converted_back_space.contains(data)
        gym_data = space_gym_utils.to_gym_data(space, data)
        assert gym_space.contains(gym_data)
        converted_back_data = space_gym_utils.from_gym_data(converted_back_space, gym_data)
        assert space.contains(converted_back_data)
        assert converted_back_space.contains(converted_back_data)
    for target_backend, target_devices in ALL_BACKENDS_AND_DEVICES.items():
        for target_device in target_devices:
            if target_backend == space.backend and target_device == space.device:
                continue
            converted_space = space.to(target_backend, target_device)
            for _ in range(NUM_SAMPLES_TEST):
                rng, data = space.sample(rng)
                converted_data = converted_space.from_other_backend(data, space.backend) if target_backend != space.backend else converted_space.from_same_backend(data)
                assert converted_space.contains(converted_data)
                converted_back_data = space.from_other_backend(converted_data, target_backend) if target_backend != space.backend else space.from_same_backend(converted_data)
                assert space.contains(converted_back_data)

def perform_box_test(backend : ComputeBackend, device : typing.Optional[typing.Any], rng : typing.Any, np_rng : np.random.Generator):
    def generate_random_box_params(rng : typing.Any, np_rng : np.random.Generator):
        shape_ndims = np_rng.integers(1, 5)
        shape = tuple([np_rng.integers(1, 10) for _ in range(shape_ndims)])
        dtype = np_rng.choice([backend.default_floating_dtype, backend.default_integer_dtype])
        rng, values_1 = backend.random.random_exponential(shape, lambd=0.5, rng=rng, dtype=dtype, device=device) if backend.dtype_is_real_floating(dtype) else backend.random.random_discrete_uniform(shape, 0, 127, rng=rng, dtype=dtype, device=device)
        rng, values_2 = backend.random.random_exponential(shape, lambd=0.5, rng=rng, dtype=dtype, device=device) if backend.dtype_is_real_floating(dtype) else backend.random.random_discrete_uniform(shape, 0, 127, rng=rng, dtype=dtype, device=device)
        min_values = backend.minimum(values_1, values_2)
        max_values = backend.maximum(values_1, values_2)
        rng, idx_min_unbound = backend.random.random_uniform(shape, rng=rng, low=0.0, high=1.0, device=device)
        rng, idx_max_unbound = backend.random.random_uniform(shape, rng=rng, low=0.0, high=1.0, device=device)
        idx_min_unbound = idx_min_unbound <= 0.2
        idx_max_unbound = idx_max_unbound <= 0.2
        min_values = backend.at(min_values)[idx_min_unbound].set(-backend.inf) if backend.dtype_is_real_floating(dtype) else min_values
        max_values = backend.at(max_values)[idx_max_unbound].set(backend.inf) if backend.dtype_is_real_floating(dtype) else max_values
        if backend.dtype_is_real_integer(dtype):
            min_values = backend.round(min_values)
            max_values = backend.round(max_values)
        return rng, np_rng, shape, dtype, min_values, max_values

    # Check shape inference
    rng, np_rng, shape, dtype, min_values, max_values = generate_random_box_params(rng, np_rng)
    space = BoxSpace(
        backend,
        min_values,
        max_values,
        dtype=dtype,
        device=device,
    )
    assert space.shape == shape

    # Do cross-backend and cross-device tests
    for space_id in range(NUM_SPACE_INSTANCES):
        rng, np_rng, shape, dtype, min_values, max_values = generate_random_box_params(rng, np_rng)
        print("Testing BoxSpace(", backend, device, ") with shape", shape, "and dtype", dtype)
        space = BoxSpace(
            backend,
            min_values,
            max_values,
            dtype=dtype,
            device=device,
        )
        perform_single_space_cross_backend_device_test(space, rng)
    
def test_box():
    for seed in SEEDS:
        for backend, devices in ALL_BACKENDS_AND_DEVICES.items():
            for device in devices:
                np_rng = np.random.default_rng(seed)
                rng = backend.random.random_number_generator(seed, device=device)
                perform_box_test(backend, device, rng, np_rng)