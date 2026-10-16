"""Property-style tests for ``BoxSpace``: sampling, gym round trips and backend/device conversion."""
import typing

import numpy as np
import pytest

from unienv_interface.backends import ComputeBackend
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space.spaces import *
from unienv_interface.space.space_utils import gym_utils as space_gym_utils
from tests_common.backends import backend_params, backends_and_devices, cpu_backend_params
from tests_common.space_conversion import can_convert

SEEDS = [0, 1024]
NUM_SAMPLES_TEST = 5
NUM_SPACE_INSTANCES = 8
NUM_SPACE_INSTANCES_JAX = 2  # jax recompiles for every new random shape


def perform_single_space_cross_backend_device_test(space: Space, rng: typing.Any) -> None:
    gym_space = space_gym_utils.to_gym_space(space)
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
    for target_backend, target_device in backends_and_devices():
        if target_backend == space.backend and target_device == space.device:
            continue
        # Conversions xbarray cannot perform on this machine (see tests_common.space_conversion)
        if not can_convert(space.backend, space.device, target_backend):
            continue
        can_convert_back = can_convert(target_backend, target_device, space.backend, origin_backend=space.backend)
        converted_space = space.to(target_backend, target_device)
        for _ in range(NUM_SAMPLES_TEST):
            rng, data = space.sample(rng)
            converted_data = space.data_to(data, target_backend, target_device)
            assert converted_space.contains(converted_data)
            if can_convert_back:
                converted_back_data = converted_space.data_to(converted_data, space.backend, space.device)
                assert space.contains(converted_back_data)


def perform_box_test(backend: ComputeBackend, device: typing.Optional[typing.Any], rng: typing.Any, np_rng: np.random.Generator):
    def generate_random_box_params(rng: typing.Any, np_rng: np.random.Generator):
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
    for space_id in range(NUM_SPACE_INSTANCES_JAX if backend.simplified_name == "jax" else NUM_SPACE_INSTANCES):
        rng, np_rng, shape, dtype, min_values, max_values = generate_random_box_params(rng, np_rng)
        space = BoxSpace(
            backend,
            min_values,
            max_values,
            dtype=dtype,
            device=device,
        )
        perform_single_space_cross_backend_device_test(space, rng)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("backend,device", backend_params())
def test_box(backend, device, seed):
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed, device=device)
    perform_box_test(backend, device, rng, np_rng)


def test_box_constructor_validation():
    B = NumpyComputeBackend
    with pytest.raises(AssertionError):
        BoxSpace(B, 1.0, 0.0, np.float32, shape=(2,))
    with pytest.raises(AssertionError):
        BoxSpace(B, np.array([0.0, np.nan]), 1.0, np.float32)
    with pytest.raises(AssertionError):
        BoxSpace(B, 0, 1, np.bool_, shape=(1,))
    with pytest.raises(ValueError):
        BoxSpace(B, [0.0], [1.0], np.float32)  # shape cannot be inferred from lists
    space = BoxSpace(B, np.zeros((1, 3)), np.ones((2, 1)), np.float32)
    assert space.shape == (2, 3)
    assert space.is_bounded() and space.is_bounded("below") and space.is_bounded("above")
    half = BoxSpace(B, 0.0, np.inf, np.float32, shape=(2,))
    assert half.is_bounded("below") and not half.is_bounded("above") and not half.is_bounded()
    with pytest.raises(ValueError):
        half.is_bounded("sideways")
    np.testing.assert_array_equal(half.clip(np.array([-1.0, 5.0], dtype=np.float32)), [0.0, 5.0])
    assert half.create_empty().shape == (2,)
    assert "BoxSpace" in repr(space) and "[" in str(space)


def test_box_float_dtype_is_strict():
    """A float64 sample is not a member of a float32 box (no implicit narrowing)."""
    space = BoxSpace(NumpyComputeBackend, 0.0, 1.0, np.float32, shape=(2,))
    assert space.contains(np.full((2,), 0.5, dtype=np.float32))
    assert not space.contains(np.full((2,), 0.5, dtype=np.float64))
    assert not space.contains(np.full((3,), 0.5, dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint8])
def test_full_range_integer_box_samples_vary(dtype):
    space = BoxSpace(NumpyComputeBackend, -np.inf, np.inf, dtype, shape=(16,))
    rng = NumpyComputeBackend.random.random_number_generator(0)
    rng, x = space.sample(rng)
    assert space.contains(x)
    assert len(np.unique(x)) > 1


def test_integer_box_samples_cover_closed_interval():
    space = BoxSpace(NumpyComputeBackend, -1, 2, np.int64, shape=(64,))
    _, x = space.sample(NumpyComputeBackend.random.random_number_generator(0))
    assert set(x.tolist()) == {-1, 0, 1, 2}


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_infinite_bounds_with_default_integer_dtype(backend, device):
    # jax without x64 materialises the default ``int`` as int32 while iinfo(int) reports int64
    space = BoxSpace(backend, -np.inf, np.inf, backend.default_integer_dtype, shape=(4,), device=device)
    info = backend.iinfo(space.low.dtype)
    assert int(space.low.min()) == int(info.min) and int(space.high.max()) == int(info.max)
    _, x = space.sample(backend.random.random_number_generator(0, device=device))
    assert space.contains(x)
