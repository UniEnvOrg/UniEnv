"""Tests for ``gym_utils`` conversions between UniEnv and gymnasium spaces/data."""

import gymnasium as gym
import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import (
    BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, TextSpace, TupleSpace, UnionSpace,
)
from unienv_interface.space.space_utils import gym_utils as sgu
from tests_common.backends import cpu_backend_params
from tests_common.trees import assert_tree_equal


@pytest.mark.parametrize("start", [0, -2, 5])
def test_discrete_bounds_are_closed(start):
    gym_space = gym.spaces.Discrete(3, start=start)
    space = sgu.from_gym_space(gym_space, B)
    assert isinstance(space, BoxSpace) and space.shape == ()
    for value in range(start - 1, start + 5):
        expected = gym_space.contains(value)
        assert space.contains(np.array(value, dtype=space.dtype)) == expected, value


def test_multi_discrete_bounds_are_closed():
    gym_space = gym.spaces.MultiDiscrete([2, 3], start=[1, 0])
    space = sgu.from_gym_space(gym_space, B)
    assert space.contains(np.array([2, 2]))
    assert not space.contains(np.array([3, 2]))
    assert not space.contains(np.array([2, 3]))
    assert not space.contains(np.array([0, 0]))


@pytest.mark.parametrize("gym_space", [
    gym.spaces.Discrete(4),
    gym.spaces.Discrete(4, start=-1),
    gym.spaces.MultiDiscrete([2, 5]),
    gym.spaces.Box(-1, 1, (2,), np.float32),
    gym.spaces.MultiBinary(3),
    gym.spaces.MultiBinary(()),
], ids=["discrete", "discrete_start", "multidiscrete", "box", "multibinary", "multibinary_scalar"])
def test_gym_samples_convert_into_space(gym_space):
    gym_space.seed(0)
    space = sgu.from_gym_space(gym_space, B)
    for _ in range(20):
        gx = gym_space.sample()
        x = sgu.from_gym_data(space, gx)
        assert space.contains(x), (gx, x)
        assert gym_space.contains(sgu.to_gym_data(space, x))


def test_from_gym_data_accepts_python_and_numpy_scalars():
    space = sgu.from_gym_space(gym.spaces.Discrete(5), B)
    for value in (3, np.int64(3), np.int32(3)):
        x = sgu.from_gym_data(space, value)
        assert space.contains(x) and int(x) == 3
    binary = BinarySpace(B, ())
    for value in (True, np.bool_(True)):
        assert binary.contains(sgu.from_gym_data(binary, value))
    with pytest.raises(TypeError):
        sgu.from_gym_data(space, "3")


def test_union_from_gym_data_keeps_index():
    gym_space = gym.spaces.OneOf([gym.spaces.Box(0, 1, (2,), np.float32), gym.spaces.Discrete(2)])
    gym_space.seed(1)
    space = sgu.from_gym_space(gym_space, B)
    for _ in range(10):
        gx = gym_space.sample()
        x = sgu.from_gym_data(space, gx)
        assert space.contains(x)
        assert x[0] == gx[0]
        back = sgu.to_gym_data(space, x)
        assert gym_space.contains(back)


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_nested_roundtrip(backend, device):
    # Use the materialised float dtype: jax keeps ``float`` and ``float32`` as distinct dtype objects
    float_dtype = backend.zeros((), dtype=backend.default_floating_dtype).dtype
    space = DictSpace(backend, {
        "box": BoxSpace(backend, -1.0, 1.0, float_dtype, shape=(2,), device=device),
        "bin": BinarySpace(backend, (3,), device=device),
        "tup": TupleSpace(backend, [TextSpace(backend, 4, charset="ab", device=device)]),
    }, device=device)
    gym_space = sgu.to_gym_space(space)
    assert isinstance(gym_space, gym.spaces.Dict)
    assert isinstance(gym_space["bin"], gym.spaces.MultiBinary) and gym_space["bin"].shape == (3,)
    back_space = sgu.from_gym_space(gym_space, backend, device=device)
    assert back_space == space
    rng = backend.random.random_number_generator(0, device=device)
    for _ in range(5):
        rng, x = space.sample(rng)
        gx = sgu.to_gym_data(space, x)
        assert gym_space.contains(gx)
        y = sgu.from_gym_data(back_space, gx)
        assert space.contains(y)
        assert_tree_equal(y, x)


def test_unsupported_spaces_raise():
    with pytest.raises(TypeError):
        sgu.to_gym_space(DynamicBoxSpace(B, 0.0, 1.0, (1,), (3,), np.float32))
    with pytest.raises(TypeError):
        sgu.from_gym_space(object(), B)
    with pytest.raises(TypeError):
        sgu.to_gym_data(DynamicBoxSpace(B, 0.0, 1.0, (1,), (3,), np.float32), np.zeros(2))
    with pytest.raises(AssertionError):
        sgu.to_gym_space(TextSpace(B, 3))  # gymnasium needs an explicit charset
