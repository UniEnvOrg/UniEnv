"""Tests for ``ActionTransformWrapper`` / ``ActionRescaleWrapper``."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace
from unienv_interface.transformations.rescale import RescaleTransformation
from unienv_interface.wrapper import ActionRescaleWrapper, ActionTransformWrapper
from tests_common.envs import StubEnv


@pytest.mark.parametrize("batch_size", [None, 3])
def test_action_rescale_maps_unit_box_to_env_box(batch_size):
    env = StubEnv(batch_size, action_low=0.0, action_high=10.0)
    wrapper = ActionRescaleWrapper(env, new_low=-1.0, new_high=1.0)
    assert isinstance(wrapper.action_space, BoxSpace)
    np.testing.assert_allclose(wrapper.action_space.low, -1.0)
    np.testing.assert_allclose(wrapper.action_space.high, 1.0)
    assert wrapper.action_space.shape == env.action_space.shape

    wrapper.reset(seed=0)
    shape = env.action_space.shape
    wrapper.step(np.zeros(shape, np.float32))
    np.testing.assert_allclose(env.last_action, np.full(shape, 5.0))
    wrapper.step(np.full(shape, -1.0, np.float32))
    np.testing.assert_allclose(env.last_action, np.zeros(shape))
    wrapper.step(np.full(shape, 1.0, np.float32))
    np.testing.assert_allclose(env.last_action, np.full(shape, 10.0))

    # reverse mapping goes back to the wrapper-facing range
    np.testing.assert_allclose(wrapper.reverse_map_action(np.full(shape, 5.0, np.float32)), 0.0, atol=1e-6)
    # sampled actions live in the wrapper-facing space
    assert wrapper.action_space.contains(wrapper.sample_action())


def test_action_transform_wrapper_rejects_incompatible_target_space():
    env = StubEnv(action_low=0.0, action_high=10.0)
    target = BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,))
    # maps [-1,1] -> [-1,1], which is not the env's [0,10] action space
    with pytest.raises(AssertionError):
        ActionTransformWrapper(env, RescaleTransformation(-1.0, 1.0), target)
