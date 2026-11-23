"""Tests for the functional ``FuncTransformWrapper``."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.env_base.funcenv import FuncEnvBasedEnv
from unienv_interface.func_wrapper import FuncTransformWrapper
from unienv_interface.space import BoxSpace
from unienv_interface.transformations.batch_and_unbatch import BatchifyTransformation
from unienv_interface.transformations.filter_dict import DictIncludeKeyTransformation
from unienv_interface.transformations.rescale import RescaleTransformation
from tests_common.envs import StubFuncEnv


@pytest.mark.parametrize("batch_size", [None, 3])
def test_transforms_context_observation_and_action(batch_size):
    func_env = StubFuncEnv(batch_size)
    inner_shape = func_env.action_space.shape
    target_action_space = BoxSpace(B, -2.0, 2.0, np.float32, shape=inner_shape)
    wrapper = FuncTransformWrapper(
        func_env,
        context_transformation=BatchifyTransformation(),
        observation_transformation=DictIncludeKeyTransformation(["position"]),
        action_transformation=RescaleTransformation(-1.0, 1.0),
        target_action_space=target_action_space,
    )
    assert wrapper.observation_space.keys() == {"position"}
    assert wrapper.context_space.shape == (1, *func_env.context_space.shape)
    assert wrapper.action_space is target_action_space

    state, context, obs, _ = wrapper.initial(seed=4)
    assert wrapper.context_space.contains(context)
    assert wrapper.observation_space.contains(obs)
    np.testing.assert_array_equal(context[0], np.full(inner_shape, 4.0))

    start = state["position"].copy()
    state, obs, *_ = wrapper.step(state, np.full(inner_shape, 2.0, np.float32))
    np.testing.assert_allclose(obs["position"], start + 1.0)  # 2 in [-2,2] -> 1 in [-1,1]
    assert wrapper.observation_space.contains(obs)

    state, context, obs, _ = wrapper.reset(state, seed=5)
    assert wrapper.context_space.contains(context)
    assert obs.keys() == {"position"}


def test_no_transformation_is_passthrough():
    func_env = StubFuncEnv()
    wrapper = FuncTransformWrapper(func_env)
    assert wrapper.observation_space is func_env.observation_space
    assert wrapper.action_space is func_env.action_space
    assert wrapper.context_space is func_env.context_space
    env = FuncEnvBasedEnv(wrapper)
    env.reset(seed=0)
    env.step(np.zeros(2, np.float32))
    env.close()


def test_action_transformation_requires_target_space():
    with pytest.raises(AssertionError):
        FuncTransformWrapper(StubFuncEnv(), action_transformation=RescaleTransformation(-1.0, 1.0))


@pytest.mark.parametrize("mask", [[False, True, False], [True, False, True], [False, False, False]])
def test_masked_reset_uses_masked_space_with_per_env_bounds(mask):
    """Regression: the masked rows were transformed with the full-batch space (wrong shape / broadcast error)."""
    high = np.array([[10.0, 10.0], [20.0, 20.0], [40.0, 40.0]], np.float32)
    func_env = StubFuncEnv(3)
    func_env.context_space = BoxSpace(B, np.zeros((3, 2), np.float32), high, np.float32, shape=(3, 2))
    wrapper = FuncTransformWrapper(func_env, context_transformation=RescaleTransformation(0.0, 1.0))
    state, _, _, _ = wrapper.initial(seed=1)
    mask = np.array(mask)
    state, context, _, _ = wrapper.reset(state, mask=mask, seed=4)
    assert context.shape == (int(mask.sum()), 2)
    np.testing.assert_allclose(context, state["goal"][mask] / high[mask], rtol=1e-6)
