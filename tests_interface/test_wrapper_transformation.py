"""Tests for ``TransformWrapper`` / ``ContextObservationTransformWrapper``."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace
from unienv_interface.transformations.batch_and_unbatch import BatchifyTransformation
from unienv_interface.transformations.filter_dict import DictIncludeKeyTransformation
from unienv_interface.transformations.rescale import RescaleTransformation
from unienv_interface.wrapper import ContextObservationTransformWrapper
from unienv_interface.wrapper.transformation import TransformWrapper
from tests_common.envs import StubEnv


@pytest.mark.parametrize("cls", [TransformWrapper, ContextObservationTransformWrapper])
def test_context_space_forwarded_without_context_transformation(any_stub_env, cls):
    wrapper = cls(any_stub_env, observation_transformation=DictIncludeKeyTransformation(["position"]))
    assert wrapper.context_space is any_stub_env.context_space
    context, obs, _ = wrapper.reset(seed=2)
    assert wrapper.context_space.contains(context)
    assert wrapper.observation_space.contains(obs)
    assert obs.keys() == {"position"}


@pytest.mark.parametrize("batch_size", [None, 3])
def test_transform_wrapper_all_transformations(batch_size):
    env = StubEnv(batch_size)
    shape = env.action_space.shape
    target_action_space = BoxSpace(B, 0.0, 4.0, np.float32, shape=shape)
    wrapper = TransformWrapper(
        env,
        context_transformation=BatchifyTransformation(),
        observation_transformation=DictIncludeKeyTransformation(["velocity"]),
        action_transformation=RescaleTransformation(-1.0, 1.0),
        target_action_space=target_action_space,
    )
    assert wrapper.action_space is target_action_space
    context, obs, _ = wrapper.reset(seed=3)
    assert wrapper.context_space.contains(context)
    np.testing.assert_array_equal(context[0], np.full(shape, 3.0))
    obs, *_ = wrapper.step(np.full(shape, 3.0, np.float32))
    np.testing.assert_allclose(env.last_action, 0.5)
    np.testing.assert_allclose(obs["velocity"], 0.5)
    assert wrapper.observation_space.contains(obs)
    with pytest.raises(AssertionError):
        wrapper.set_action_transformation(RescaleTransformation(-1.0, 1.0), None)


def test_context_observation_transform_wrapper_reverse_maps():
    env = StubEnv()
    wrapper = ContextObservationTransformWrapper(env, context_transformation=BatchifyTransformation())
    context, obs, _ = wrapper.reset(seed=1)
    assert context.shape == (1, 2)
    np.testing.assert_array_equal(wrapper.reverse_map_context(context), env.goal)
    assert wrapper.reverse_map_observation(obs) is obs
    non_invertible = ContextObservationTransformWrapper(
        env, observation_transformation=DictIncludeKeyTransformation(["position"]))
    with pytest.raises(NotImplementedError):
        non_invertible.reverse_map_observation({"position": np.zeros(2, np.float32)})
    with pytest.raises(AssertionError):
        ContextObservationTransformWrapper(env)
