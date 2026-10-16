"""Tests for ``FlattenActionWrapper`` / ``FlattenContextObservationWrapper``."""

from typing import Optional

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.space.space_utils import flatten_utils as sfu
from unienv_interface.wrapper import FlattenActionWrapper, FlattenContextObservationWrapper
from tests_common.envs import StubEnv
from tests_common.trees import assert_tree_equal


def _shape(batch_size: Optional[int], *inner: int):
    return tuple(inner) if batch_size is None else (batch_size, *inner)


class DictActionEnv(StubEnv):
    """StubEnv whose action is ``{"move": Box(2), "scale": Box(1)}``; applied action = move * scale."""

    def __init__(self, batch_size=None):
        super().__init__(batch_size)
        self.action_space = DictSpace(B, {
            "move": BoxSpace(B, -1.0, 1.0, np.float32, shape=_shape(batch_size, 2)),
            "scale": BoxSpace(B, 0.0, 2.0, np.float32, shape=_shape(batch_size, 1)),
        })
        self.last_dict_action = None

    def step(self, action):
        self.last_dict_action = action
        return super().step(np.asarray(action["move"] * action["scale"], np.float32))


@pytest.mark.parametrize("batch_size", [None, 3])
def test_flatten_observation_maps_dict_to_flat_and_back(batch_size):
    env = StubEnv(batch_size)
    wrapper = FlattenContextObservationWrapper(env)
    expected_space = sfu.flatten_space(env.observation_space, start_dim=0 if batch_size is None else 1)
    assert wrapper.observation_space == expected_space
    assert wrapper.observation_space.shape == _shape(batch_size, 4)

    context, obs, _ = wrapper.reset(seed=1)
    assert wrapper.observation_space.contains(obs)
    inner_obs = env._obs()
    assert_tree_equal(wrapper.reverse_map_observation(obs), inner_obs)
    # keys are flattened in sorted order: position then velocity
    np.testing.assert_array_equal(obs[..., :2], inner_obs["position"])
    assert_tree_equal(context, env.goal)  # context untouched by default

    obs, *_ = wrapper.step(np.full(env.action_space.shape, 0.5, np.float32))
    assert wrapper.observation_space.contains(obs)
    assert_tree_equal(wrapper.reverse_map_observation(obs), env._obs())
    np.testing.assert_array_equal(wrapper.map_observation(env._obs()), obs)


@pytest.mark.parametrize("batch_size", [None, 3])
def test_flatten_context(batch_size):
    env = StubEnv(batch_size)
    wrapper = FlattenContextObservationWrapper(env, flatten_context=True, flatten_observation=False)
    context, obs, _ = wrapper.reset(seed=2)
    assert wrapper.context_space.contains(context)
    assert wrapper.observation_space is env.observation_space
    assert_tree_equal(wrapper.reverse_map_context(context), env.goal)
    assert_tree_equal(obs, env._obs())


@pytest.mark.parametrize("batch_size", [None, 3])
def test_flatten_action_round_trip(batch_size):
    env = DictActionEnv(batch_size)
    wrapper = FlattenActionWrapper(env)
    assert wrapper.action_space.shape == _shape(batch_size, 3)
    wrapper.reset(seed=0)
    flat = wrapper.sample_action()
    assert wrapper.action_space.contains(flat)
    wrapper.step(flat)
    assert env.action_space.contains(env.last_dict_action)
    np.testing.assert_array_equal(wrapper.reverse_map_action(env.last_dict_action), flat)
    np.testing.assert_array_equal(env.last_dict_action["move"], flat[..., :2])
