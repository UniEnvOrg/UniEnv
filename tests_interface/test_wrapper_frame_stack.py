"""Tests for the stateful ``FrameStackWrapper`` and ``SpaceDataQueue``."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace
from unienv_interface.utils.framestack_queue import FuncSpaceDataQueue, SpaceDataQueue
from unienv_interface.wrapper import FrameStackWrapper
from tests_common.envs import StubEnv

ZERO_ACTION = np.zeros(2, np.float32)


def test_unbatched_obs_stack_rolls():
    env = StubEnv()
    wrapper = FrameStackWrapper(env, obs_stack_size=2)
    assert wrapper.observation_space["position"].shape == (3, 2)
    _, obs, _ = wrapper.reset(seed=0)
    assert wrapper.observation_space.contains(obs)
    start = env.position.copy()
    np.testing.assert_array_equal(obs["position"], np.stack([start] * 3))
    actions = [np.full(2, 0.1 * (i + 1), np.float32) for i in range(3)]
    history = [start]
    for a in actions:
        obs, *_ = wrapper.step(a)
        history.append(env.position.copy())
        np.testing.assert_allclose(obs["position"][-1], history[-1])
    np.testing.assert_allclose(obs["position"], np.stack(history[-3:]))
    np.testing.assert_allclose(obs["velocity"][-1], actions[-1])
    # reverse map returns the newest frame
    np.testing.assert_allclose(wrapper.reverse_map_observation(obs)["position"], history[-1])


def test_unbatched_past_actions_only():
    env = StubEnv()
    wrapper = FrameStackWrapper(env, action_stack_size=2, action_default_value=ZERO_ACTION)
    assert set(wrapper.observation_space.keys()) == {"position", "velocity", "past_actions"}
    assert "past_actions" not in env.observation_space.keys()
    _, obs, _ = wrapper.reset(seed=0)
    assert wrapper.observation_space.contains(obs)
    np.testing.assert_array_equal(obs["past_actions"], np.zeros((2, 2)))
    a1, a2, a3 = (np.full(2, v, np.float32) for v in (0.1, 0.2, 0.3))
    wrapper.step(a1)
    obs, *_ = wrapper.step(a2)
    np.testing.assert_allclose(obs["past_actions"], np.stack([a1, a2]))
    obs, *_ = wrapper.step(a3)
    np.testing.assert_allclose(obs["past_actions"], np.stack([a2, a3]))
    assert wrapper.reverse_map_observation(obs).keys() == {"position", "velocity"}


def test_batched_obs_and_action_stack_masked_reset():
    env = StubEnv(batch_size=3)
    wrapper = FrameStackWrapper(env, obs_stack_size=1, action_stack_size=2, action_default_value=ZERO_ACTION)
    assert wrapper.observation_space["position"].shape == (3, 2, 2)
    assert wrapper.observation_space["past_actions"].shape == (3, 2, 2)
    _, obs, _ = wrapper.reset(seed=0)
    assert wrapper.observation_space.contains(obs)
    action = np.arange(6, dtype=np.float32).reshape(3, 2) / 10
    obs, *_ = wrapper.step(action)
    before = {k: v.copy() for k, v in obs.items()}

    mask = np.array([True, False, True])
    _, reset_obs, info = wrapper.reset(mask=mask, seed=5)
    # only the selected rows are returned
    assert reset_obs["position"].shape == (2, 2, 2)
    assert reset_obs["past_actions"].shape == (2, 2, 2)
    np.testing.assert_array_equal(info["step"], [0, 0])
    np.testing.assert_allclose(reset_obs["position"][:, 0], env.position[mask])
    np.testing.assert_allclose(reset_obs["position"][:, 1], env.position[mask])
    np.testing.assert_array_equal(reset_obs["past_actions"], 0.0)
    merged = wrapper.update_observation_post_reset(before, reset_obs, mask)
    assert wrapper.observation_space.contains(merged)
    np.testing.assert_allclose(merged["position"][1], before["position"][1])

    obs, *_ = wrapper.step(action)
    # untouched row keeps its history, reset rows restarted from the reset obs
    np.testing.assert_allclose(obs["position"][1, 0], before["position"][1, 1])
    np.testing.assert_allclose(obs["past_actions"][1], np.stack([action[1], action[1]]))
    np.testing.assert_allclose(obs["past_actions"][0], np.stack([ZERO_ACTION, action[0]]))
    np.testing.assert_allclose(obs["position"][0, 0], merged["position"][0, 1])


def test_space_data_queue_masked_reset():
    space = BoxSpace(B, -10, 10, np.float32, shape=(3, 2))
    queue = SpaceDataQueue(space, 3, maxlen=2)
    queue.reset(np.zeros((3, 2), np.float32))
    queue.add(np.ones((3, 2), np.float32))
    queue.reset(np.full((1, 2), 5.0, np.float32), mask=np.array([False, True, False]))
    out = queue.get_output_data()
    assert out.shape == (3, 2, 2)
    np.testing.assert_array_equal(out[1], 5.0)
    np.testing.assert_array_equal(out[0], [[0, 0], [1, 1]])


def test_space_data_queue_rejects_mask_when_unbatched():
    space = BoxSpace(B, -10, 10, np.float32, shape=(2,))
    func_queue = FuncSpaceDataQueue(space, None, maxlen=2)
    state = func_queue.init(np.zeros(2, np.float32))
    with pytest.raises(AssertionError):
        func_queue.reset(state, np.zeros(2, np.float32), mask=np.array([True]))
    with pytest.raises(AssertionError):
        SpaceDataQueue(space, None, 2).reset(np.zeros(2, np.float32), mask=np.array([True]))
