"""Tests for the functional ``FuncFrameStackWrapper``."""

import numpy as np
import pytest

from unienv_interface.env_base.funcenv import FuncEnvBasedEnv
from unienv_interface.func_wrapper import FuncFrameStackWrapper, FuncFrameStackWrapperState
from tests_common.envs import StubFuncEnv

ZERO_ACTION = np.zeros(2, np.float32)


class _RecordingFuncEnv(StubFuncEnv):
    """Records the state objects handed to close / render hooks."""

    def __init__(self, batch_size=None):
        super().__init__(batch_size)
        self.seen_states = []

    def render_init(self, state, **kwargs):
        self.seen_states.append(("render_init", state))
        return super().render_init(state, **kwargs)

    def render_image(self, state, render_state):
        self.seen_states.append(("render_image", state))
        return super().render_image(state, render_state)

    def render_close(self, state, render_state):
        self.seen_states.append(("render_close", state))
        return super().render_close(state, render_state)

    def close(self, state):
        self.seen_states.append(("close", state))
        super().close(state)


def test_obs_and_action_stack_unbatched():
    func_env = StubFuncEnv()
    wrapper = FuncFrameStackWrapper(func_env, obs_stack_size=1, action_stack_size=2, action_default_value=ZERO_ACTION)
    assert wrapper.observation_space["position"].shape == (2, 2)
    assert wrapper.observation_space["past_actions"].shape == (2, 2)
    assert "past_actions" not in func_env.observation_space.keys()

    state, _, obs, _ = wrapper.initial(seed=0)
    assert isinstance(state, FuncFrameStackWrapperState)
    assert wrapper.observation_space.contains(obs)
    start = state.env_state["position"].copy()
    np.testing.assert_array_equal(obs["position"], np.stack([start, start]))
    np.testing.assert_array_equal(obs["past_actions"], 0.0)

    a1, a2, a3 = (np.full(2, v, np.float32) for v in (0.1, 0.2, 0.3))
    state, obs, *_ = wrapper.step(state, a1)
    state, obs, *_ = wrapper.step(state, a2)
    np.testing.assert_allclose(obs["past_actions"], np.stack([a1, a2]))
    state2, obs3, *_ = wrapper.step(state, a3)
    np.testing.assert_allclose(obs3["past_actions"], np.stack([a2, a3]))
    np.testing.assert_allclose(obs3["position"][-1], state2.env_state["position"])
    np.testing.assert_allclose(obs3["position"][0], state.env_state["position"])
    # functional: the old state is unchanged
    _, obs3b, *_ = wrapper.step(state, a3)
    np.testing.assert_allclose(obs3b["past_actions"], obs3["past_actions"])

    state, _, obs, _ = wrapper.reset(state2, seed=1)
    np.testing.assert_array_equal(obs["past_actions"], 0.0)
    np.testing.assert_array_equal(obs["position"][0], obs["position"][1])


def test_masked_reset_batched():
    func_env = StubFuncEnv(batch_size=3)
    wrapper = FuncFrameStackWrapper(func_env, obs_stack_size=1, action_stack_size=1, action_default_value=ZERO_ACTION)
    state, _, obs, _ = wrapper.initial(seed=0)
    assert wrapper.observation_space.contains(obs)
    action = np.full((3, 2), 0.25, np.float32)
    state, obs, *_ = wrapper.step(state, action)
    before = obs

    mask = np.array([False, True, True])
    state, _, reset_obs, info = wrapper.reset(state, seed=3, mask=mask)
    assert reset_obs["position"].shape == (2, 2, 2)
    assert reset_obs["past_actions"].shape == (2, 1, 2)
    np.testing.assert_array_equal(reset_obs["past_actions"], 0.0)
    np.testing.assert_allclose(reset_obs["position"][:, 0], state.env_state["position"][mask])
    merged = wrapper.update_observation_post_reset(before, reset_obs, mask)
    assert wrapper.observation_space.contains(merged)
    np.testing.assert_allclose(merged["past_actions"][0], before["past_actions"][0])

    state, obs, *_ = wrapper.step(state, action)
    np.testing.assert_allclose(obs["position"][0, 0], before["position"][0, 1])


def test_wrapper_unwraps_its_state_for_render_and_close():
    func_env = _RecordingFuncEnv()
    env = FuncEnvBasedEnv(FuncFrameStackWrapper(func_env, obs_stack_size=1))
    env.reset(seed=0)
    env.step(ZERO_ACTION)
    frame = env.render()
    assert frame.shape == (4, 6, 3)
    env.close()
    assert [name for name, _ in func_env.seen_states] == ["render_init", "render_image", "render_close", "close"]
    for name, state in func_env.seen_states:
        assert isinstance(state, dict), f"{name} received {type(state).__name__}"


def test_action_stack_requires_default_value():
    with pytest.raises(AssertionError):
        FuncFrameStackWrapper(StubFuncEnv(), action_stack_size=1)


def test_close_before_initial_is_forwarded_without_state():
    func_env = _RecordingFuncEnv()
    env = FuncEnvBasedEnv(FuncFrameStackWrapper(func_env, obs_stack_size=1))
    env.close()  # never reset
    assert func_env.seen_states == [("close", None)]


@pytest.mark.parametrize("mask", [np.array([False, True, True]), None], ids=["masked", "full"])
def test_reset_keeps_previous_obs_and_state_intact(mask):
    # FuncFrameStackWrapper must be pure: resetting a state may not write into
    # the observation returned earlier nor into the state that was passed in.
    func_env = StubFuncEnv(batch_size=3)
    wrapper = FuncFrameStackWrapper(func_env, obs_stack_size=1, action_stack_size=1, action_default_value=ZERO_ACTION)
    state, *_ = wrapper.initial(seed=0)
    action = np.full((3, 2), 0.25, np.float32)
    state, obs, *_ = wrapper.step(state, action)
    obs_snapshot = {k: v.copy() for k, v in obs.items()}
    queue_snapshot = state.obs_queue_state.data["position"].copy()
    action_snapshot = state.action_queue_state.data.copy()

    wrapper.reset(state, seed=3, mask=mask)

    for key, value in obs_snapshot.items():
        np.testing.assert_array_equal(obs[key], value, err_msg=key)
    np.testing.assert_array_equal(state.obs_queue_state.data["position"], queue_snapshot)
    np.testing.assert_array_equal(state.action_queue_state.data, action_snapshot)
    # stepping the untouched old state again reproduces the original observation
    _, obs_again, *_ = wrapper.step(state, action)
    np.testing.assert_array_equal(obs_again["past_actions"][:, :-1], obs_snapshot["past_actions"][:, 1:])
