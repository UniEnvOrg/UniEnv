"""Tests for ``ToGymnasiumEnv`` / ``FromGymnasiumEnv``."""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.wrapper.gym_compat import FromGymnasiumEnv, ToGymnasiumEnv
from tests_common.env_gym import TinyGymEnv
from tests_common.envs import StubEnv


class NoContextStubEnv(StubEnv):
    def __init__(self, batch_size=None):
        super().__init__(batch_size)
        self.context_space = None

    def reset(self, **kwargs):
        _, obs, info = super().reset(**kwargs)
        return None, obs, info


class BoxObsStubEnv(StubEnv):
    """Non-dict observation (position only) and no context."""

    def __init__(self):
        super().__init__()
        self.observation_space = self.observation_space["position"]
        self.context_space = None

    def reset(self, **kwargs):
        _, obs, info = super().reset(**kwargs)
        return None, obs["position"], info

    def step(self, action):
        obs, *rest = super().step(action)
        return (obs["position"], *rest)


# ---------------------------------------------------------------------------
# ToGymnasiumEnv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env_cls", [StubEnv, NoContextStubEnv, BoxObsStubEnv])
def test_to_gymnasium_passes_check_env(env_cls):
    gym_env = ToGymnasiumEnv(env_cls())
    check_env(gym_env, skip_render_check=False)
    gym_env.close()


def test_to_gymnasium_observations_do_not_share_the_episode_context():
    gym_env = ToGymnasiumEnv(StubEnv())
    reset_obs, _ = gym_env.reset(seed=3)
    step_obs, *_ = gym_env.step(gym_env.action_space.sample())
    assert step_obs["context"] is not reset_obs["context"]
    reset_obs["context"][:] = -100.0
    next_obs, *_ = gym_env.step(gym_env.action_space.sample())
    np.testing.assert_array_equal(next_obs["context"], np.full(2, 3.0, np.float32))
    gym_env.close()


def test_to_gymnasium_combines_context_and_observation():
    env = StubEnv()
    gym_env = ToGymnasiumEnv(env)
    assert isinstance(gym_env.observation_space, gym.spaces.Dict)
    assert set(gym_env.observation_space.spaces) == {"position", "velocity", "context"}
    obs, info = gym_env.reset(seed=4)
    assert gym_env.observation_space.contains(obs)
    np.testing.assert_array_equal(obs["context"], [4.0, 4.0])
    assert info == {"step": 0}
    obs, reward, terminated, truncated, info = gym_env.step(np.full(2, 0.5, np.float32))
    assert gym_env.observation_space.contains(obs)
    np.testing.assert_array_equal(obs["context"], [4.0, 4.0])
    assert isinstance(reward, float) and terminated is False and truncated is False
    assert gym_env.metadata["render_fps"] == StubEnv.render_fps
    assert gym_env.render_mode == "rgb_array"
    assert gym_env.render().shape == (4, 6, 3)
    assert "StubEnv" in str(gym_env)
    gym_env.close()
    assert env.closed == 1


def test_to_gymnasium_forwards_reset_options_as_kwargs():
    env = StubEnv()
    gym_env = ToGymnasiumEnv(env)
    gym_env.reset(seed=1, options={"difficulty": 3})
    assert env.reset_calls[-1]["difficulty"] == 3
    assert env.reset_calls[-1]["seed"] == 1


def test_to_gymnasium_no_context_keeps_observation_space():
    env = NoContextStubEnv()
    gym_env = ToGymnasiumEnv(env)
    assert set(gym_env.observation_space.spaces) == {"position", "velocity"}
    obs, _ = gym_env.reset(seed=0)
    assert gym_env.observation_space.contains(obs)


def test_to_gymnasium_np_random_is_a_generator():
    gym_env = ToGymnasiumEnv(StubEnv())
    rng = gym_env.np_random
    assert isinstance(rng, np.random.Generator)
    assert gym_env.np_random is rng
    gym_env.action_space.seed(0)
    assert gym_env.action_space.contains(gym_env.action_space.sample())


def test_to_gymnasium_rejects_batched_env_and_masks():
    with pytest.raises(AssertionError):
        ToGymnasiumEnv(StubEnv(batch_size=2))
    with pytest.raises(AssertionError):
        ToGymnasiumEnv(StubEnv()).reset(mask=np.array([True]))


# ---------------------------------------------------------------------------
# FromGymnasiumEnv
# ---------------------------------------------------------------------------

def test_from_gymnasium_round_trip():
    gym_env = TinyGymEnv()
    env = FromGymnasiumEnv(gym_env)
    assert env.backend is B and env.device is None and env.batch_size is None
    assert isinstance(env.action_space, BoxSpace) and env.action_space.shape == (2,)
    assert env.context_space is None
    assert env.render_fps == 15 and env.render_mode == "rgb_array"

    context, obs, info = env.reset(seed=3)
    assert context is None and info == {"reset": True}
    assert env.observation_space.contains(obs)
    ref_obs, _ = TinyGymEnv().reset(seed=3)
    np.testing.assert_array_equal(obs, ref_obs)

    action = env.sample_action()
    assert env.action_space.contains(action)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert reward == pytest.approx(-np.abs(obs).sum())
    assert info == {"step": True}
    assert env.render().shape == (4, 6, 3)
    env.close()
    assert gym_env.closed


def test_from_gymnasium_reset_kwargs_become_options():
    gym_env = TinyGymEnv()
    env = FromGymnasiumEnv(gym_env)
    env.reset(seed=0, options={"a": 1})
    assert gym_env.last_options == {"a": 1}
    env.reset(seed=0, level=2)
    assert gym_env.last_options == {"level": 2}


def test_from_gymnasium_sample_action_advances_rng():
    env = FromGymnasiumEnv(TinyGymEnv())
    env.reset(seed=0)
    a1, a2 = env.sample_action(), env.sample_action()
    assert not np.array_equal(a1, a2)
    other = np.random.default_rng(5)
    env.rng = other
    assert env.rng is other


def test_from_gymnasium_wrapper_attrs():
    gym_env = gym.wrappers.TimeLimit(TinyGymEnv(), max_episode_steps=5)
    env = FromGymnasiumEnv(gym_env)
    assert env.has_wrapper_attr("action_space")        # on the unienv side
    assert env.has_wrapper_attr("_max_episode_steps")  # on the gymnasium wrapper
    assert env.has_wrapper_attr("position")            # inside the gymnasium wrapper stack
    assert not env.has_wrapper_attr("does_not_exist")
    assert env.get_wrapper_attr("_max_episode_steps") == 5
    assert env.get_wrapper_attr("position").shape == (2,)
    env.set_wrapper_attr("_max_episode_steps", 7)
    assert gym_env._max_episode_steps == 7
    env.set_wrapper_attr("render_fps", 30)
    assert env.render_fps == 30
    meta = {"render_modes": []}
    env.metadata = meta
    assert env.metadata is meta


def test_gymnasium_round_trip_through_both_adapters():
    unienv_env = FromGymnasiumEnv(ToGymnasiumEnv(NoContextStubEnv()))
    reference = NoContextStubEnv()
    _, obs, _ = unienv_env.reset(seed=6)
    _, ref_obs, _ = reference.reset(seed=6)
    np.testing.assert_array_equal(obs["position"], ref_obs["position"])
    action = np.full(2, 0.25, np.float32)
    obs, reward, *_ = unienv_env.step(action)
    ref_obs, ref_reward, *_ = reference.step(action)
    np.testing.assert_allclose(obs["position"], ref_obs["position"])
    assert reward == pytest.approx(ref_reward)


def _box(n=1):
    return BoxSpace(B, -1.0, 1.0, np.float32, shape=(n,))


def test_combine_context_obs_space_branches():
    combine = ToGymnasiumEnv.combine_context_obs_space
    both = combine(DictSpace(B, {"goal": _box()}), DictSpace(B, {"pos": _box()}))
    assert set(both.keys()) == {"goal", "pos"}
    ctx_dict = combine(DictSpace(B, {"goal": _box()}), _box(2))
    assert set(ctx_dict.keys()) == {"goal", "observation"}
    obs_dict = combine(_box(), DictSpace(B, {"pos": _box()}))
    assert set(obs_dict.keys()) == {"pos", "context"}
    neither = combine(_box(), _box(2))
    assert set(neither.keys()) == {"context", "observation"}
    obs_only = _box(3)
    assert combine(None, obs_only) is obs_only


def test_combine_context_obs_branches():
    combine = ToGymnasiumEnv.combine_context_obs
    assert combine({"goal": 1}, {"pos": 2}) == {"goal": 1, "pos": 2}
    assert combine({"goal": 1}, 2) == {"goal": 1, "observation": 2}
    assert combine(1, {"pos": 2}) == {"pos": 2, "context": 1}
    assert combine(1, 2) == {"context": 1, "observation": 2}
    assert combine(None, 2) == 2


def test_to_gymnasium_metadata_and_render_mode_setters():
    env = StubEnv()
    gym_env = ToGymnasiumEnv(env)
    gym_env.metadata = {"render_modes": []}
    assert gym_env.metadata == {"render_modes": []}
    gym_env.render_mode = None
    assert env.render_mode is None
    env.render_fps = None
    assert ToGymnasiumEnv(env).metadata is env.metadata
    gym_env.np_random = np.random.default_rng(1)
    assert gym_env.np_random_seed == -1


def test_from_gymnasium_misc():
    gym_env = TinyGymEnv()
    env = FromGymnasiumEnv(gym_env)
    env.render_mode = None
    assert gym_env.render_mode is None
    assert "TinyGymEnv" in str(env)
    gym_env.plain_attr = 1
    assert env.get_wrapper_attr("plain_attr") == 1
    env.set_wrapper_attr("plain_attr", 2)
    assert gym_env.plain_attr == 2
    env.set_wrapper_attr("batch_size", None)
    with pytest.raises(AssertionError):
        env.reset(mask=np.array([True]))
