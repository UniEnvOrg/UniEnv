"""Tests for ``ToGymnasiumEnv`` / ``FromGymnasiumEnv``."""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.wrapper.gym_compat import FromGymnasiumEnv, ToGymnasiumEnv
from tests_common.env_stubs import ArrayStepsStubEnv as StubEnv


class TinyGymEnv(gym.Env):
    """Deterministic gymnasium env: obs = position (2,) float32, action Box(-1, 1, (2,))."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, render_mode: Optional[str] = "rgb_array"):
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.position = np.zeros(2, np.float32)
        self.last_options = None
        self.closed = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.last_options = options
        self.position = self.np_random.uniform(-1, 1, 2).astype(np.float32)
        return self.position.copy(), {"reset": True}

    def step(self, action):
        self.position = (self.position + np.asarray(action, np.float32)).astype(np.float32)
        terminated = bool(np.all(self.position > 3))
        return self.position.copy(), float(-np.abs(self.position).sum()), terminated, False, {"step": True}

    def render(self):
        return np.zeros((4, 6, 3), np.uint8)

    def close(self):
        self.closed = True


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
