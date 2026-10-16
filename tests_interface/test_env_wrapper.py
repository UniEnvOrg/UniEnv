"""Tests for the stateful ``Wrapper`` base classes in ``unienv_interface.env_base.wrapper``."""

import numpy as np
import pytest

from unienv_interface.env_base.wrapper import ActionWrapper, ContextObservationWrapper, Wrapper
from tests_common.envs import StubEnv


class _Plain(Wrapper):
    pass


def test_rng_falls_through_to_inner_env_until_overridden(stub_env):
    wrapper = _Plain(_Plain(stub_env))
    assert wrapper.rng is stub_env.rng
    own = np.random.default_rng(123)
    wrapper.rng = own
    assert wrapper.rng is own
    assert wrapper.env.rng is stub_env.rng


def test_rng_with_jax_key_does_not_call_bool():
    jax = pytest.importorskip("jax")
    env = StubEnv()
    env.rng = jax.random.key(0)
    wrapper = _Plain(env)
    assert wrapper.rng is env.rng
    wrapper.rng = jax.random.key(1)
    assert not bool(jax.numpy.all(jax.random.key_data(wrapper.rng) == jax.random.key_data(env.rng)))


def test_attribute_forwarding_and_overrides(stub_env):
    wrapper = _Plain(stub_env)
    assert wrapper.metadata is stub_env.metadata
    meta = {"render_modes": ["human"]}
    wrapper.metadata = meta
    assert wrapper.metadata is meta and stub_env.metadata is not meta
    assert wrapper.render_mode == "rgb_array" and wrapper.render_fps == 10
    assert wrapper.backend is stub_env.backend and wrapper.device is None
    assert wrapper.batch_size is None
    assert wrapper.action_space is stub_env.action_space
    assert wrapper.observation_space is stub_env.observation_space
    assert wrapper.context_space is stub_env.context_space
    wrapper.context_space = None
    assert wrapper.context_space is None
    assert str(wrapper) == repr(wrapper) == "<_Plain<StubEnv instance>>"
    assert wrapper.render().shape == (4, 6, 3)
    wrapper.close()
    assert stub_env.closed == 1


def test_wrapper_attr_helpers_through_layers(stub_env):
    outer = _Plain(_Plain(stub_env))
    stub_env.custom = 1
    assert outer.has_wrapper_attr("custom")
    assert outer.get_wrapper_attr("custom") == 1
    outer.set_wrapper_attr("custom", 5)
    assert stub_env.custom == 5
    assert not outer.has_wrapper_attr("missing")
    with pytest.raises(AttributeError, match="missing"):
        outer.get_wrapper_attr("missing")
    with pytest.raises(AttributeError, match="missing"):
        outer.set_wrapper_attr("missing", 1)
    # attributes defined on an outer layer win over the inner env
    outer.env.custom = "middle"
    assert outer.get_wrapper_attr("custom") == "middle"
    outer.set_wrapper_attr("custom", "set")
    assert outer.env.custom == "set" and stub_env.custom == 5


def test_base_env_helpers(any_stub_env):
    env = any_stub_env
    assert env.observation_space.contains(env.sample_observation())
    assert env.context_space.contains(env.sample_context())
    assert env.unwrapped is env and env.prev_wrapper_layer is None
    assert str(env) == "<StubEnv instance>"
    env.set_wrapper_attr("custom", 3)
    assert env.get_wrapper_attr("custom") == 3 and env.has_wrapper_attr("custom")
    if env.batch_size is not None:
        context, obs, _ = env.reset(seed=1)
        mask = np.array([False, True, False])
        new_context, new_obs, _ = env.reset(mask=mask, seed=2)
        merged = env.update_context_post_reset(context, new_context, mask)
        np.testing.assert_array_equal(merged[:, 0], [1.0, 2.0, 1.0])
        env.context_space = None
        assert env.update_context_post_reset(context, new_context, mask) is None
        assert env.sample_context() is None
    else:
        with pytest.raises(AssertionError):
            env.update_observation_post_reset({}, {}, np.array([True]))


class _Negate(ActionWrapper):
    def map_action(self, action):
        return -action


class _Identity(ContextObservationWrapper):
    pass


def test_action_wrapper_maps_actions(stub_env):
    wrapper = _Negate(stub_env)
    wrapper.reset(seed=0)
    wrapper.step(np.full(2, 0.5, np.float32))
    np.testing.assert_array_equal(stub_env.last_action, [-0.5, -0.5])
    with pytest.raises(NotImplementedError):
        wrapper.reverse_map_action(np.zeros(2))


def test_context_observation_wrapper_defaults(stub_env):
    wrapper = _Identity(stub_env)
    context, obs, info = wrapper.reset(seed=4)
    np.testing.assert_array_equal(context, [4.0, 4.0])
    obs, reward, terminated, truncated, info = wrapper.step(np.zeros(2, np.float32))
    assert info == {"step": 1}
    with pytest.raises(NotImplementedError):
        wrapper.reverse_map_context(context)
    with pytest.raises(NotImplementedError):
        wrapper.reverse_map_observation(obs)
