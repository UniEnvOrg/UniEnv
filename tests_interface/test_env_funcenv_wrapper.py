"""Tests for ``FuncEnvWrapper`` and the ``FuncEnvBasedEnv`` adapter's attribute plumbing."""

import numpy as np
import pytest

from unienv_interface.env_base.funcenv import FuncEnvBasedEnv
from unienv_interface.env_base.funcenv_wrapper import FuncEnvWrapper
from tests_common.envs import StubFuncEnv


class _Plain(FuncEnvWrapper):
    pass


def test_forwarding(any_stub_func_env):
    inner = any_stub_func_env
    wrapper = _Plain(_Plain(inner))
    assert wrapper.metadata is inner.metadata
    meta = {"render_modes": []}
    wrapper.metadata = meta
    assert wrapper.metadata is meta
    assert wrapper.backend is inner.backend and wrapper.device is None
    assert wrapper.batch_size == inner.batch_size
    assert wrapper.action_space is inner.action_space
    assert wrapper.observation_space is inner.observation_space
    assert wrapper.context_space is inner.context_space
    wrapper.context_space = None
    assert wrapper.context_space is None
    wrapper.action_space = inner.observation_space
    assert wrapper.action_space is inner.observation_space
    wrapper.observation_space = inner.action_space
    assert wrapper.observation_space is inner.action_space
    assert wrapper.unwrapped is inner
    assert wrapper.prev_wrapper_layer.prev_wrapper_layer is inner
    assert inner.unwrapped is inner and inner.prev_wrapper_layer is None
    assert str(wrapper) == repr(wrapper)
    assert str(wrapper).startswith("<_Plain<_Plain<")


def test_lifecycle_is_forwarded():
    inner = StubFuncEnv(batch_size=2)
    wrapper = _Plain(inner)
    state, context, obs, info = wrapper.initial(seed=1)
    state, obs, reward, terminated, truncated, info = wrapper.step(state, np.full((2, 2), 0.5, np.float32))
    np.testing.assert_array_equal(info["step"], [1, 1])
    state, context, obs, info = wrapper.reset(state, seed=2, mask=np.array([True, False]))
    assert obs["position"].shape == (1, 2)
    state, render_state, render_info = wrapper.render_init(state)
    assert render_info.render_fps == 10
    frame, state, render_state = wrapper.render_image(state, render_state)
    assert frame.shape == (2, 4, 6, 3) and render_state == {"frames": 1}
    state = wrapper.render_close(state, render_state)
    wrapper.close(state)
    assert [c[0] for c in inner.calls] == ["initial", "step", "reset", "render_init", "render_close", "close"]


def test_wrapper_attr_helpers():
    inner = StubFuncEnv()
    wrapper = _Plain(_Plain(inner))
    inner.custom = 1
    assert wrapper.has_wrapper_attr("custom") and wrapper.get_wrapper_attr("custom") == 1
    wrapper.set_wrapper_attr("custom", 2)
    assert inner.custom == 2
    assert not wrapper.has_wrapper_attr("missing")
    with pytest.raises(AttributeError, match="missing"):
        wrapper.get_wrapper_attr("missing")
    with pytest.raises(AttributeError, match="missing"):
        wrapper.set_wrapper_attr("missing", 0)
    inner.set_wrapper_attr("other", 3)
    assert inner.get_wrapper_attr("other") == 3 and inner.has_wrapper_attr("other")


def test_func_env_based_env_attributes_and_render():
    func_env = StubFuncEnv(batch_size=2)
    env = FuncEnvBasedEnv(func_env, render_mode="rgb_array", render_kwargs={})
    assert env.render_mode == "rgb_array"
    assert env.metadata is func_env.metadata
    env.metadata = {"render_modes": ["x"]}
    assert env.metadata == {"render_modes": ["x"]}
    assert env.batch_size == 2 and env.backend is func_env.backend and env.device is None
    assert env.context_space is func_env.context_space
    # attribute plumbing reaches the functional env
    func_env.custom = 1
    assert env.has_wrapper_attr("custom") and env.get_wrapper_attr("custom") == 1
    env.set_wrapper_attr("custom", 2)
    assert func_env.custom == 2
    env.set_wrapper_attr("state", None)  # defined on the adapter itself
    assert not env.has_wrapper_attr("missing")
    with pytest.raises(AttributeError):
        env.get_wrapper_attr("missing")

    with pytest.raises(AssertionError):
        env.reset(mask=np.array([True, False]))  # first reset must not be partial
    env.reset(seed=0)
    assert env.render().shape == (2, 4, 6, 3)
    assert env.render_state == {"frames": 1}
    env.close()
    assert ("render_close",) in func_env.calls and env.state is None and env.render_state is None
    # closing resets the adapter: the next reset is an initial one again
    env.reset(seed=1)
    assert func_env.calls[-1] == ("initial", 1)
