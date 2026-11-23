"""Tests for ``FuncEnv`` / ``FuncEnvBasedEnv`` in ``unienv_interface.env_base.funcenv``."""

import numpy as np
import pytest

from unienv_interface.env_base.funcenv import FuncEnv, FuncEnvBasedEnv, FuncEnvCommonRenderInfo
from tests_common.envs import StubFuncEnv


class _NoRenderFuncEnv(StubFuncEnv):
    """A functional env that does not implement any rendering hooks."""
    render_mode = None
    render_init = FuncEnv.render_init
    render_image = FuncEnv.render_image
    render_close = FuncEnv.render_close


def test_render_fps_without_renderer_does_not_raise():
    env = FuncEnvBasedEnv(_NoRenderFuncEnv())
    assert env.render_fps == 10  # declared on the functional env
    assert env.has_wrapper_attr("render_fps")
    assert env.get_wrapper_attr("render_fps") == 10
    env.reset(seed=0)
    env.step(np.zeros(2, np.float32))
    env.close()


def test_render_fps_uses_render_info_when_renderer_exists():
    class _FpsFromRenderer(StubFuncEnv):
        def render_init(self, state, *, seed=None, render_mode=None, **kwargs):
            return state, {"frames": 0}, FuncEnvCommonRenderInfo(render_mode="rgb_array", render_fps=25)

    env = FuncEnvBasedEnv(_FpsFromRenderer())
    assert env.render_fps == 25


# ---------------------------------------------------------------------------
# Regression: render_fps before the first reset (no state yet)
# ---------------------------------------------------------------------------

class _StatefulRendererFuncEnv(StubFuncEnv):
    """A renderer that needs the env state (like most real renderers)."""

    def render_init(self, state, *, seed=None, render_mode=None, **kwargs):
        self.calls.append(("render_init",))
        _ = state["position"]  # raises TypeError for state=None
        return state, {"frames": 0}, FuncEnvCommonRenderInfo(render_mode="rgb_array", render_fps=25)


def test_render_fps_before_reset_with_stateful_renderer():
    func_env = _StatefulRendererFuncEnv()
    env = FuncEnvBasedEnv(func_env)
    assert env.render_fps == 10  # declared fps; the renderer cannot start without a state
    env.reset(seed=0)
    assert env.render_fps == 25  # now from the renderer
    assert env.render().shape == (4, 6, 3)
    env.close()


def _framestack_env():
    from unienv_interface.func_wrapper.frame_stack import FuncFrameStackWrapper
    return FuncEnvBasedEnv(FuncFrameStackWrapper(StubFuncEnv(), obs_stack_size=2))


def test_render_fps_before_reset_through_func_frame_stack_and_control_frequency_wrapper():
    from unienv_interface.wrapper import ControlFrequencyLimitWrapper
    env = _framestack_env()
    assert env.render_fps == 10
    wrapper = ControlFrequencyLimitWrapper(_framestack_env())
    assert wrapper.control_dt == pytest.approx(0.1)
    env.reset(seed=0)
    assert env.render().shape == (4, 6, 3)
    env.close()


def test_async_vec_env_of_func_frame_stack_envs():
    from unienv_interface.env_base.vec_env import AsyncVecEnv
    env = AsyncVecEnv([_framestack_env, _framestack_env], shared_memory=False)
    try:
        assert env.render_fps == 10
        _, obs, _ = env.reset(seed=0)
        assert obs["position"].shape == (2, 3, 2)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Regression: wrapper attributes below a FuncEnvWrapper stack
# ---------------------------------------------------------------------------

def test_wrapper_attrs_reach_through_func_env_wrappers():
    from unienv_interface.func_wrapper.frame_stack import FuncFrameStackWrapper
    inner = StubFuncEnv()
    stack = FuncFrameStackWrapper(inner, obs_stack_size=1)
    env = FuncEnvBasedEnv(stack)
    assert env.has_wrapper_attr("terminate_above")
    assert env.get_wrapper_attr("terminate_above") == 3.0
    env.set_wrapper_attr("terminate_above", 0.1)
    assert inner.terminate_above == 0.1
    assert "terminate_above" not in vars(stack)
    assert not env.has_wrapper_attr("no_such_attribute")
    with pytest.raises(AttributeError):
        env.get_wrapper_attr("no_such_attribute")
    with pytest.raises(AttributeError):
        env.set_wrapper_attr("no_such_attribute", 1)
    # attributes of the adapter itself are still found / set there
    env.set_wrapper_attr("metadata", {"render_modes": []})
    assert env.metadata == {"render_modes": []} and inner.metadata != env.metadata
