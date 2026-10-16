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
