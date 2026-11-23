"""Regression tests for world / node composition bugs (timesteps, dispatch, root nodes, kwargs)."""

from __future__ import annotations

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.world import FuncWorldEnv, RealWorld, WorldEnv
from unienv_interface.world.funcnode import FuncWorldNode
from unienv_interface.world.funcnodes.combined_funcnode import CombinedFuncWorldNode
from unienv_interface.world.funcnodes.flat_combined_funcnode import FlatCombinedFuncWorldNode
from unienv_interface.world.node import WorldNode
from unienv_interface.world.nodes.combined_node import CombinedWorldNode
from unienv_interface.world.nodes.env_node import EnvAsWorldNode
from unienv_interface.world.nodes.flat_combined_node import FlatCombinedWorldNode
from tests_common.env_worlds import CounterFuncNode, CounterNode, SimFuncWorld, SimWorld
from tests_interface.test_env_node import FakeEnv


def _box():
    return BoxSpace(B, -1.0, 1.0, np.float32, shape=(1,))


class _LogFuncNode(CounterFuncNode):
    """``CounterFuncNode`` that records post-step dts and action deliveries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []

    def post_environment_step(self, world_state, node_state, dt, *, priority=0):
        self.log.append(("post_step", dt))
        return super().post_environment_step(world_state, node_state, dt, priority=priority)

    def set_next_action(self, world_state, node_state, action):
        self.log.append(("set_next_action", None))
        return super().set_next_action(world_state, node_state, action)


# ---------------------------------------------------------------------------
# Control-rate validation of combined nodes
# ---------------------------------------------------------------------------

def test_combined_node_rejects_control_timestep_not_multiple_of_smallest():
    # 0.06 is a multiple of 0.02 and 0.03, but 0.03 is not a multiple of the smallest
    # (0.02): its action ratio would be round(1.5) == 2, i.e. every 0.04 s.
    world = SimWorld(0.01)
    nodes = [CounterNode(world, n, control_timestep=c, update_timestep=0.01)
             for n, c in (("a", 0.02), ("b", 0.03), ("c", 0.06))]
    with pytest.raises(AssertionError, match="control_timestep"):
        CombinedWorldNode("combined", nodes)


def test_combined_funcnode_rejects_control_timestep_not_multiple_of_smallest():
    world = SimFuncWorld(0.01)
    nodes = [CounterFuncNode(world, n, control_timestep=c, update_timestep=0.01)
             for n, c in (("a", 0.02), ("b", 0.03), ("c", 0.06))]
    with pytest.raises(AssertionError, match="control_timestep"):
        CombinedFuncWorldNode("combined", nodes)


def test_combined_node_multi_rate_actions_follow_control_timesteps():
    world = SimWorld(0.01)
    nodes = [CounterNode(world, n, control_timestep=c, update_timestep=0.01)
             for n, c in (("a", 0.02), ("b", 0.04), ("c", 0.08))]
    env = WorldEnv(world, nodes)
    env.reset()
    for _ in range(8):
        env.step({n.name: [0.1] for n in nodes})
    counts = {n.name: sum(1 for m, _ in n.log if m == "set_next_action") for n in nodes}
    assert counts == {"a": 8, "b": 4, "c": 2}


# ---------------------------------------------------------------------------
# Control timestep must be a multiple of the update timestep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("control_ts,update_ts", [(0.03, 0.02), (0.02, 0.04), (0.05, 0.02)])
def test_world_env_rejects_control_not_multiple_of_update(control_ts, update_ts):
    world = SimWorld(0.01)
    with pytest.raises(AssertionError, match="Control timestep"):
        WorldEnv(world, CounterNode(world, control_timestep=control_ts, update_timestep=update_ts))


@pytest.mark.parametrize("control_ts,update_ts", [(0.03, 0.02), (0.02, 0.04), (0.05, 0.02)])
def test_func_world_env_rejects_control_not_multiple_of_update(control_ts, update_ts):
    world = SimFuncWorld(0.01)
    with pytest.raises(AssertionError, match="Control timestep"):
        FuncWorldEnv(world, CounterFuncNode(world, control_timestep=control_ts, update_timestep=update_ts))


@pytest.mark.parametrize("world_ts,update_ts,control_ts,n_world", [
    (0.01, 0.01, 0.05, 5), (0.01, 0.02, 0.1, 10), (0.1 / 3, 0.1, 0.3, 9),
])
def test_world_env_accepts_float_multiple_control_and_update(world_ts, update_ts, control_ts, n_world):
    world = SimWorld(world_ts)
    env = WorldEnv(world, CounterNode(world, control_timestep=control_ts, update_timestep=update_ts))
    env.reset()
    env.step(np.zeros((1,), np.float32))
    assert world.steps == n_world


# ---------------------------------------------------------------------------
# dt forwarded to children without their own timesteps
# ---------------------------------------------------------------------------

def test_combined_node_forwards_dt_to_untimed_children():
    world = SimWorld(0.01)
    untimed = CounterNode(world, "a")
    timed = CounterNode(world, "b", control_timestep=0.02)
    env = WorldEnv(world, [untimed, timed])
    env.reset()
    untimed.log.clear(); timed.log.clear()
    env.step({"a": [0.1], "b": [0.1]})
    assert [p for m, p in untimed.log if m in ("pre_step", "post_step")] == [(0, 0.02), (0, 0.02)]
    assert [p for m, p in timed.log if m in ("pre_step", "post_step")] == [(0, 0.02), (0, 0.02)]


def test_combined_node_of_untimed_children_gets_world_dt():
    world = SimWorld(0.01)
    a, b = CounterNode(world, "a"), CounterNode(world, "b")
    env = WorldEnv(world, [a, b])
    env.reset()
    a.log.clear()
    env.step({"a": [0.1], "b": [0.1]})
    assert [p for m, p in a.log if m in ("pre_step", "post_step")] == [(0, 0.01), (0, 0.01)]


def test_combined_funcnode_forwards_dt_to_untimed_children():
    world = SimFuncWorld(0.01)
    untimed = _LogFuncNode(world, "a")
    timed = _LogFuncNode(world, "b", control_timestep=0.02)
    env = FuncWorldEnv(world, [untimed, timed])
    state, *_ = env.initial()
    env.step(state, {"a": [0.1], "b": [0.1]})
    assert [dt for m, dt in untimed.log if m == "post_step"] == [0.02]
    assert [dt for m, dt in timed.log if m == "post_step"] == [0.02]


# ---------------------------------------------------------------------------
# Nested combined func nodes across initial / reload priorities
# ---------------------------------------------------------------------------

def _nested_func_env():
    world = SimFuncWorld(0.01)
    a = CounterFuncNode(world, "a")
    a.initial_priorities = {1}
    a.reload_priorities = {1}
    b = CounterFuncNode(world, "b")
    inner = CombinedFuncWorldNode("inner", [a, b])
    outer = CombinedFuncWorldNode("outer", [inner, CounterFuncNode(world, "c")])
    return FuncWorldEnv(world, outer)


def test_nested_combined_funcnode_keeps_state_of_all_initial_priorities():
    env = _nested_func_env()
    state, _, obs, _ = env.initial()
    assert {"a", "b"} <= set(state.node_state["inner"])
    assert set(obs) == {"inner", "c"} and set(obs["inner"]) == {"a", "b"}
    state, obs, *_ = env.step(state, {"inner": {"a": [1.0], "b": [0.5]}, "c": [0.25]})
    np.testing.assert_allclose(obs["inner"]["a"], [1.0])
    np.testing.assert_allclose(obs["inner"]["b"], [0.5])


def test_nested_combined_funcnode_keeps_state_of_all_reload_priorities():
    env = _nested_func_env()
    state, *_ = env.initial()
    state, _, obs, _ = env.reset(state, reload=True)
    assert {"a", "b"} <= set(state.node_state["inner"])
    assert set(obs["inner"]) == {"a", "b"}
