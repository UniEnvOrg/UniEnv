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


# ---------------------------------------------------------------------------
# Flat combined func node: no nested-style root/child-name collision check
# ---------------------------------------------------------------------------

class _DictObsFuncNode(FuncWorldNode):
    def __init__(self, world, name, keys):
        self.world = world
        self.name = name
        self.observation_space = DictSpace(B, {k: _box() for k in keys})
        self.initial_priorities = {0}

    def initial(self, world_state, *, priority=0, seed=None, **kwargs):
        return world_state, {}

    def reset(self, world_state, node_state, **kwargs):
        return world_state, node_state

    def get_observation(self, world_state, node_state):
        return {k: np.zeros(1, np.float32) for k in self.observation_space.spaces}


class _DictObsNode(WorldNode):
    def __init__(self, world, name, keys):
        self.world = world
        self.name = name
        self.observation_space = DictSpace(B, {k: _box() for k in keys})


def test_flat_combined_funcnode_root_key_may_equal_child_name():
    world = SimFuncWorld()
    node = FlatCombinedFuncWorldNode(
        "flat", [_DictObsFuncNode(world, "arm", ["arm_pos"])],
        root_node=_DictObsFuncNode(world, "root", ["arm"]),
    )
    assert set(node.observation_space.spaces) == {"arm", "arm_pos"}
    assert node.action_space is None
    env = FuncWorldEnv(world, node)
    _, _, obs, _ = env.initial()
    assert set(obs) == {"arm", "arm_pos"}


def test_flat_combined_funcnode_still_rejects_overlapping_flat_keys():
    world = SimFuncWorld()
    with pytest.raises(ValueError, match="Overlapping key"):
        FlatCombinedFuncWorldNode(
            "flat", [_DictObsFuncNode(world, "arm", ["pos"])],
            root_node=_DictObsFuncNode(world, "root", ["pos"]),
        )


# ---------------------------------------------------------------------------
# Extra reset kwargs reach the children of combined nodes
# ---------------------------------------------------------------------------

class _KwargsNode(CounterNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs_log = []

    def reset(self, *, priority=0, seed=None, mask=None, **kwargs):
        self.kwargs_log.append(("reset", kwargs))
        super().reset(priority=priority, seed=seed, mask=mask)

    def reload(self, *, priority=0, seed=None, mask=None, **kwargs):
        self.kwargs_log.append(("reload", kwargs))
        super().reload(priority=priority, seed=seed, mask=mask)


def test_combined_node_forwards_extra_reset_kwargs():
    world = SimWorld()
    a, b = _KwargsNode(world, "a"), _KwargsNode(world, "b")
    env = WorldEnv(world, [a, b])
    env.reset(options={"x": 1})
    env.reset(options={"x": 2}, pernode_kwargs={"b": {"options": {"x": 3}}})
    assert a.kwargs_log == [("reload", {"options": {"x": 1}}), ("reset", {"options": {"x": 2}})]
    assert b.kwargs_log == [("reload", {"options": {"x": 1}}), ("reset", {"options": {"x": 3}})]


class _KwargsFuncNode(CounterFuncNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs_log = []

    def initial(self, world_state, *, priority=0, seed=None, **kwargs):
        self.kwargs_log.append(("initial", kwargs))
        return super().initial(world_state, priority=priority, seed=seed)

    def reload(self, world_state, *, priority=0, seed=None, **kwargs):
        self.kwargs_log.append(("reload", kwargs))
        return super().initial(world_state, priority=priority, seed=seed)

    def reset(self, world_state, node_state, *, priority=0, seed=None, mask=None, **kwargs):
        self.kwargs_log.append(("reset", kwargs))
        return super().reset(world_state, node_state, priority=priority, seed=seed, mask=mask)


def test_combined_funcnode_forwards_extra_reset_kwargs():
    world = SimFuncWorld()
    a, b = _KwargsFuncNode(world, "a"), _KwargsFuncNode(world, "b")
    env = FuncWorldEnv(world, [a, b])
    state, *_ = env.initial(options=1)
    state, *_ = env.reset(state, options=2)
    state, *_ = env.reset(state, reload=True, options=3)
    assert a.kwargs_log == [("initial", {"options": 1}), ("reset", {"options": 2}), ("reload", {"options": 3})]
    assert b.kwargs_log == a.kwargs_log


# ---------------------------------------------------------------------------
# EnvAsWorldNode without a world inside a combined node
# ---------------------------------------------------------------------------

def test_combined_env_nodes_without_world_can_step():
    world = RealWorld(B, world_timestep=0.01)
    n1 = EnvAsWorldNode(FakeEnv(term_after=2), "e1", control_timestep=0.01)
    n2 = EnvAsWorldNode(FakeEnv(), "e2", control_timestep=0.01)
    env = WorldEnv(world, [n1, n2])
    env.reset()
    action = np.zeros(2, np.float32)
    _, reward, terminated, truncated, _ = env.step({"e1": action, "e2": action})
    assert reward == 2.0 and terminated is False and truncated is False
    _, _, terminated, _, _ = env.step({"e1": action, "e2": action})
    assert terminated is True


# ---------------------------------------------------------------------------
# root_node display name must not collide with a child name
# ---------------------------------------------------------------------------

def test_combined_node_rejects_root_named_like_child():
    world = SimWorld()
    with pytest.raises(ValueError, match="root_node"):
        CombinedWorldNode("c", [CounterNode(world, "arm")], root_node=_DictObsNode(world, "arm", ["root_obs"]))
    with pytest.raises(ValueError, match="root_node"):
        FlatCombinedWorldNode("c", [_DictObsNode(world, "arm", ["x"])], root_node=_DictObsNode(world, "arm", ["y"]))


def test_combined_funcnode_rejects_root_named_like_child():
    world = SimFuncWorld()
    with pytest.raises(ValueError, match="root_node"):
        CombinedFuncWorldNode("c", [CounterFuncNode(world, "arm")], root_node=_DictObsFuncNode(world, "arm", ["root_obs"]))
