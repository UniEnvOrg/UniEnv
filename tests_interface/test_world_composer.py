"""Tests for ``WorldEnv`` / ``FuncWorldEnv`` (world + node composition)."""

from __future__ import annotations

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.env_base.funcenv import FuncEnvBasedEnv
from unienv_interface.space import BoxSpace
from unienv_interface.world import FuncWorldEnv, RealWorld, WorldEnv
from unienv_interface.world.node import WorldNode
from unienv_interface.world.funcnode import FuncWorldNode
from tests_common.env_worlds import CounterFuncNode, CounterNode, SimFuncWorld, SimWorld


def _action(env, value=1.0):
    return np.full(env.action_space.shape, value, np.float32)


# ---------------------------------------------------------------------------
# WorldEnv
# ---------------------------------------------------------------------------

def test_first_reset_goes_through_reload_then_reset():
    world = SimWorld(0.01)
    node = CounterNode(world, control_timestep=0.01)
    env = WorldEnv(world, node)
    env.reset(seed=0)
    assert world.log == ["reload", "reset", "after_reload"]
    assert [m for m, _ in node.log] == ["reload", "after_reload"]
    world.log.clear(); node.log.clear()
    env.reset(seed=1)
    assert world.log == ["reset", "after_reset"]
    assert [m for m, _ in node.log] == ["reset", "after_reset"]
    world.log.clear(); node.log.clear()
    env.reset(reload=True)
    assert world.log == ["reload", "reset", "after_reload"]


def test_priorities_run_in_descending_order():
    world = SimWorld(0.01)
    node = CounterNode(world, control_timestep=0.01, priorities=(-50, 100, 0))
    env = WorldEnv(world, node)
    env.reset()
    env.reset()
    env.step(_action(env))
    for method in ("reload", "after_reload", "reset", "after_reset"):
        assert [p for m, p in node.log if m == method] == [100, 0, -50]
    assert [p[0] for m, p in node.log if m == "pre_step"] == [100, 0, -50]
    assert [p[0] for m, p in node.log if m == "post_step"] == [100, 0, -50]


@pytest.mark.parametrize("world_ts,update_ts,control_ts,n_update,n_world", [
    (0.01, None, None, 1, 1),
    (0.01, None, 0.05, 1, 5),
    (0.01, 0.02, 0.06, 3, 2),
    (0.002, 0.01, 0.01, 1, 5),
])
def test_sub_step_counts(world_ts, update_ts, control_ts, n_update, n_world):
    world = SimWorld(world_ts)
    node = CounterNode(world, control_timestep=control_ts, update_timestep=update_ts)
    env = WorldEnv(world, node)
    env.reset()
    _, _, _, _, info = env.step(_action(env))
    assert world.steps == n_update * n_world
    assert [m for m, _ in node.log].count("post_step") == n_update
    expected_update_dt = update_ts or control_ts or world_ts
    assert all(p[1] == pytest.approx(expected_update_dt) for m, p in node.log if m in ("pre_step", "post_step"))
    assert info["control_step_elapsed_time"] == pytest.approx(n_update * n_world * world_ts)
    assert env.control_timestep == pytest.approx(control_ts or update_ts or world_ts)
    assert env.render_fps == round(1 / env.control_timestep)
    # node value integrates the action once per update sub-step
    np.testing.assert_allclose(env.step(_action(env))[0], 2 * n_update)


def test_unbatched_default_signals_without_node_signals():
    world = SimWorld(0.01)
    env = WorldEnv(world, CounterNode(world))
    env.reset()
    _, reward, terminated, truncated, info = env.step(_action(env))
    assert reward == 0.0 and terminated is False and truncated is False
    assert info["time"] == pytest.approx(0.01)


def test_batched_default_signals_without_node_signals():
    world = SimWorld(0.01, batch_size=3)
    env = WorldEnv(world, CounterNode(world))
    env.reset()
    _, reward, terminated, truncated, _ = env.step(_action(env))
    assert reward.shape == terminated.shape == truncated.shape == (3,)
    assert terminated.dtype == bool and not terminated.any() and not truncated.any()
    np.testing.assert_array_equal(reward, 0.0)


def test_node_signals_are_used():
    world = SimWorld(0.01, batch_size=2)
    env = WorldEnv(world, CounterNode(world, with_signals=True))
    env.reset()
    _, reward, terminated, _, _ = env.step(_action(env, 0.5))
    np.testing.assert_allclose(reward, [-0.5, -0.5])
    np.testing.assert_array_equal(terminated, [False, False])


def test_batched_masked_reset_returns_selected_rows():
    world = SimWorld(0.01, batch_size=3)
    node = CounterNode(world)
    env = WorldEnv(world, node)
    env.reset()
    obs, *_ = env.step(_action(env))
    mask = np.array([True, False, True])
    _, masked_obs, _ = env.reset(mask=mask)
    assert masked_obs.shape == (2, 1)
    np.testing.assert_array_equal(masked_obs, 0.0)
    merged = env.update_observation_post_reset(obs, masked_obs, mask)
    np.testing.assert_array_equal(merged[:, 0], [0.0, 1.0, 0.0])


def test_world_env_properties_and_close():
    world = SimWorld(0.01)
    node = CounterNode(world, control_timestep=0.02)
    env = WorldEnv(world, node)
    assert env.backend is B and env.device is None and env.batch_size is None
    assert env.observation_space is node.observation_space
    assert env.action_space is node.action_space
    assert env.context_space is None
    assert env.render_mode is None
    assert env.render() is None
    assert env.get_node([]) is node
    assert env.get_node("anything") is None
    assert env.get_nodes_by_type(CounterNode) == [node]
    assert env.get_nodes_by_fn(lambda n: n.name == "counter") == [node]
    env.close()
    assert world.log[-1] == "close"


def test_world_env_with_multiple_nodes_combines_them():
    world = SimWorld(0.01)
    a, b = CounterNode(world, "a"), CounterNode(world, "b")
    env = WorldEnv(world, [a, b])
    assert set(env.observation_space.keys()) == {"a", "b"}
    _, obs, _ = env.reset()
    obs, *_ = env.step({"a": np.ones(1, np.float32), "b": -np.ones(1, np.float32)})
    np.testing.assert_allclose(obs["a"], [1.0])
    np.testing.assert_allclose(obs["b"], [-1.0])
    assert env.get_node("a") is a


def test_real_world_reports_elapsed_wall_time(monkeypatch):
    import importlib
    world_module = importlib.import_module("unienv_interface.world.world")
    now = [10.0]
    monkeypatch.setattr(world_module.time, "monotonic", lambda: now[0])
    world = RealWorld(B, world_timestep=None)
    with pytest.raises(AssertionError):
        world.step()
    world.reset()
    now[0] += 0.25
    assert world.step() == pytest.approx(0.25)
    now[0] += 0.5
    assert world.step() == pytest.approx(0.5)
    world.after_reset(); world.after_reload(); world.close()

    env = WorldEnv(RealWorld(B), CounterNode(world))
    assert env.control_timestep is None and env.render_fps is None


# ---------------------------------------------------------------------------
# FuncWorldEnv
# ---------------------------------------------------------------------------

def test_func_world_env_lifecycle():
    world = SimFuncWorld(0.01)
    func_env = FuncWorldEnv(world, CounterFuncNode(world, control_timestep=0.03, update_timestep=0.01))
    assert func_env.control_timestep == pytest.approx(0.03)
    assert func_env.render_fps == 33
    state, context, obs, info = func_env.initial(seed=0)
    assert context is None and info == {"time": 0.0}
    np.testing.assert_array_equal(obs, [0.0])
    state, obs, reward, terminated, truncated, info = func_env.step(state, np.ones(1, np.float32))
    np.testing.assert_allclose(obs, [3.0])  # three update sub-steps
    assert state.world_state["steps"] == 3
    assert info["control_step_elapsed_time"] == pytest.approx(0.03)
    assert reward == 0.0 and terminated is False and truncated is False
    state, _, obs, _ = func_env.reset(state)
    np.testing.assert_array_equal(obs, [0.0])
    assert state.world_state["steps"] == 3
    state, _, obs, _ = func_env.reset(state, reload=True)
    assert state.world_state["steps"] == 0
    image, state, render_state = func_env.render_image(*func_env.render_init(state)[:2])
    assert image is None
    assert func_env.render_close(state, render_state) is state
    func_env.close(state)


def test_func_world_env_batched_signals_and_masked_reset():
    world = SimFuncWorld(0.01, batch_size=3)
    func_env = FuncWorldEnv(world, CounterFuncNode(world))
    state, _, obs, _ = func_env.initial()
    action = np.ones((3, 1), np.float32)
    state, obs, reward, terminated, truncated, _ = func_env.step(state, action)
    assert reward.shape == terminated.shape == truncated.shape == (3,)
    np.testing.assert_array_equal(reward, 0.0)
    mask = np.array([False, True, False])
    state, _, masked_obs, _ = func_env.reset(state, mask=mask)
    assert masked_obs.shape == (1, 1)
    merged = func_env.update_observation_post_reset(obs, masked_obs, mask)
    np.testing.assert_array_equal(merged[:, 0], [1.0, 0.0, 1.0])


def test_func_world_env_with_signals_and_queries():
    world = SimFuncWorld(0.01, batch_size=2)
    node = CounterFuncNode(world, with_signals=True)
    func_env = FuncWorldEnv(world, node)
    assert func_env.backend is B and func_env.device is None and func_env.batch_size == 2
    assert func_env.get_node([]) is node
    assert func_env.get_nodes_by_type(CounterFuncNode) == [node]
    assert func_env.get_nodes_by_fn(lambda n: False) == []
    state, *_ = func_env.initial()
    _, _, reward, terminated, truncated, _ = func_env.step(state, np.full((2, 1), 0.5, np.float32))
    np.testing.assert_allclose(reward, [-0.5, -0.5])
    np.testing.assert_array_equal(terminated, [False, False])


def test_func_world_env_close_before_initial_is_noop():
    world = SimFuncWorld(0.01)
    env = FuncEnvBasedEnv(FuncWorldEnv(world, CounterFuncNode(world)))
    env.close()  # never reset: no state to release
