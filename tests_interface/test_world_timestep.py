"""Tests for control/world timestep compatibility in ``World`` / ``FuncWorld`` and the composers."""

import pytest

from unienv_interface.world import FuncWorldEnv, WorldEnv
from tests_common.env_worlds import CounterFuncNode, CounterNode, SimFuncWorld, SimWorld

COMPATIBLE = [(0.02, 0.1), (0.1, 0.3), (0.01, 0.07), (0.002, 0.03), (0.1, 0.1), (0.5, 2.0), (1 / 3, 1.0)]
INCOMPATIBLE = [(0.02, 0.05), (0.1, 0.25), (0.02, 0.01), (0.3, 0.2), (0.1, 0.1000001 * 3 + 0.01)]


@pytest.mark.parametrize("world_cls", [SimWorld, SimFuncWorld])
@pytest.mark.parametrize("world_ts,control_ts", COMPATIBLE)
def test_integer_multiples_are_compatible(world_cls, world_ts, control_ts):
    assert world_cls(world_ts).is_control_timestep_compatible(control_ts)


@pytest.mark.parametrize("world_cls", [SimWorld, SimFuncWorld])
@pytest.mark.parametrize("world_ts,control_ts", INCOMPATIBLE)
def test_non_multiples_are_incompatible(world_cls, world_ts, control_ts):
    assert not world_cls(world_ts).is_control_timestep_compatible(control_ts)


@pytest.mark.parametrize("world_cls", [SimWorld, SimFuncWorld])
def test_unknown_timesteps_are_compatible(world_cls):
    assert world_cls(None).is_control_timestep_compatible(0.1)
    assert world_cls(0.1).is_control_timestep_compatible(None)


def test_world_env_accepts_float_multiple_timesteps():
    world = SimWorld(0.1)
    env = WorldEnv(world, CounterNode(world, control_timestep=0.3))
    assert env.control_timestep == pytest.approx(0.3)
    env.reset()
    env.step(env.action_space.create_empty() * 0)
    assert world.steps == 3


def test_func_world_env_accepts_float_multiple_timesteps():
    world = SimFuncWorld(0.02)
    func_env = FuncWorldEnv(world, CounterFuncNode(world, control_timestep=0.1))
    state, *_ = func_env.initial()
    state, *_ = func_env.step(state, func_env.action_space.create_empty() * 0)
    assert state.world_state["steps"] == 5


def test_world_env_rejects_incompatible_timestep():
    world = SimWorld(0.02)
    with pytest.raises(AssertionError):
        WorldEnv(world, CounterNode(world, control_timestep=0.05))
