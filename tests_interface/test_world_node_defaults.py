"""Default behaviour of the ``WorldNode`` / ``FuncWorldNode`` / ``World`` / ``FuncWorld`` base classes."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.world.funcnode import FuncWorldNode
from unienv_interface.world.funcworld import FuncWorld
from unienv_interface.world.node import WorldNode
from unienv_interface.world.world import World
from tests_common.env_worlds import SimFuncWorld, SimWorld


class BareNode(WorldNode):
    def __init__(self, world):
        self.name = "bare"
        self.world = world


class BareFuncNode(FuncWorldNode):
    def __init__(self, world):
        self.name = "bare"
        self.world = world

    def initial(self, world_state, *, priority=0, seed=None, **kwargs):
        return world_state, {"init": seed}

    def reset(self, world_state, node_state, *, priority=0, seed=None, mask=None, **kwargs):
        return super().reset(world_state, node_state, priority=priority, seed=seed, mask=mask)


class BareWorld(World):
    def __init__(self):
        self.backend = B
        self.device = None
        self.world_timestep = 0.1
        self.resets = []

    def step(self):
        return super().step()

    def reset(self, *, seed=None, mask=None, **kwargs):
        self.resets.append(seed)


class BareFuncWorld(FuncWorld):
    def __init__(self):
        self.backend = B
        self.device = None
        self.world_timestep = 0.1

    def initial(self, *, seed=None, **kwargs):
        return {"seed": seed}

    def step(self, state):
        return super().step(state)

    def reset(self, state, *, seed=None, mask=None, **kwargs):
        return super().reset(state, seed=seed, mask=mask)


def test_world_node_defaults():
    world = SimWorld()
    node = BareNode(world)
    assert node.backend is B and node.device is None
    assert not node.can_render and node.render() is None
    assert node.effective_update_timestep is None
    node.control_timestep = 0.05
    assert node.effective_update_timestep == 0.05
    node.update_timestep = 0.01
    assert node.effective_update_timestep == 0.01
    assert node.get_context() is None and node.get_info() is None
    assert node.get_reward() == 0 and node.get_termination() is False and node.get_truncation() is False
    with pytest.raises(NotImplementedError):
        node.get_observation()
    with pytest.raises(NotImplementedError):
        node.set_next_action(None)
    node.pre_environment_step(0.1)
    node.post_environment_step(0.1)
    node.reset()
    node.reload(seed=1)
    node.after_reset()
    node.after_reload()
    assert node.get_node("x") is None and node.get_node([]) is node
    assert node.get_nodes_by_type(BareNode) == [node]
    assert node.unwrapped is node and node.prev_wrapper_layer is None
    node.set_wrapper_attr("custom", 1)
    assert node.has_wrapper_attr("custom") and node.get_wrapper_attr("custom") == 1
    node.close()


def test_func_world_node_defaults():
    world = SimFuncWorld()
    node = BareFuncNode(world)
    assert node.backend is B and node.device is None
    assert not node.can_render and node.render({}, {}) is None
    assert node.effective_update_timestep is None
    ws = {"t": 0}
    assert node.reload(ws, seed=3) == (ws, {"init": 3})
    ns = {"init": 3}
    assert node.reset(ws, ns) == (ws, ns)
    assert node.after_reset(ws, ns) == (ws, ns)
    assert node.after_reload(ws, ns) == (ws, ns)
    assert node.pre_environment_step(ws, ns, 0.1) == (ws, ns)
    assert node.post_environment_step(ws, ns, 0.1) == (ws, ns)
    assert node.get_context(ws, ns) is None and node.get_info(ws, ns) is None
    for method in (node.get_observation, node.get_reward, node.get_termination, node.get_truncation):
        with pytest.raises(NotImplementedError):
            method(ws, ns)
    with pytest.raises(NotImplementedError):
        node.set_next_action(ws, ns, None)
    assert node.close(ws, ns) is ws
    assert node.get_node("x") is None and node.get_node(()) is node
    assert node.get_nodes_by_type(BareFuncNode) == [node]
    assert node.get_nodes_by_fn(lambda n: False) == []
    assert node.unwrapped is node and node.prev_wrapper_layer is None
    node.set_wrapper_attr("custom", 1)
    assert node.has_wrapper_attr("custom") and node.get_wrapper_attr("custom") == 1


def test_world_defaults():
    world = BareWorld()
    world.reload(seed=4)
    assert world.resets == [4]
    world.after_reset(); world.after_reload(); world.close()
    with pytest.raises(NotImplementedError):
        world.step()


def test_func_world_defaults():
    world = BareFuncWorld()
    state = world.reload(None, seed=5)
    assert state == {"seed": 5}
    assert world.after_reset(state) is state and world.after_reload(state) is state
    world.close(state)
    with pytest.raises(NotImplementedError):
        world.step(state)
    with pytest.raises(NotImplementedError):
        world.reset(state)
