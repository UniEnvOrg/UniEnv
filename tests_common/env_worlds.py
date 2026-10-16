"""Minimal deterministic worlds and nodes for testing the world/env composers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace
from unienv_interface.world.funcnode import FuncWorldNode
from unienv_interface.world.funcworld import FuncWorld
from unienv_interface.world.node import WorldNode
from unienv_interface.world.world import World


def _shape(batch_size: Optional[int], *inner: int) -> Tuple[int, ...]:
    return tuple(inner) if batch_size is None else (batch_size, *inner)


class SimWorld(World):
    """Simulated world whose clock advances by ``world_timestep`` per step; records lifecycle calls."""

    def __init__(self, world_timestep: Optional[float] = 0.01, batch_size: Optional[int] = None):
        self.backend = B
        self.device = None
        self.world_timestep = world_timestep
        self.batch_size = batch_size
        self.time = 0.0
        self.steps = 0
        self.log: List[str] = []

    def step(self) -> float:
        self.steps += 1
        self.time += self.world_timestep
        self.log.append("step")
        return self.world_timestep

    def reset(self, *, seed=None, mask=None, **kwargs) -> None:
        self.log.append("reset")
        self.time = 0.0

    def reload(self, *, seed=None, mask=None, **kwargs) -> None:
        self.log.append("reload")
        super().reload(seed=seed, mask=mask, **kwargs)

    def after_reset(self, *, seed=None, mask=None, **kwargs) -> None:
        self.log.append("after_reset")

    def after_reload(self, *, seed=None, mask=None, **kwargs) -> None:
        self.log.append("after_reload")

    def close(self) -> None:
        self.log.append("close")


class CounterNode(WorldNode):
    """Node with a 1-D action/observation: obs = accumulated action * world time; logs dispatches."""

    def __init__(self, world: World, name: str = "counter", *, control_timestep: Optional[float] = None,
                 update_timestep: Optional[float] = None, priorities=(0,), with_signals: bool = False):
        self.name = name
        self.world = world
        self.control_timestep = control_timestep
        self.update_timestep = update_timestep
        bs = world.batch_size
        self.action_space = BoxSpace(B, -1.0, 1.0, np.float32, shape=_shape(bs, 1))
        self.observation_space = BoxSpace(B, -np.inf, np.inf, np.float32, shape=_shape(bs, 1))
        self.context_space = None
        self.has_reward = with_signals
        self.has_termination_signal = with_signals
        self.has_truncation_signal = with_signals
        priorities = set(priorities)
        self.reset_priorities = set(priorities)
        self.reload_priorities = set(priorities)
        self.after_reset_priorities = set(priorities)
        self.after_reload_priorities = set(priorities)
        self.pre_environment_step_priorities = set(priorities)
        self.post_environment_step_priorities = set(priorities)
        self.log: List[Tuple[str, Any]] = []
        self.value = np.zeros(_shape(bs, 1), np.float32)
        self.action = np.zeros(_shape(bs, 1), np.float32)

    def reset(self, *, priority=0, seed=None, mask=None, **kwargs):
        self.log.append(("reset", priority))
        if mask is None:
            self.value[...] = 0.0
        else:
            self.value[np.asarray(mask)] = 0.0

    def reload(self, *, priority=0, seed=None, mask=None, **kwargs):
        self.log.append(("reload", priority))
        self.value[...] = 0.0

    def after_reset(self, *, priority=0, mask=None):
        self.log.append(("after_reset", priority))

    def after_reload(self, *, priority=0, mask=None):
        self.log.append(("after_reload", priority))

    def set_next_action(self, action):
        self.log.append(("set_next_action", None))
        self.action = np.asarray(action, np.float32)

    def pre_environment_step(self, dt, *, priority=0):
        self.log.append(("pre_step", (priority, dt)))

    def post_environment_step(self, dt, *, priority=0):
        self.log.append(("post_step", (priority, dt)))
        self.value = self.value + self.action

    def get_observation(self):
        return self.value.copy()

    def get_reward(self):
        return -np.abs(self.value[..., 0]) if self.world.batch_size is not None else float(-abs(self.value[0]))

    def get_termination(self):
        done = self.value[..., 0] > 100
        return done if self.world.batch_size is not None else bool(done)

    def get_truncation(self):
        return np.zeros(self.world.batch_size, bool) if self.world.batch_size is not None else False

    def get_info(self):
        return {"time": self.world.time}


class SimFuncWorld(FuncWorld):
    """Functional world; state is ``{"time": float, "steps": int}``."""

    def __init__(self, world_timestep: Optional[float] = 0.01, batch_size: Optional[int] = None):
        self.backend = B
        self.device = None
        self.world_timestep = world_timestep
        self.batch_size = batch_size

    def initial(self, *, seed=None, **kwargs):
        return {"time": 0.0, "steps": 0}

    def step(self, state):
        return {"time": state["time"] + self.world_timestep, "steps": state["steps"] + 1}, self.world_timestep

    def reset(self, state, *, seed=None, mask=None, **kwargs):
        return {"time": 0.0, "steps": state["steps"]}


class CounterFuncNode(FuncWorldNode):
    """Functional twin of :class:`CounterNode`; node state is ``{"value", "action"}``."""

    def __init__(self, world: FuncWorld, name: str = "counter", *, control_timestep: Optional[float] = None,
                 update_timestep: Optional[float] = None, with_signals: bool = False):
        self.name = name
        self.world = world
        self.control_timestep = control_timestep
        self.update_timestep = update_timestep
        bs = world.batch_size
        self.action_space = BoxSpace(B, -1.0, 1.0, np.float32, shape=_shape(bs, 1))
        self.observation_space = BoxSpace(B, -np.inf, np.inf, np.float32, shape=_shape(bs, 1))
        self.has_reward = with_signals
        self.has_termination_signal = with_signals
        self.has_truncation_signal = with_signals
        self.initial_priorities = {0}
        self.reset_priorities = {0}
        self.reload_priorities = {0}
        self.after_reset_priorities = {0}
        self.after_reload_priorities = {0}
        self.pre_environment_step_priorities = {0}
        self.post_environment_step_priorities = {0}
        self._shape = _shape(bs, 1)

    def initial(self, world_state, *, priority=0, seed=None, **kwargs):
        zeros = np.zeros(self._shape, np.float32)
        return world_state, {"value": zeros, "action": zeros.copy()}

    def reset(self, world_state, node_state, *, priority=0, seed=None, mask=None, **kwargs):
        value = node_state["value"].copy()
        if mask is None:
            value[...] = 0.0
        else:
            value[np.asarray(mask)] = 0.0
        return world_state, {"value": value, "action": node_state["action"]}

    def set_next_action(self, world_state, node_state, action):
        return world_state, {"value": node_state["value"], "action": np.asarray(action, np.float32)}

    def post_environment_step(self, world_state, node_state, dt, *, priority=0):
        return world_state, {"value": node_state["value"] + node_state["action"], "action": node_state["action"]}

    def get_observation(self, world_state, node_state):
        return node_state["value"].copy()

    def get_reward(self, world_state, node_state):
        value = node_state["value"]
        return -np.abs(value[..., 0]) if self.world.batch_size is not None else float(-abs(value[0]))

    def get_termination(self, world_state, node_state):
        done = node_state["value"][..., 0] > 100
        return done if self.world.batch_size is not None else bool(done)

    def get_truncation(self, world_state, node_state):
        return np.zeros(self.world.batch_size, bool) if self.world.batch_size is not None else False

    def get_info(self, world_state, node_state):
        return {"time": world_state["time"]}
