"""Regression tests for observation-only CombinedWorldNode / CombinedFuncWorldNode.

Covers the case where a combined node contains ONLY observation-only child
nodes (no child provides an action space, and typically no child registers a
``pre_environment_step`` priority). Before the shared-counter fix this raised
``AssertionError: self._pre_substeps == self._post_substeps`` because the post
counter advanced while the pre counter stayed at zero.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.world import RealWorld, WorldEnv
from unienv_interface.world.node import WorldNode
from unienv_interface.world.nodes.combined_node import CombinedWorldNode

from unienv_interface.world.funcworld import FuncWorld
from unienv_interface.world.funcnode import FuncWorldNode
from unienv_interface.world.funcnodes.combined_funcnode import CombinedFuncWorldNode
from unienv_interface.world.funcenv_composer import FuncWorldEnv, WorldFuncEnvState


# ---------------------------------------------------------------------------
# Stateful (WorldNode) stubs
# ---------------------------------------------------------------------------

class ObsOnlyNode(WorldNode):
    """Minimal observation-only WorldNode with a post_step hook (no pre_step)."""

    after_reset_priorities = {0}
    post_environment_step_priorities = {0}
    # Intentionally NO pre_environment_step_priorities.

    def __init__(
        self,
        world,
        name: str,
        update_timestep: Optional[float] = 0.01,
        control_timestep: Optional[float] = 0.01,
        obs_key: str = "value",
    ):
        self.name = name
        self.world = world
        self.control_timestep = control_timestep
        self.update_timestep = update_timestep
        self._obs_key = obs_key
        self.observation_space = DictSpace(
            NumpyComputeBackend,
            {obs_key: BoxSpace(NumpyComputeBackend, low=0.0, high=1e9, dtype=np.float32, shape=(1,))},
        )
        self.action_space = None
        self.post_count = 0
        self.pre_count = 0
        self._value = np.zeros(1, dtype=np.float32)

    def after_reset(self, *, priority: int = 0, mask=None) -> None:
        self.post_count = 0
        self.pre_count = 0
        self._value = np.zeros(1, dtype=np.float32)

    def pre_environment_step(self, dt, *, priority: int = 0) -> None:
        self.pre_count += 1

    def post_environment_step(self, dt, *, priority: int = 0) -> None:
        self.post_count += 1
        self._value = self._value + 1.0

    def get_observation(self) -> Dict[str, Any]:
        return {self._obs_key: self._value.copy()}

    def set_next_action(self, action) -> None:
        pass

    def close(self) -> None:
        pass


class ActionNode(ObsOnlyNode):
    """A node that also exposes an action space and a pre_step hook."""

    pre_environment_step_priorities = {0}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = BoxSpace(
            NumpyComputeBackend, low=-1.0, high=1.0, dtype=np.float32, shape=(2,)
        )
        self.last_action = None

    def set_next_action(self, action) -> None:
        self.last_action = np.asarray(action, dtype=np.float32).copy()

    def pre_environment_step(self, dt, *, priority: int = 0) -> None:
        self.pre_count += 1


# ---------------------------------------------------------------------------
# Functional (FuncWorldNode) stubs
# ---------------------------------------------------------------------------

class FuncObsOnlyNode(FuncWorldNode):
    """Minimal observation-only FuncWorldNode with a post_step hook."""

    initial_priorities = {0}
    after_reset_priorities = {0}
    post_environment_step_priorities = {0}
    # Intentionally NO pre_environment_step_priorities.

    def __init__(self, world, name, update_timestep=0.01, control_timestep=0.01, obs_key="value"):
        self.name = name
        self.world = world
        self.control_timestep = control_timestep
        self.update_timestep = update_timestep
        self._obs_key = obs_key
        self.observation_space = DictSpace(
            NumpyComputeBackend,
            {obs_key: BoxSpace(NumpyComputeBackend, low=0.0, high=1e9, dtype=np.float32, shape=(1,))},
        )
        self.action_space = None

    def initial(self, world_state, *, priority=0, seed=None, **kwargs):
        return world_state, {"post_count": 0, "pre_count": 0, "value": np.zeros(1, dtype=np.float32)}

    def reset(self, world_state, node_state, *, priority=0, seed=None, mask=None, **kwargs):
        node_state = dict(node_state)
        node_state["post_count"] = 0
        node_state["pre_count"] = 0
        node_state["value"] = np.zeros(1, dtype=np.float32)
        return world_state, node_state

    def after_reset(self, world_state, node_state, *, priority=0, mask=None):
        return world_state, node_state

    def pre_environment_step(self, world_state, node_state, dt, *, priority=0):
        node_state = dict(node_state)
        node_state["pre_count"] = node_state["pre_count"] + 1
        return world_state, node_state

    def post_environment_step(self, world_state, node_state, dt, *, priority=0):
        node_state = dict(node_state)
        node_state["post_count"] = node_state["post_count"] + 1
        node_state["value"] = node_state["value"] + 1.0
        return world_state, node_state

    def get_observation(self, world_state, node_state):
        return {self._obs_key: node_state["value"].copy()}

    def set_next_action(self, world_state, node_state, action):
        return world_state, node_state

    def close(self, world_state, node_state):
        return world_state


class FuncActionNode(FuncObsOnlyNode):
    """A functional node that also exposes an action space and a pre_step hook."""

    pre_environment_step_priorities = {0}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = BoxSpace(
            NumpyComputeBackend, low=-1.0, high=1.0, dtype=np.float32, shape=(2,)
        )

    def initial(self, world_state, *, priority=0, seed=None, **kwargs):
        ws, ns = super().initial(world_state, priority=priority, seed=seed, **kwargs)
        ns["last_action"] = None
        return ws, ns

    def set_next_action(self, world_state, node_state, action):
        node_state = dict(node_state)
        node_state["last_action"] = np.asarray(action, dtype=np.float32).copy()
        return world_state, node_state


class DummyFuncWorld(FuncWorld):
    """Trivial FuncWorld with a fixed timestep and an integer counter state."""

    def __init__(self, world_timestep=0.01):
        self.backend = NumpyComputeBackend
        self.device = None
        self.world_timestep = world_timestep
        self.world_subtimestep = None
        self.batch_size = None

    def initial(self, *, seed=None, **kwargs):
        return 0

    def step(self, state):
        return state + 1, self.world_timestep

    def reset(self, state, *, seed=None, mask=None, **kwargs):
        return 0

    def close(self, state):
        pass


# ---------------------------------------------------------------------------
# Stateful (WorldEnv) tests
# ---------------------------------------------------------------------------

def test_obs_only_combined_world_env_step_none():
    """Two observation-only nodes composed in a WorldEnv: reset + 5x step(None)."""
    world = RealWorld(NumpyComputeBackend, world_timestep=0.01)
    n1 = ObsOnlyNode(world, "n1", update_timestep=0.01, control_timestep=0.01)
    n2 = ObsOnlyNode(world, "n2", update_timestep=0.01, control_timestep=0.01)
    env = WorldEnv(world, [n1, n2])
    assert env.action_space is None

    ctx, obs, info = env.reset()
    assert "n1" in obs and "n2" in obs

    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(None)

    # Each node's post hook should have fired once per step (ratio 1).
    assert n1.post_count == 5
    assert n2.post_count == 5
    # Observation value should have incremented 5 times.
    assert obs["n1"]["value"][0] == 5.0
    assert obs["n2"]["value"][0] == 5.0


def test_obs_only_combined_world_env_ratio_dispatch():
    """Node A updates every step (0.01), node B every 2nd step (0.02)."""
    world = RealWorld(NumpyComputeBackend, world_timestep=0.01)
    n1 = ObsOnlyNode(world, "n1", update_timestep=0.01, control_timestep=0.02)
    n2 = ObsOnlyNode(world, "n2", update_timestep=0.02, control_timestep=0.02)
    env = WorldEnv(world, [n1, n2])
    env.reset()

    for _ in range(5):
        env.step(None)

    # n1 ratio = 1 -> fires every update substep. control_ts=0.02, update_ts=0.01
    # -> 2 update substeps per control step -> 5 steps * 2 = 10 fires.
    assert n1.post_count == 10
    # n2 ratio = 2 -> fires every 2nd update substep -> 5 fires.
    assert n2.post_count == 5


def test_obs_only_combined_set_next_action_none_is_noop():
    """set_next_action(None) is a no-op when action_space is None; real actions assert."""
    world = RealWorld(NumpyComputeBackend, world_timestep=0.01)
    n1 = ObsOnlyNode(world, "n1")
    n2 = ObsOnlyNode(world, "n2")
    combined = CombinedWorldNode("combined", [n1, n2])
    # step(None) path: no-op, must not raise.
    combined.set_next_action(None)
    # Providing a real action when action_space is None must raise.
    with pytest.raises(AssertionError):
        combined.set_next_action({"n1": np.zeros(2)})


def test_mixed_combined_world_env_step():
    """One obs-only node + one action node: step with a proper action mapping works."""
    world = RealWorld(NumpyComputeBackend, world_timestep=0.01)
    obs_node = ObsOnlyNode(world, "obs", update_timestep=0.01, control_timestep=0.01)
    act_node = ActionNode(world, "act", update_timestep=0.01, control_timestep=0.01)
    env = WorldEnv(world, [obs_node, act_node])
    assert env.action_space is not None

    env.reset()
    obs_node.post_count = 0
    act_node.post_count = 0
    act_node.pre_count = 0

    action = np.array([0.5, -0.5], dtype=np.float32)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(action)

    # Both post hooks fire every step; only the action node has a pre hook.
    assert obs_node.post_count == 3
    assert act_node.post_count == 3
    assert act_node.pre_count == 3
    # The action node received the routed action (direct_return single-node).
    np.testing.assert_allclose(act_node.last_action, np.array([0.5, -0.5], dtype=np.float32))
    # Observation value advanced.
    assert obs["obs"]["value"][0] == 3.0


# ---------------------------------------------------------------------------
# Functional (FuncWorldEnv) tests
# ---------------------------------------------------------------------------

def test_obs_only_combined_func_env_step_none():
    """Two observation-only FuncWorldNodes composed: initial + 5x step(None)."""
    world = DummyFuncWorld(world_timestep=0.01)
    n1 = FuncObsOnlyNode(world, "n1", update_timestep=0.01, control_timestep=0.01)
    n2 = FuncObsOnlyNode(world, "n2", update_timestep=0.01, control_timestep=0.01)
    env = FuncWorldEnv(world, [n1, n2])
    assert env.action_space is None

    state, ctx, obs, info = env.initial()
    assert "n1" in obs and "n2" in obs

    for _ in range(5):
        state, obs, reward, terminated, truncated, info = env.step(state, None)

    ns1 = state.node_state["n1"]
    ns2 = state.node_state["n2"]
    assert ns1["post_count"] == 5
    assert ns2["post_count"] == 5
    assert obs["n1"]["value"][0] == 5.0
    assert obs["n2"]["value"][0] == 5.0


def test_obs_only_combined_func_env_ratio_dispatch():
    """FuncNode A updates every step, B every 2nd step."""
    world = DummyFuncWorld(world_timestep=0.01)
    n1 = FuncObsOnlyNode(world, "n1", update_timestep=0.01, control_timestep=0.02)
    n2 = FuncObsOnlyNode(world, "n2", update_timestep=0.02, control_timestep=0.02)
    env = FuncWorldEnv(world, [n1, n2])
    state, _, _, _ = env.initial()

    for _ in range(5):
        state, _, _, _, _, _ = env.step(state, None)

    assert state.node_state["n1"]["post_count"] == 10
    assert state.node_state["n2"]["post_count"] == 5


def test_obs_only_combined_func_set_next_action_none_is_noop():
    """CombinedFuncWorldNode.set_next_action(None) is a no-op when action_space is None."""
    world = DummyFuncWorld(world_timestep=0.01)
    n1 = FuncObsOnlyNode(world, "n1")
    n2 = FuncObsOnlyNode(world, "n2")
    combined = CombinedFuncWorldNode("combined", [n1, n2])
    env = FuncWorldEnv(world, [combined])
    state, _, _, _ = env.initial()
    # Should not raise and should leave state unchanged.
    ws, ns = combined.set_next_action(state.world_state, state.node_state, None)
    assert ws is state.world_state
    assert ns == state.node_state
    # Providing a real action when action_space is None must raise.
    with pytest.raises(AssertionError):
        combined.set_next_action(state.world_state, state.node_state, {"n1": np.zeros(2)})


def test_mixed_combined_func_env_step():
    """One obs-only func node + one action func node: step with action mapping works."""
    world = DummyFuncWorld(world_timestep=0.01)
    obs_node = FuncObsOnlyNode(world, "obs", update_timestep=0.01, control_timestep=0.01)
    act_node = FuncActionNode(world, "act", update_timestep=0.01, control_timestep=0.01)
    env = FuncWorldEnv(world, [obs_node, act_node])
    assert env.action_space is not None

    state, _, _, _ = env.initial()
    # Reset per-node counters after initial.
    state.node_state["obs"]["post_count"] = 0
    state.node_state["act"]["post_count"] = 0
    state.node_state["act"]["pre_count"] = 0

    action = np.array([0.5, -0.5], dtype=np.float32)
    for _ in range(3):
        state, obs, reward, terminated, truncated, info = env.step(state, action)

    assert state.node_state["obs"]["post_count"] == 3
    assert state.node_state["act"]["post_count"] == 3
    assert state.node_state["act"]["pre_count"] == 3
    np.testing.assert_allclose(
        state.node_state["act"]["last_action"], np.array([0.5, -0.5], dtype=np.float32)
    )
    assert obs["obs"]["value"][0] == 3.0


# ---------------------------------------------------------------------------
# Flat variants
# ---------------------------------------------------------------------------

def test_flat_combined_world_env_obs_only():
    """FlatCombinedWorldNode with two obs-only DictSpace nodes steps with None."""
    from unienv_interface.world.nodes.flat_combined_node import FlatCombinedWorldNode

    world = RealWorld(NumpyComputeBackend, world_timestep=0.01)
    n1 = ObsOnlyNode(world, "n1", update_timestep=0.01, control_timestep=0.01)
    n2 = ObsOnlyNode(world, "n2", update_timestep=0.01, control_timestep=0.01, obs_key="value2")
    combined = FlatCombinedWorldNode("flat", [n1, n2])
    env = WorldEnv(world, combined)
    assert env.action_space is None

    env.reset()
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(None)

    assert n1.post_count == 3
    assert n2.post_count == 3
    # Flat merge: keys from both nodes appear at the top level.
    assert "value" in obs and "value2" in obs


def test_flat_combined_func_env_obs_only():
    """FlatCombinedFuncWorldNode with two obs-only DictSpace nodes steps with None."""
    from unienv_interface.world.funcnodes.flat_combined_funcnode import FlatCombinedFuncWorldNode

    world = DummyFuncWorld(world_timestep=0.01)
    n1 = FuncObsOnlyNode(world, "n1", update_timestep=0.01, control_timestep=0.01)
    n2 = FuncObsOnlyNode(world, "n2", update_timestep=0.01, control_timestep=0.01, obs_key="value2")
    combined = FlatCombinedFuncWorldNode("flat", [n1, n2])
    env = FuncWorldEnv(world, combined)
    assert env.action_space is None

    state, _, obs, _ = env.initial()
    for _ in range(3):
        state, obs, reward, terminated, truncated, info = env.step(state, None)

    assert state.node_state["n1"]["post_count"] == 3
    assert state.node_state["n2"]["post_count"] == 3
    assert "value" in obs


# ---------------------------------------------------------------------------
# Routing-state reset on env.reset() (FIX 3 regression)
# ---------------------------------------------------------------------------

class PostOnlyNoAfterResetNode(ObsOnlyNode):
    """ObsOnlyNode variant that registers NO after_reset priority.

    Used to verify the combined node resets its routing counter even when no
    child participates in the after_reset phase.
    """

    after_reset_priorities = set()


class FuncPostOnlyNoAfterResetNode(FuncObsOnlyNode):
    """FuncObsOnlyNode variant that registers NO after_reset priority."""

    after_reset_priorities = set()


def test_combined_reset_restarts_routing_phase():
    """Routing counter resets on env.reset() even with no after_reset children.

    A post-only ratio-2 node fires when the shared update counter is odd
    (counter % 2 == 1). With ``control_timestep=None`` the env takes exactly
    one update substep per ``env.step()``, so the counter advances by 1 per
    step. After an odd-numbered step the counter is odd; without resetting the
    routing state, the next step would land on an even counter and the node
    would NOT fire. With the fix, ``env.reset()`` resets the counter to 0, so
    the node fires on the first step after reset (and not on the second).
    """
    world = RealWorld(NumpyComputeBackend, world_timestep=0.01)
    n1 = PostOnlyNoAfterResetNode(world, "n1", update_timestep=0.01, control_timestep=None)
    n2 = PostOnlyNoAfterResetNode(world, "n2", update_timestep=0.02, control_timestep=None)
    env = WorldEnv(world, [n1, n2])
    env.reset()

    # Step once: counter 0 -> 1 (odd), n2 (ratio 2) fires.
    env.step(None)
    assert n2.post_count == 1

    # Reset must restart the routing phase (counter -> 0).
    env.reset()

    # First step after reset: counter 0 -> 1 (odd), n2 fires.
    env.step(None)
    assert n2.post_count == 2
    # Second step after reset: counter 1 -> 2 (even), n2 does NOT fire.
    env.step(None)
    assert n2.post_count == 2


def test_combined_func_reset_restarts_routing_phase():
    """Func variant: routing counter resets on reset even with no after_reset children."""
    world = DummyFuncWorld(world_timestep=0.01)
    n1 = FuncPostOnlyNoAfterResetNode(world, "n1", update_timestep=0.01, control_timestep=None)
    n2 = FuncPostOnlyNoAfterResetNode(world, "n2", update_timestep=0.02, control_timestep=None)
    env = FuncWorldEnv(world, [n1, n2])
    state, _, _, _ = env.initial()

    # Step once: counter 0 -> 1 (odd), n2 (ratio 2) fires.
    state, _, _, _, _, _ = env.step(state, None)
    assert state.node_state["n2"]["post_count"] == 1

    # Reset must restart the routing phase (counter -> 0).
    state, _, _, _ = env.reset(state)

    # First step after reset: counter 0 -> 1 (odd), n2 fires.
    state, _, _, _, _, _ = env.step(state, None)
    assert state.node_state["n2"]["post_count"] == 2
    # Second step after reset: counter 1 -> 2 (even), n2 does NOT fire.
    state, _, _, _, _, _ = env.step(state, None)
    assert state.node_state["n2"]["post_count"] == 2