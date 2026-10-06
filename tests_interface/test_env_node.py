"""Tests for :class:`EnvAsWorldNode` — wrapping an unbatched ``Env`` as a ``WorldNode``."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.env_base.env import Env
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.world import RealWorld, WorldEnv
from unienv_interface.world.node import WorldNode
from unienv_interface.world.nodes.env_node import EnvAsWorldNode


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class FakeEnv(Env):
    """Minimal unbatched numpy-backed ``Env`` with an internal step counter."""

    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"
    render_fps: Optional[int] = None
    batch_size = None

    def __init__(self, term_after: int = 10**9):
        self.backend = NumpyComputeBackend
        self.device = None
        self.observation_space = DictSpace(
            NumpyComputeBackend,
            {"count": BoxSpace(NumpyComputeBackend, low=0.0, high=1e9, dtype=np.float32, shape=(1,))},
        )
        self.action_space = BoxSpace(
            NumpyComputeBackend, low=-1.0, high=1.0, dtype=np.float32, shape=(2,)
        )
        self.context_space = None
        self.term_after = term_after
        self.count = 0
        self.last_action = None
        self.last_seed = None
        self.reset_calls = 0
        self.closed = False

    def _obs(self) -> Dict[str, Any]:
        return {"count": np.array([self.count], dtype=np.float32)}

    def step(self, action):
        self.count += 1
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        terminated = self.count >= self.term_after
        return self._obs(), 1.0, terminated, False, {"step_info": self.count}

    def reset(self, *, mask=None, seed=None, **kwargs):
        assert mask is None
        self.reset_calls += 1
        self.last_seed = seed
        self.count = 0
        return None, self._obs(), {"reset_info": True}

    def render(self):
        return self._obs()["count"]

    def close(self):
        self.closed = True


class ObsOnlyStubNode(WorldNode):
    """Observation-only node used to exercise ``CombinedWorldNode`` nesting."""

    after_reset_priorities = {0}
    post_environment_step_priorities = {0}

    def __init__(self, world, name: str, update_timestep: float = 0.01, control_timestep: float = 0.01):
        self.name = name
        self.world = world
        self.update_timestep = update_timestep
        self.control_timestep = control_timestep
        self.observation_space = DictSpace(
            NumpyComputeBackend,
            {"stub_value": BoxSpace(NumpyComputeBackend, low=0.0, high=1e9, dtype=np.float32, shape=(1,))},
        )
        self.action_space = None
        self._value = np.zeros(1, dtype=np.float32)

    def after_reset(self, *, priority: int = 0, mask=None) -> None:
        self._value = np.zeros(1, dtype=np.float32)

    def post_environment_step(self, dt, *, priority: int = 0) -> None:
        self._value = self._value + 1.0

    def get_observation(self) -> Dict[str, Any]:
        return {"stub_value": self._value.copy()}

    def close(self) -> None:
        pass


class OtherBackend:
    """Distinct backend type used only for the backend-mismatch check."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_spaces_passthrough():
    inner = FakeEnv()
    node = EnvAsWorldNode(inner, "env")

    assert node.env is inner
    assert node.name == "env"
    assert node.action_space is inner.action_space
    assert node.observation_space is inner.observation_space
    assert node.context_space is inner.context_space
    assert node.has_reward is True
    assert node.has_termination_signal is True
    assert node.has_truncation_signal is True
    assert node.render_mode == inner.render_mode == "rgb_array"
    assert node.supported_render_modes == ("rgb_array",)
    # Without a world the node falls back to the inner env's backend/device.
    assert node.backend is inner.backend
    assert node.device is inner.device

    minimal = EnvAsWorldNode(
        FakeEnv(),
        "min",
        expose_reward=False,
        expose_termination=False,
        expose_truncation=False,
        forward_info=False,
    )
    assert minimal.has_reward is False
    assert minimal.has_termination_signal is False
    assert minimal.has_truncation_signal is False


def test_step_through_worldenv():
    dt = 0.01
    world = RealWorld(NumpyComputeBackend, world_timestep=dt)
    inner = FakeEnv(term_after=100)
    node = EnvAsWorldNode(inner, "env", world=world, control_timestep=dt)
    env = WorldEnv(world, node)

    ctx, obs, info = env.reset()
    assert obs["count"][0] == 0.0
    assert info["reset_info"] is True

    for i in range(3):
        action = np.array([0.1 * i, -0.1], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs["count"][0] == float(i + 1)
        assert reward == 1.0
        assert terminated is False
        assert truncated is False
        assert "control_step_elapsed_time" in info
        assert info["control_step_elapsed_time"] is not None
        assert info["step_info"] == i + 1

    assert inner.count == 3


def test_action_forwarding():
    dt = 0.01
    world = RealWorld(NumpyComputeBackend, world_timestep=dt)
    inner = FakeEnv()
    node = EnvAsWorldNode(inner, "env", world=world, control_timestep=dt)
    env = WorldEnv(world, node)
    env.reset()

    action = np.array([0.25, -0.75], dtype=np.float32)
    env.step(action)
    np.testing.assert_array_equal(inner.last_action, action)

    with pytest.raises(ValueError, match="not contained"):
        env.step(np.array([5.0, 0.0], dtype=np.float32))


def test_combined_nesting():
    dt = 0.01
    world = RealWorld(NumpyComputeBackend, world_timestep=dt)
    inner = FakeEnv(term_after=100)
    env_node = EnvAsWorldNode(inner, "env", world=world, control_timestep=dt, update_timestep=dt)
    obs_stub = ObsOnlyStubNode(world, "stub", update_timestep=dt, control_timestep=dt)
    env = WorldEnv(world, [env_node, obs_stub])

    ctx, obs, info = env.reset()
    assert set(obs.keys()) == {"env", "stub"}
    assert obs["env"]["count"][0] == 0.0

    action = np.array([0.5, -0.5], dtype=np.float32)
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(action)

    assert obs["env"]["count"][0] == 2.0
    assert obs["stub"]["stub_value"][0] == 2.0
    np.testing.assert_array_equal(inner.last_action, action)
    assert info["env"]["step_info"] == 2

    # Single action node: CombinedWorldNode routes it as {"env": action}.
    routed = env.node._split_child_actions(action)
    assert set(routed.keys()) == {"env"}
    np.testing.assert_array_equal(routed["env"], action)


def test_hold_last_action():
    dt = 0.01
    action = np.array([0.1, 0.2], dtype=np.float32)

    # hold_last_action=False: the cached action is consumed by one pre-step.
    inner = FakeEnv(term_after=100)
    node = EnvAsWorldNode(inner, "env")
    node.set_next_action(action)
    node.pre_environment_step(dt)
    assert inner.count == 1
    with pytest.raises(RuntimeError, match="no cached action"):
        node.pre_environment_step(dt)
    with pytest.raises(ValueError, match="requires an action every step"):
        node.set_next_action(None)

    # hold_last_action=True: the action is re-used for subsequent pre-steps.
    inner_hold = FakeEnv(term_after=100)
    node_hold = EnvAsWorldNode(inner_hold, "env", hold_last_action=True)
    node_hold.set_next_action(action)
    node_hold.pre_environment_step(dt)
    node_hold.pre_environment_step(dt)
    assert inner_hold.count == 2
    np.testing.assert_array_equal(inner_hold.last_action, action)

    # set_next_action(None) keeps the previously cached action while holding.
    node_hold.set_next_action(None)
    node_hold.pre_environment_step(dt)
    assert inner_hold.count == 3
    np.testing.assert_array_equal(inner_hold.last_action, action)

    # With no cached action at all, holding still has nothing to step with.
    fresh = EnvAsWorldNode(FakeEnv(), "fresh", hold_last_action=True)
    fresh.set_next_action(None)
    with pytest.raises(RuntimeError, match="no cached action"):
        fresh.pre_environment_step(dt)


def test_termination_forwarding():
    dt = 0.01
    action = np.zeros(2, dtype=np.float32)

    world = RealWorld(NumpyComputeBackend, world_timestep=dt)
    inner = FakeEnv(term_after=2)
    node = EnvAsWorldNode(inner, "env", world=world, control_timestep=dt)
    env = WorldEnv(world, node)
    env.reset()

    _, _, terminated, _, _ = env.step(action)
    assert terminated is False
    _, _, terminated, _, _ = env.step(action)
    assert terminated is True

    world2 = RealWorld(NumpyComputeBackend, world_timestep=dt)
    inner2 = FakeEnv(term_after=1)
    node2 = EnvAsWorldNode(inner2, "env", world=world2, control_timestep=dt, expose_termination=False)
    env2 = WorldEnv(world2, node2)
    env2.reset()

    for _ in range(2):
        _, _, terminated, _, _ = env2.step(action)
        assert terminated is False


def test_batched_env_rejected():
    inner = FakeEnv()
    inner.batch_size = 2
    with pytest.raises(ValueError, match="unbatched"):
        EnvAsWorldNode(inner, "env")


def test_backend_mismatch_rejected():
    world = RealWorld(OtherBackend(), world_timestep=0.01)
    with pytest.raises(ValueError, match="Backend mismatch"):
        EnvAsWorldNode(FakeEnv(), "env", world=world)

    # A world with a matching backend type is accepted.
    ok_world = RealWorld(NumpyComputeBackend, world_timestep=0.01)
    node = EnvAsWorldNode(FakeEnv(), "env", world=ok_world)
    assert node.backend is NumpyComputeBackend


def test_reload_dispatches_reset():
    dt = 0.01
    world = RealWorld(NumpyComputeBackend, world_timestep=dt)
    inner = FakeEnv(term_after=100)
    node = EnvAsWorldNode(inner, "env", world=world, control_timestep=dt)
    env = WorldEnv(world, node)

    env.reset()
    assert inner.reset_calls == 1

    action = np.zeros(2, dtype=np.float32)
    env.step(action)
    env.step(action)
    assert inner.count == 2

    # Reload must dispatch through the node's reload_priorities -> reset.
    env.reset(reload=True)
    assert inner.reset_calls == 2
    assert inner.count == 0

    obs, _, _, _, _ = env.step(action)
    assert obs["count"][0] == 1.0

    # The base WorldNode.reload default also delegates to reset.
    node.reload()
    assert inner.reset_calls == 3
    assert inner.count == 0


def test_reset_seed_forwarding():
    inner = FakeEnv()
    node = EnvAsWorldNode(inner, "env", reset_seed=123)

    node.reset()
    assert inner.last_seed == 123

    node.reset(seed=7)
    assert inner.last_seed == 7

    node.reset()
    assert inner.last_seed == 123

    # Through the composer: a plain reset() falls back to reset_seed.
    dt = 0.01
    world = RealWorld(NumpyComputeBackend, world_timestep=dt)
    inner2 = FakeEnv()
    node2 = EnvAsWorldNode(inner2, "env", world=world, control_timestep=dt, reset_seed=42)
    env = WorldEnv(world, node2)

    env.reset()
    assert inner2.last_seed == 42

    env.reset(seed=5)
    assert inner2.last_seed == 5
    env.reset()
    assert inner2.last_seed == 42
