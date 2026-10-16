"""Small deterministic numpy environments for wrapper / env-contract tests.

Dynamics (per instance): ``position += action``; ``velocity = action``.
Reward is ``-sum(|position|)``; an instance terminates once every coordinate of
``position`` exceeds ``terminate_above``. ``reset(seed=s)`` draws the start
position from ``np.random.default_rng(s)`` in ``[-1, 1)`` so seeding is
observable, and the context is the per-instance goal ``(s or 0) * ones``.
Info carries ``{"step": int-array-or-int}``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.env_base.env import Env
from unienv_interface.env_base.funcenv import FuncEnv, FuncEnvCommonRenderInfo
from unienv_interface.space import BoxSpace, DictSpace

DIM = 2


def _shape(batch_size: Optional[int], *inner: int) -> Tuple[int, ...]:
    return tuple(inner) if batch_size is None else (batch_size, *inner)


def make_spaces(batch_size: Optional[int], action_low: float = -1.0, action_high: float = 1.0):
    """Return ``(action_space, observation_space, context_space)`` for the stub envs."""
    action_space = BoxSpace(B, action_low, action_high, np.float32, shape=_shape(batch_size, DIM))
    observation_space = DictSpace(B, {
        "position": BoxSpace(B, -np.inf, np.inf, np.float32, shape=_shape(batch_size, DIM)),
        "velocity": BoxSpace(B, -np.inf, np.inf, np.float32, shape=_shape(batch_size, DIM)),
    })
    context_space = BoxSpace(B, -np.inf, np.inf, np.float32, shape=_shape(batch_size, DIM))
    return action_space, observation_space, context_space


def _initial_position(batch_size: Optional[int], seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=_shape(batch_size, DIM)).astype(np.float32)


class StubEnv(Env):
    """Stateful numpy env; unbatched when ``batch_size is None``."""

    backend = B
    device = None
    render_mode = "rgb_array"
    render_fps = 10
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, batch_size: Optional[int] = None, *, terminate_above: float = 3.0,
                 action_low: float = -1.0, action_high: float = 1.0) -> None:
        self.batch_size = batch_size
        self.terminate_above = terminate_above
        self.action_space, self.observation_space, self.context_space = make_spaces(
            batch_size, action_low, action_high)
        self.rng = B.random.random_number_generator(0)
        self.position = np.zeros(_shape(batch_size, DIM), np.float32)
        self.velocity = np.zeros_like(self.position)
        self.goal = np.zeros_like(self.position)
        self.steps = np.zeros(_shape(batch_size), np.int64)
        self.last_action: Optional[np.ndarray] = None
        self.reset_calls: list = []
        self.closed = 0

    def _obs(self, index: Any = ...) -> Dict[str, np.ndarray]:
        return {"position": self.position[index].copy(), "velocity": self.velocity[index].copy()}

    def reset(self, *, mask: Optional[np.ndarray] = None, seed: Optional[int] = None,
              **kwargs: Any) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        self.reset_calls.append({"mask": None if mask is None else np.array(mask), "seed": seed, **kwargs})
        start = _initial_position(self.batch_size, seed)
        goal = np.full_like(self.position, float(seed or 0))
        index: Any = ... if mask is None else np.asarray(mask, dtype=bool)
        assert mask is None or self.batch_size is not None, "mask is only valid for batched envs"
        self.position[index] = start[index]
        self.velocity[index] = 0.0
        self.goal[index] = goal[index]
        self.steps[index] = 0
        steps = self.steps[index]
        return self.goal[index].copy(), self._obs(index), {"step": steps.copy() if self.batch_size else int(steps)}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == self.position.shape, f"action shape {action.shape} != {self.position.shape}"
        self.last_action = action.copy()
        self.position = self.position + action
        self.velocity = action.copy()
        self.steps = self.steps + 1
        reward = -np.abs(self.position).sum(axis=-1)
        terminated = np.all(self.position > self.terminate_above, axis=-1)
        if self.batch_size is None:
            return (self._obs(), float(reward), bool(terminated), False, {"step": int(self.steps)})
        truncated = np.zeros(self.batch_size, dtype=bool)
        return (self._obs(), reward.astype(np.float32), terminated, truncated, {"step": self.steps.copy()})

    def render(self) -> np.ndarray:
        value = int(np.clip(np.abs(self.position).sum(), 0, 255))
        frame = np.full((4, 6, 3), value, dtype=np.uint8)
        return frame if self.batch_size is None else np.stack([frame] * self.batch_size)

    def close(self) -> None:
        self.closed += 1


class StubFuncEnv(FuncEnv):
    """Functional twin of :class:`StubEnv`; state is ``{"position", "velocity", "goal", "steps"}``."""

    backend = B
    device = None
    render_mode = "rgb_array"
    render_fps = 10
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, batch_size: Optional[int] = None, *, terminate_above: float = 3.0) -> None:
        self.batch_size = batch_size
        self.terminate_above = terminate_above
        self.action_space, self.observation_space, self.context_space = make_spaces(batch_size)
        self.rng = B.random.random_number_generator(0)
        self.calls: list = []

    @staticmethod
    def _obs(state: Dict[str, np.ndarray], index: Any = ...) -> Dict[str, np.ndarray]:
        return {"position": state["position"][index].copy(), "velocity": state["velocity"][index].copy()}

    def initial(self, *, seed: Optional[int] = None, **kwargs: Any):
        self.calls.append(("initial", seed))
        position = _initial_position(self.batch_size, seed)
        state = {
            "position": position,
            "velocity": np.zeros_like(position),
            "goal": np.full_like(position, float(seed or 0)),
            "steps": np.zeros(_shape(self.batch_size), np.int64),
        }
        steps = state["steps"]
        return state, state["goal"].copy(), self._obs(state), {"step": steps.copy() if self.batch_size else int(steps)}

    def reset(self, state: Dict[str, np.ndarray], *, seed: Optional[int] = None,
              mask: Optional[np.ndarray] = None, **kwargs: Any):
        self.calls.append(("reset", seed, None if mask is None else np.array(mask)))
        fresh, _, _, _ = self.initial(seed=seed)
        self.calls.pop()
        index: Any = ... if mask is None else np.asarray(mask, dtype=bool)
        new_state = {k: v.copy() for k, v in state.items()}
        for key in new_state:
            new_state[key][index] = fresh[key][index]
        steps = new_state["steps"][index]
        return (new_state, new_state["goal"][index].copy(), self._obs(new_state, index),
                {"step": steps.copy() if self.batch_size else int(steps)})

    def step(self, state: Dict[str, np.ndarray], action: np.ndarray):
        self.calls.append(("step",))
        action = np.asarray(action, dtype=np.float32)
        position = state["position"] + action
        new_state = {"position": position, "velocity": action.copy(), "goal": state["goal"].copy(),
                     "steps": state["steps"] + 1}
        reward = -np.abs(position).sum(axis=-1)
        terminated = np.all(position > self.terminate_above, axis=-1)
        if self.batch_size is None:
            return new_state, self._obs(new_state), float(reward), bool(terminated), False, {"step": int(new_state["steps"])}
        return (new_state, self._obs(new_state), reward.astype(np.float32), terminated,
                np.zeros(self.batch_size, dtype=bool), {"step": new_state["steps"].copy()})

    def render_init(self, state, *, seed: Optional[int] = None, render_mode: Optional[str] = None, **kwargs):
        self.calls.append(("render_init",))
        return state, {"frames": 0}, FuncEnvCommonRenderInfo(render_mode=render_mode or self.render_mode, render_fps=self.render_fps)

    def render_image(self, state, render_state):
        render_state = {"frames": render_state["frames"] + 1}
        value = int(np.clip(np.abs(state["position"]).sum(), 0, 255))
        frame = np.full((4, 6, 3), value, dtype=np.uint8)
        if self.batch_size is not None:
            frame = np.stack([frame] * self.batch_size)
        return frame, state, render_state

    def render_close(self, state, render_state):
        self.calls.append(("render_close",))
        return state

    def close(self, state) -> None:
        self.calls.append(("close",))
