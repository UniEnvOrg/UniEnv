"""A small goal-reaching environment and policy for the MCP tests (importable as CLI targets)."""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.env_base.env import Env
from unienv_interface.policy import Policy
from unienv_interface.space import BoxSpace, DictSpace, TextSpace

GOAL = np.array([2.0, 0.0], np.float32)


def _action_space():
    return BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,), metadata={"kind": "action", "names": ["dx", "dy"],
                                                                     "units": ["m", "m"]})


class GridEnv(Env):
    """A point on a plane moved by ``(dx, dy)``; the episode succeeds within 0.25 of the goal ``(2, 0)``.

    Observations: the named position and a 16x16 camera image whose brightness tracks x; the
    context is the instruction. ``step_delay`` slows steps down (seconds).
    """

    backend, device, batch_size = B, None, None
    render_mode = "rgb_array"
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, step_delay: float = 0.0) -> None:
        self.step_delay = step_delay
        self.action_space = _action_space()
        self.observation_space = DictSpace(B, {
            "pos": BoxSpace(B, -10.0, 10.0, np.float32, shape=(2,), metadata={"names": ["x", "y"], "units": ["m", "m"]}),
            "camera": BoxSpace(B, 0, 255, np.uint8, shape=(16, 16, 3), metadata={"kind": "image", "format": "rgb",
                                                                               "layout": "HWC"}),
        })
        self.context_space = TextSpace(B, 64)
        self.rng = B.random.random_number_generator(0)
        self.pos = np.zeros(2, np.float32)
        self.closed_on: Optional[str] = None
        self.threads = set()

    def _obs(self) -> Dict[str, np.ndarray]:
        camera = np.zeros((16, 16, 3), np.uint8)
        camera[..., 0] = int(np.clip((self.pos[0] + 5) * 25, 0, 255))
        return {"pos": self.pos.copy(), "camera": camera}

    def reset(self, *, mask=None, seed: Optional[int] = None, **kwargs: Any):
        self.threads.add(threading.current_thread().name)
        self.pos = np.array([-1.0 - (seed or 0), 0.0], np.float32)
        return "reach the goal at (2, 0)", self._obs(), {"seed": seed}

    def step(self, action):
        self.threads.add(threading.current_thread().name)
        if self.step_delay:
            time.sleep(self.step_delay)
        self.pos = self.pos + np.asarray(action, np.float32)
        distance = float(np.linalg.norm(self.pos - GOAL))
        success = distance < 0.25
        return self._obs(), -distance, success, False, {"success": success, "distance": distance}

    def render(self):
        return self._obs()["camera"]

    def close(self) -> None:
        self.closed_on = threading.current_thread().name


def make_grid() -> GridEnv:
    return GridEnv()


class GoalPolicy(Policy):
    """Moves toward a goal: ``"go to (x, y)"`` in the task (context), else ``(2, 0)``; at most 0.5 per step."""

    backend, device, batch_size = B, None, None

    def __init__(self) -> None:
        self.action_space = _action_space()
        self.observation_space = GridEnv().observation_space
        self.context_space = TextSpace(B, 64)
        self.goal = GOAL.copy()
        self.pos = np.zeros(2, np.float32)
        self.dts = []

    def reset(self, context, observation, *, seed=None, mask=None, **kwargs) -> None:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", str(context or ""))
        self.goal = np.array([float(n) for n in numbers[:2]], np.float32) if len(numbers) >= 2 and "go to" in str(
            context) else GOAL.copy()
        self.pos = np.asarray(observation["pos"], np.float32)

    def observe_transition(self, action, new_observation, dt) -> None:
        self.pos = np.asarray(new_observation["pos"], np.float32)
        self.dts.append(dt)

    def get_action(self, **kwargs):
        return np.clip(self.goal - self.pos, -0.5, 0.5).astype(np.float32)


def make_goal_policy() -> GoalPolicy:
    return GoalPolicy()
