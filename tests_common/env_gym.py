"""A tiny deterministic gymnasium environment for adapter tests."""

from typing import Optional

import gymnasium as gym
import numpy as np


class TinyGymEnv(gym.Env):
    """Deterministic gymnasium env: obs = position (2,) float32, action Box(-1, 1, (2,))."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, render_mode: Optional[str] = "rgb_array"):
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.position = np.zeros(2, np.float32)
        self.last_options = None
        self.closed = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.last_options = options
        self.position = self.np_random.uniform(-1, 1, 2).astype(np.float32)
        return self.position.copy(), {"reset": True}

    def step(self, action):
        self.position = (self.position + np.asarray(action, np.float32)).astype(np.float32)
        terminated = bool(np.all(self.position > 3))
        return self.position.copy(), float(-np.abs(self.position).sum()), terminated, False, {"step": True}

    def render(self):
        return np.zeros((4, 6, 3), np.uint8)

    def close(self):
        self.closed = True
