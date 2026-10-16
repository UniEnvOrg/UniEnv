"""Deterministic envs for remote-execution performance / wire-format tests."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.env_base.env import Env
from unienv_interface.space import BoxSpace, DictSpace


class LiveImageEnv(Env):
    """Returns live (mutated-in-place) ``state`` / ``image`` buffers, like many simulators.

    The image is large enough for the codec's zero-copy attachment path.
    """

    backend, device = B, None
    render_mode = "rgb_array"
    metadata: Dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(self, batch_size: Optional[int] = None, image_shape: Tuple[int, ...] = (32, 32, 3)) -> None:
        self.batch_size = batch_size
        bs = () if batch_size is None else (batch_size,)
        self.action_space = BoxSpace(B, -1.0, 1.0, np.float32, shape=bs + (3,))
        self.observation_space = DictSpace(B, {
            "state": BoxSpace(B, -np.inf, np.inf, np.float32, shape=bs + (3,)),
            "image": BoxSpace(B, 0, 255, np.uint8, shape=bs + tuple(image_shape)),
        })
        self.context_space = BoxSpace(B, -np.inf, np.inf, np.float32, shape=bs + (3,))
        self.state = np.zeros(bs + (3,), np.float32)
        self.image = np.zeros(bs + tuple(image_shape), np.uint8)
        self.steps = 0

    def _obs(self) -> Dict[str, np.ndarray]:
        return {"state": self.state, "image": self.image}

    def reset(self, *, mask=None, seed=None, **kwargs):
        index = ... if mask is None else np.asarray(mask, bool)
        self.state[index] = float(seed or 0)
        self.image[index] = (seed or 0) % 256
        self.steps = 0
        obs = self._obs() if mask is None else {k: v[index] for k, v in self._obs().items()}
        return self.state[index].copy(), obs, {"seed": seed}

    def step(self, action):
        self.steps += 1
        self.state += np.asarray(action, np.float32)
        self.image += np.uint8(self.steps % 7)
        if self.batch_size is None:
            return self._obs(), float(self.state.sum()), False, False, {"steps": self.steps, "raw": b"\x01\x02"}
        rewards = self.state.sum(axis=-1)
        done = np.zeros(self.batch_size, bool)
        return self._obs(), rewards, done, done.copy(), {"steps": np.full(self.batch_size, self.steps)}

    def render(self) -> np.ndarray:
        return np.full((8, 8, 3), self.steps % 256, np.uint8)
