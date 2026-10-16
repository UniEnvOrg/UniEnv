"""Picklable (module-level) environments for ``AsyncVecEnv`` tests and benchmarks.

:class:`ArrayEnv` is an unbatched env whose observations are deterministic
functions of ``(seed, step)`` so the parent can check exactly what every worker
wrote. It can run on the numpy, pytorch or jax backend and on any device, and can
be told to misbehave (wrong shape / dtype, raise, die) to exercise error paths.

Observation (``obs_kind="dict"``)::

    {"vec": Box(float32, (4,)), "flag": Binary((2,)),
     "nested": Tuple(Box(int32, (3,)), Box(float32, ())),
     ["image": Box(uint8, image_shape)]}

``obs_kind="box"`` uses just the ``vec`` leaf. The context is ``Box(float32, (2,))``
equal to ``[seed, seed]``; the action is ``Box(float32, (2,))``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from unienv_interface.env_base.env import Env
from unienv_interface.space import BinarySpace, BoxSpace, DictSpace, TupleSpace


def _resolve_backend(backend: str, device: Optional[str]):
    if backend == "numpy":
        from unienv_interface.backends.numpy import NumpyComputeBackend
        return NumpyComputeBackend, None
    if backend == "pytorch":
        import torch
        from unienv_interface.backends.pytorch import PyTorchComputeBackend
        return PyTorchComputeBackend, torch.device(device or "cpu")
    if backend == "jax":
        import jax
        from unienv_interface.backends.jax import JaxComputeBackend
        return JaxComputeBackend, jax.devices(device or "cpu")[0]
    raise ValueError(backend)


def expected_obs(seed: int, t: int, *, obs_kind: str = "dict",
                 image_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
    """Numpy version of the observation :class:`ArrayEnv` emits after ``t`` steps."""
    vec = (np.arange(4, dtype=np.float32) + seed * 10 + t).astype(np.float32)
    if obs_kind == "box":
        return vec
    obs = {
        "vec": vec,
        "flag": np.array([t % 2 == 0, t % 3 == 0], dtype=bool),
        "nested": (np.arange(3, dtype=np.int32) * (t + 1) + seed, np.float32(seed + 0.5 * t)),
    }
    if image_shape is not None:
        obs["image"] = np.full(image_shape, (seed + t) % 256, dtype=np.uint8)
    return obs


class ArrayEnv(Env):
    """Deterministic unbatched env with a configurable backend/device (see module doc)."""

    render_mode = None
    render_fps = None
    metadata = {"render_modes": []}

    def __init__(self, backend: str = "numpy", device: Optional[str] = None, *,
                 obs_kind: str = "dict", image_shape: Optional[Tuple[int, ...]] = None,
                 misbehave: Optional[str] = None, misbehave_after: int = 0,
                 forbidden_pid: Optional[int] = None) -> None:
        if forbidden_pid is not None and os.getpid() == forbidden_pid:
            raise AssertionError("ArrayEnv must not be constructed in this process")
        self.backend, self.device = _resolve_backend(backend, device)
        self.batch_size = None
        self.obs_kind = obs_kind
        self.image_shape = None if image_shape is None else tuple(image_shape)
        self.misbehave = misbehave
        self.misbehave_after = misbehave_after
        B, dev = self.backend, self.device
        f32, i32, u8 = np.float32, np.int32, np.uint8
        if backend == "pytorch":
            import torch
            f32, i32, u8 = torch.float32, torch.int32, torch.uint8
        elif backend == "jax":
            import jax.numpy as jnp
            f32, i32, u8 = jnp.float32, jnp.int32, jnp.uint8
        vec_space = BoxSpace(B, -np.inf, np.inf, f32, shape=(4,), device=dev)
        if obs_kind == "box":
            self.observation_space = vec_space
        else:
            spaces = {
                "vec": vec_space,
                "flag": BinarySpace(B, shape=(2,), device=dev),
                "nested": TupleSpace(B, [
                    BoxSpace(B, -1000, 1000, i32, shape=(3,), device=dev),
                    BoxSpace(B, -np.inf, np.inf, f32, shape=(), device=dev),
                ], device=dev),
            }
            if self.image_shape is not None:
                spaces["image"] = BoxSpace(B, 0, 255, u8, shape=self.image_shape, device=dev)
            self.observation_space = DictSpace(B, spaces, device=dev)
        self.action_space = BoxSpace(B, -1.0, 1.0, f32, shape=(2,), device=dev)
        self.context_space = BoxSpace(B, -np.inf, np.inf, f32, shape=(2,), device=dev)
        self.rng = B.random.random_number_generator(0, device=dev)
        self.seed = 0
        self.t = 0
        self.closed = False

    def _to_backend(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._to_backend(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._to_backend(v) for v in value)
        value = np.asarray(value)
        if self.backend.simplified_name == "jax":
            import jax
            return jax.device_put(value, self.device)
        return self.backend.from_numpy(value, dtype=None, device=self.device)

    def _obs(self) -> Any:
        obs = self._to_backend(expected_obs(self.seed, self.t, obs_kind=self.obs_kind,
                                            image_shape=self.image_shape))
        if self.misbehave is not None and self.t > self.misbehave_after:
            target = obs if self.obs_kind == "box" else obs["vec"]
            if self.misbehave == "shape":
                target = target[:2]
            elif self.misbehave == "dtype":
                target = self._to_backend(np.zeros(4, np.int64)) if self.backend.simplified_name != "jax" \
                    else target.astype("int32")
            elif self.misbehave == "raise":
                raise ValueError("ArrayEnv misbehaved on purpose")
            elif self.misbehave == "exit":
                os._exit(3)
            if self.obs_kind == "box":
                obs = target
            else:
                obs["vec"] = target
        return obs

    def reset(self, *, mask=None, seed: Optional[int] = None, **kwargs):
        assert mask is None
        self.seed = 0 if seed is None else int(seed)
        self.t = 0
        context = self._to_backend(np.full(2, self.seed, np.float32))
        return context, self._obs(), {"t": self.t, "seed": self.seed}

    def step(self, action):
        self.t += 1
        obs = self._obs()
        return obs, float(self.t), self.t >= 1000, False, {"t": self.t}

    def render(self):
        return None

    def close(self) -> None:
        self.closed = True
