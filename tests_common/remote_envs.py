"""Hosted objects and connection helpers for the remote-execution tests and benchmarks."""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.env_base.env import Env
from unienv_interface.remote import RemoteClient, RemoteServer
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.transformations.transformation import DataTransformation
from unienv_interface.world import WorldEnv

from tests_common.env_worlds import CounterNode, SimWorld

TRANSPORTS = ("memory", "tcp", "unix", "websocket")


def connect(server: RemoteServer, transport: str, **options: Any) -> RemoteClient:
    """A client of ``server`` over ``transport`` (one of :data:`TRANSPORTS`)."""
    if transport == "memory":
        return server.connect(**options)
    if transport == "tcp":
        address = server.listen("tcp://127.0.0.1:0")
    elif transport == "unix":
        address = server.listen("unix://" + os.path.join(tempfile.mkdtemp(prefix="uenv"), "s.sock"))
    else:
        address = server.listen("ws://127.0.0.1:0")
    return RemoteClient.connect(address, **options)


class LiveImageEnv(Env):
    """Returns live (mutated-in-place) ``state`` / ``image`` buffers, like many simulators."""

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
        self.context_space = None
        self.state = np.zeros(bs + (3,), np.float32)
        self.image = np.zeros(bs + tuple(image_shape), np.uint8)
        self.steps = 0

    def _obs(self) -> Dict[str, np.ndarray]:
        return {"state": self.state, "image": self.image}

    def reset(self, *, mask=None, seed=None, **kwargs):
        self.state[...] = float(seed or 0)
        self.image[...] = (seed or 0) % 256
        self.steps = 0
        return None, self._obs(), {"seed": seed}

    def step(self, action):
        self.steps += 1
        self.state += np.asarray(action, np.float32)
        self.image += np.uint8(1)
        return self._obs(), float(self.state.sum()), False, False, {"steps": self.steps, "empty": np.zeros((0, 2))}

    def render(self) -> np.ndarray:
        return np.full((8, 8, 3), self.steps % 256, np.uint8)


class ProbeEnv(Env):
    """Counts steps; records worker threads; can block, fail, resize its spaces or return odd values.

    ``reset(size=n)`` rebuilds the observation space with ``n`` dimensions.
    ``step("fail")`` raises, ``step("object")`` returns an unsendable value,
    ``step("block")`` waits for :attr:`release`.
    """

    backend, device = B, None
    batch_size = None
    metadata: Dict[str, Any] = {"render_modes": [], "not_portable": object()}

    def __init__(self, size: int = 2) -> None:
        self.action_space = BoxSpace(B, -1.0, 1.0, np.float32, shape=(1,))
        self.context_space = None
        self.observation_space = self._space(size)
        self.count = 0
        self.threads: List[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed_on: Optional[str] = None

    @staticmethod
    def _space(size: int) -> BoxSpace:
        return BoxSpace(B, -np.inf, np.inf, np.float32, shape=(size,))

    def reset(self, *, mask=None, seed=None, size: Optional[int] = None, **kwargs):
        self.threads.append(threading.current_thread().name)
        if size is not None and size != self.observation_space.shape[0]:
            self.observation_space = self._space(size)
        return None, np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        self.threads.append(threading.current_thread().name)
        if isinstance(action, str):
            if action == "fail":
                raise ValueError("step failed on purpose")
            if action == "object":
                return object()
            if action == "block":
                self.entered.set()
                self.release.wait(10)
        self.count += 1
        return np.full(self.observation_space.shape, self.count, np.float32), float(self.count), False, False, {}

    def render(self):
        return None

    def close(self) -> None:
        self.closed_on = threading.current_thread().name


def make_world_env(names: Tuple[str, ...] = ("counter",), batch_size: Optional[int] = None) -> WorldEnv:
    """A ``WorldEnv`` over :class:`SimWorld` with one :class:`CounterNode` per name."""
    world = SimWorld(world_timestep=0.01, batch_size=batch_size)
    nodes = [CounterNode(world, name, control_timestep=0.02, with_signals=True) for name in names]
    return WorldEnv(world, nodes[0] if len(nodes) == 1 else nodes)


class ScaleTransformation(DataTransformation):
    """Scales a ``BoxSpace`` (and its data) by ``factor``; records what it received."""

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor
        self.received: List[Any] = []

    def get_target_space_from_source(self, source_space):
        if not isinstance(source_space, BoxSpace):
            raise ValueError("ScaleTransformation needs a BoxSpace")
        return BoxSpace(source_space.backend, source_space.low * self.factor, source_space.high * self.factor,
                        dtype=source_space.dtype, device=source_space.device, shape=source_space.shape)

    def transform(self, source_space, data):
        self.received.append((source_space, data))
        return data * self.factor

    def serialize(self, source_space=None):
        return {"factor": self.factor}

    @classmethod
    def deserialize_from(cls, json_data, source_space=None):
        return cls(json_data["factor"])
