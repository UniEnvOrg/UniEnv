"""UniEnv-compatible views of resources that remain on their server."""

from __future__ import annotations

from numbers import Real
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING, TypeVar, Union

import numpy as np

from unienv_interface.env_base.env import Env
from unienv_interface.world.world import World
from unienv_interface.world.node import WorldNode
from unienv_interface.world.env_composer import WorldEnv
from unienv_interface.space.space_utils.serialization_utils import json_to_space, space_to_json

from .protocol import PRIORITIES

if TYPE_CHECKING:
    from unienv_interface.backends import ArrayAPIArray, ComputeBackend
    from .client import RemoteClient
    from .protocol import Descriptor, OperationKind, ResetResult, Signal, StepResult

_Result = TypeVar("_Result")


def _descriptors_equal(left: Any, right: Any) -> Union[bool, np.bool_]:
    """Compare serialized spaces, including independently decoded NaN values."""
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(_descriptors_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_descriptors_equal(a, b) for a, b in zip(left, right)))
    # Compare actual scalar values without NumPy's mixed-precision coercion.
    if isinstance(left, (np.generic, np.ndarray)):
        if left.ndim != 0:
            return False
        left = left.item()
    if isinstance(right, (np.generic, np.ndarray)):
        if right.ndim != 0:
            return False
        right = right.item()
    if type(left) is not type(right) and not (type(left) in (int, float) and type(right) in (int, float)):
        return False
    return (left == right or (isinstance(left, Real) and isinstance(right, Real)
                             and left != left and right != right))


class _Proxy:
    def __init__(self, client: RemoteClient, resource_id: str) -> None:
        self.client, self.resource_id = client, resource_id
        self._closed = False
        descriptor = client.descriptors.get(resource_id) or client.describe(resource_id)
        self._refresh(descriptor)
        client._proxies[resource_id] = self

    def _refresh(self, descriptor: Descriptor) -> None:
        self.descriptor = descriptor
        for name in ("name", "batch_size", "world_timestep", "world_subtimestep", "control_timestep",
                     "update_timestep", "render_mode", "render_fps", "supported_render_modes",
                     "has_reward", "has_termination_signal", "has_truncation_signal", "metadata"):
            if name in descriptor:
                setattr(self, name, descriptor[name])
        for name in ("observation_space", "action_space", "context_space"):
            if name in descriptor:
                value = descriptor[name]
                current = getattr(self, name, None)
                if current is None or value is None or not _descriptors_equal(space_to_json(current), value):
                    setattr(self, name, None if value is None else json_to_space(value, self.client.backend, self.client.device))
        for name in PRIORITIES:
            if name in descriptor:
                setattr(self, name, set(descriptor[name]))

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise ConnectionError("Proxy is closed")
        return self.client.call(self.resource_id, method, *args, **kwargs)

    def _read(self, field: str, method: str) -> Any:
        if self._closed:
            raise ConnectionError("Proxy is closed")
        return self.client._read_snapshot(self.resource_id, field, method)

    def close(self) -> None:
        """Close this proxy and release shared control, retaining subscriptions."""
        if not self._closed:
            self._closed = True
            self.client.release(self.resource_id)

    def __del__(self) -> None:
        # WorldNode's destructor calls close; GC must never release another
        # live proxy's shared control ownership. Use explicit close/context exit.
        pass


class RemoteEnv(_Proxy, Env):
    def __init__(self, client: RemoteClient, resource_id: str) -> None:
        self.backend, self.device = client.backend, client.device
        super().__init__(client, resource_id)
        self.rng = self.backend.random.random_number_generator(device=self.device)

    def reset(self, *, mask: Optional[ArrayAPIArray] = None, seed: Optional[int] = None,
              **kwargs: Any) -> ResetResult:
        return self._call("reset", mask=mask, seed=seed, **kwargs)

    def reload(self, *, mask: Optional[ArrayAPIArray] = None, seed: Optional[int] = None,
               **kwargs: Any) -> ResetResult:
        return self._call("reload", mask=mask, seed=seed, **kwargs)

    def step(self, action: Any) -> StepResult:
        return self._call("step", action)

    def render(self) -> Any:
        return self._call("render")

    def get_node(self, nested_keys: Union[str, Sequence[str]]) -> Optional[RemoteWorldNode]:
        node_id = self.descriptor["node_id"]
        return None if node_id is None else self.client.node(node_id).get_node(nested_keys)


class RemoteWorld(_Proxy, World):
    def __init__(self, client: RemoteClient, resource_id: str) -> None:
        self.backend, self.device = client.backend, client.device
        super().__init__(client, resource_id)

    def step(self) -> Union[float, ArrayAPIArray]:
        return self._call("step")

    def reset(self, *, seed: Optional[int] = None, mask: Optional[ArrayAPIArray] = None,
              **kwargs: Any) -> None:
        return self._call("reset", seed=seed, mask=mask, **kwargs)

    def reload(self, *, seed: Optional[int] = None, mask: Optional[ArrayAPIArray] = None,
               **kwargs: Any) -> None:
        return self._call("reload", seed=seed, mask=mask, **kwargs)

    def after_reset(self, *, seed: Optional[int] = None, mask: Optional[ArrayAPIArray] = None,
                    **kwargs: Any) -> None:
        return self._call("after_reset", seed=seed, mask=mask, **kwargs)

    def after_reload(self, *, seed: Optional[int] = None, mask: Optional[ArrayAPIArray] = None,
                     **kwargs: Any) -> None:
        return self._call("after_reload", seed=seed, mask=mask, **kwargs)


class RemoteWorldNode(_Proxy, WorldNode):
    def __init__(self, client: RemoteClient, resource_id: str) -> None:
        super().__init__(client, resource_id)
        world_id = self.descriptor["world_id"]
        self.world = None if world_id is None else client.world(world_id)

    @property
    def backend(self) -> ComputeBackend:
        return self.client.backend

    @property
    def device(self) -> Optional[object]:
        return self.client.device

    def pre_environment_step(self, dt: Union[float, ArrayAPIArray], *, priority: int = 0) -> None:
        return self._call("pre_environment_step", dt, priority=priority)

    def post_environment_step(self, dt: Union[float, ArrayAPIArray], *, priority: int = 0) -> None:
        return self._call("post_environment_step", dt, priority=priority)

    def set_next_action(self, action: Any) -> None:
        return self._call("set_next_action", action)

    def get_context(self) -> Any:
        return self._read("context", "get_context")

    def get_observation(self) -> Any:
        return self._read("observation", "get_observation")

    def get_reward(self) -> Union[float, ArrayAPIArray]:
        return self._read("reward", "get_reward")

    def get_termination(self) -> Signal:
        return self._read("terminated", "get_termination")

    def get_truncation(self) -> Signal:
        return self._read("truncated", "get_truncation")

    def get_info(self) -> Optional[Dict[str, Any]]:
        return self._read("info", "get_info")

    def render(self) -> Any:
        return self._call("render")

    def reset(self, *, priority: int = 0, seed: Optional[int] = None,
              mask: Optional[ArrayAPIArray] = None, **kwargs: Any) -> None:
        return self._call("reset", priority=priority, seed=seed, mask=mask, **kwargs)

    def reload(self, *, priority: int = 0, seed: Optional[int] = None,
               mask: Optional[ArrayAPIArray] = None, **kwargs: Any) -> None:
        return self._call("reload", priority=priority, seed=seed, mask=mask, **kwargs)

    def after_reset(self, *, priority: int = 0, mask: Optional[ArrayAPIArray] = None) -> None:
        return self._call("after_reset", priority=priority, mask=mask)

    def after_reload(self, *, priority: int = 0, mask: Optional[ArrayAPIArray] = None) -> None:
        return self._call("after_reload", priority=priority, mask=mask)

    def get_node(self, nested_keys: Union[str, Sequence[str]]) -> Optional[RemoteWorldNode]:
        keys = [nested_keys] if isinstance(nested_keys, str) else list(nested_keys)
        current: RemoteWorldNode = self
        for key in keys:
            resource_id = current.descriptor["children"].get(key)
            if resource_id is None:
                return None
            current = self.client.node(resource_id)
        return current

    def get_nodes_by_fn(self, fn: Callable[[WorldNode], bool]) -> List[WorldNode]:
        result: List[WorldNode] = [self] if fn(self) else []
        for resource_id in self.descriptor["children"].values():
            result.extend(self.client.node(resource_id).get_nodes_by_fn(fn))
        return result


class RemoteWorldEnv(WorldEnv):
    """Local lifecycle composition with explicit remote operation boundaries.

    Pass a RemoteWorld and RemoteWorldNode(s) from the same client and world.
    Underlying nodes always run on the world server. Client-local combinations
    are supported; combining different simulator worlds isn't a WorldEnv model.
    """

    def __init__(self, world: RemoteWorld, node_or_nodes: Union[RemoteWorldNode, Iterable[RemoteWorldNode]],
                 *, render_mode: Optional[str] = "auto") -> None:
        if not isinstance(world, RemoteWorld):
            raise TypeError("RemoteWorldEnv requires a RemoteWorld; detached nodes must be used standalone")
        nodes = [node_or_nodes] if isinstance(node_or_nodes, WorldNode) else list(node_or_nodes)
        if any(isinstance(node, RemoteWorldNode) and node.world is None for node in nodes):
            raise ValueError("Detached nodes cannot be composed with RemoteWorldEnv")
        if not nodes or any(not isinstance(node, RemoteWorldNode) or node.client is not world.client
                            or node.world is not world for node in nodes):
            raise ValueError("Nodes must be remote proxies belonging to this client and world")
        self.client = world.client
        self._boundary_depth = 0
        self._closed = False
        super().__init__(world, nodes[0] if isinstance(node_or_nodes, WorldNode) else nodes, render_mode=render_mode)

    def _run(self, kind: OperationKind, method: Callable[..., _Result], *args: Any, **kwargs: Any) -> _Result:
        if self._closed:
            raise ConnectionError("Composer is closed")
        # WorldEnv.reset invokes self.reload on first reset. That is one
        # operation, not a nested reset + reload pair.
        if self._boundary_depth:
            return method(*args, **kwargs)
        with self.client.operation(self.world.resource_id, kind):
            self._boundary_depth += 1
            try:
                return method(*args, **kwargs)
            finally:
                self._boundary_depth -= 1

    def reset(self, *, mask: Optional[ArrayAPIArray] = None, seed: Optional[int] = None,
              reload: bool = False, **kwargs: Any) -> ResetResult:
        return self._run("reload" if self._first_reset or reload else "reset", super().reset,
                         mask=mask, seed=seed, reload=reload, **kwargs)

    def reload(self, *, mask: Optional[ArrayAPIArray] = None, seed: Optional[int] = None,
               **kwargs: Any) -> ResetResult:
        return self._run("reload", super().reload, mask=mask, seed=seed, **kwargs)

    def step(self, action: Any) -> StepResult:
        return self._run("step", super().step, action)

    def close(self) -> None:
        """Close this composer and release shared control, retaining subscriptions."""
        if not self._closed:
            self._closed = True
            self.client.release(self.world.resource_id)
