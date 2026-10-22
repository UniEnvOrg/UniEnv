"""Local ``Env`` / ``World`` / ``WorldNode`` / ``DataTransformation`` views of objects hosted by a server.

Every method is one remote call. Spaces, timings and metadata come from the
server's descriptor; spaces are refreshed whenever a reset changed them.
``close()`` on a proxy does nothing: hosted objects belong to the server.

Components mode is a plain local ``WorldEnv`` over remote proxies (host them with
``server.register_recursive("sim", env)``)::

    env = WorldEnv(client.world("sim/world"), client.node("sim/node"))
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

from unienv_interface.env_base.env import Env
from unienv_interface.transformations.transformation import DataTransformation
from unienv_interface.world.node import WorldNode
from unienv_interface.world.world import World

if TYPE_CHECKING:
    from unienv_interface.backends import ComputeBackend
    from .client import RemoteClient

__all__ = ["RemoteEnv", "RemoteWorld", "RemoteWorldNode", "RemoteDataTransformation"]


class _Remote:
    client: RemoteClient
    remote_name: str

    def _bind(self, client: RemoteClient, name: str) -> None:
        self.client, self.remote_name = client, name
        descriptor = client.resources[name]
        self.remote_methods = frozenset(descriptor["methods"])
        for attr, value in descriptor["attributes"].items():
            setattr(self, attr, value)
        for attr, value in descriptor.get("priorities", {}).items():
            setattr(self, attr, set(value))
        self._set_spaces(descriptor["spaces"])

    def _set_spaces(self, spaces: Dict[str, Any]) -> None:
        for attr, space in spaces.items():
            setattr(self, attr, space)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self.client._request(self.remote_name, method, list(args), kwargs)

    def close(self) -> None:
        """No-op: the hosted object stays open (it is closed by its server)."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.remote_name!r}>"


class RemoteEnv(_Remote, Env):
    """An ``Env`` hosted by a :class:`RemoteServer`."""

    def __init__(self, client: RemoteClient, name: str) -> None:
        self.backend, self.device = client.backend, client.device
        self._bind(client, name)
        self.rng = self.backend.random.random_number_generator(device=self.device)

    def reset(self, *, mask: Any = None, seed: Optional[int] = None, **kwargs: Any) -> Any:
        return self._call("reset", mask=mask, seed=seed, **kwargs)

    def reload(self, *, mask: Any = None, seed: Optional[int] = None, **kwargs: Any) -> Any:
        return self._call("reload", mask=mask, seed=seed, **kwargs)

    def step(self, action: Any) -> Any:
        return self._call("step", action)

    def render(self) -> Any:
        return self._call("render")

    def get_node(self, nested_keys: Union[str, Sequence[str]]) -> Optional[RemoteWorldNode]:
        """Node proxy by key path, for envs built on a ``WorldEnv``."""
        node = self.client.resources[self.remote_name].get("node")
        return None if node is None else self.client.node(node).get_node(nested_keys)


class RemoteWorld(_Remote, World):
    """A ``World`` hosted by a :class:`RemoteServer`."""

    world_timestep: Optional[float] = None

    def __init__(self, client: RemoteClient, name: str) -> None:
        self.backend, self.device = client.backend, client.device
        self._bind(client, name)

    def step(self) -> Any:
        return self._call("step")

    def reset(self, *, seed: Optional[int] = None, mask: Any = None, **kwargs: Any) -> None:
        return self._call("reset", seed=seed, mask=mask, **kwargs)

    def reload(self, *, seed: Optional[int] = None, mask: Any = None, **kwargs: Any) -> None:
        return self._call("reload", seed=seed, mask=mask, **kwargs)

    def after_reset(self, *, seed: Optional[int] = None, mask: Any = None, **kwargs: Any) -> None:
        return self._call("after_reset", seed=seed, mask=mask, **kwargs)

    def after_reload(self, *, seed: Optional[int] = None, mask: Any = None, **kwargs: Any) -> None:
        return self._call("after_reload", seed=seed, mask=mask, **kwargs)


class RemoteWorldNode(_Remote, WorldNode):
    """A ``WorldNode`` hosted by a :class:`RemoteServer`."""

    def __init__(self, client: RemoteClient, name: str) -> None:
        self._bind(client, name)
        world = client.resources[name].get("world")
        self.world = None if world is None else client.world(world)

    @property
    def backend(self) -> ComputeBackend:
        return self.client.backend

    @property
    def device(self) -> Any:
        return self.client.device

    @property
    def nodes(self) -> List[RemoteWorldNode]:
        """Proxies of the child nodes (empty for a leaf node)."""
        return [self.client.node(name) for name in self.client.resources[self.remote_name].get("children", {}).values()]

    def reset(self, *, priority: int = 0, seed: Optional[int] = None, mask: Any = None, **kwargs: Any) -> None:
        return self._call("reset", priority=priority, seed=seed, mask=mask, **kwargs)

    def reload(self, *, priority: int = 0, seed: Optional[int] = None, mask: Any = None, **kwargs: Any) -> None:
        return self._call("reload", priority=priority, seed=seed, mask=mask, **kwargs)

    def after_reset(self, *, priority: int = 0, mask: Any = None) -> None:
        return self._call("after_reset", priority=priority, mask=mask)

    def after_reload(self, *, priority: int = 0, mask: Any = None) -> None:
        return self._call("after_reload", priority=priority, mask=mask)

    def pre_environment_step(self, dt: Any, *, priority: int = 0) -> None:
        return self._call("pre_environment_step", dt, priority=priority)

    def post_environment_step(self, dt: Any, *, priority: int = 0) -> None:
        return self._call("post_environment_step", dt, priority=priority)

    def set_next_action(self, action: Any) -> None:
        return self._call("set_next_action", action)

    def get_context(self) -> Any:
        return self._call("get_context")

    def get_observation(self) -> Any:
        return self._call("get_observation")

    def get_reward(self) -> Any:
        return self._call("get_reward")

    def get_termination(self) -> Any:
        return self._call("get_termination")

    def get_truncation(self) -> Any:
        return self._call("get_truncation")

    def get_info(self) -> Any:
        return self._call("get_info")

    def render(self) -> Any:
        return self._call("render")

    def get_node(self, nested_keys: Union[str, Sequence[str]]) -> Optional[RemoteWorldNode]:
        node: RemoteWorldNode = self
        for key in [nested_keys] if isinstance(nested_keys, str) else nested_keys:
            name = self.client.resources[node.remote_name].get("children", {}).get(key)
            if name is None:
                return None
            node = self.client.node(name)
        return node

    def get_nodes_by_fn(self, fn: Callable[[WorldNode], bool]) -> List[WorldNode]:
        result: List[WorldNode] = [self] if fn(self) else []
        for child in self.nodes:
            result.extend(child.get_nodes_by_fn(fn))
        return result


class RemoteDataTransformation(_Remote, DataTransformation):
    """A ``DataTransformation`` hosted by a :class:`RemoteServer`.

    ``transform`` and ``get_target_space_from_source`` run on the server: the
    source space and the data are rebuilt there on the backend / device the
    transformation was registered with, and the result comes back on the
    client's. Use it to run a transformation whose dependencies live in another
    environment (e.g. a grasp synthesizer pinned to an old CUDA toolchain).

    The proxy has no inverse and cannot be serialized: it is bound to its
    connection (serialize the hosted transformation on the server instead).
    """

    has_inverse = False

    def __init__(self, client: RemoteClient, name: str) -> None:
        self._bind(client, name)

    def _set_spaces(self, spaces: Dict[str, Any]) -> None:
        """A transformation has no spaces of its own (they derive from the source space)."""

    def get_target_space_from_source(self, source_space: Any) -> Any:
        return self._call("get_target_space_from_source", source_space)

    def transform(self, source_space: Any, data: Any) -> Any:
        return self._call("transform", source_space, data)

    def serialize(self, source_space: Any = None) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self!r} is bound to a remote connection and cannot be serialized; "
            "serialize the transformation hosted by the server instead"
        )

    @classmethod
    def deserialize_from(cls, json_data: Dict[str, Any], source_space: Any = None) -> RemoteDataTransformation:
        raise NotImplementedError("Remote transformations are created with RemoteClient.transformation()")
