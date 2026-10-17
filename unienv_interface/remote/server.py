"""Host environments, worlds and nodes for remote clients."""

from __future__ import annotations

import logging
import traceback
from queue import SimpleQueue
from threading import Condition, Event, Lock, Thread
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
from urllib.parse import quote

from unienv_interface.env_base.env import Env
from unienv_interface.space.space_utils.serialization_utils import space_to_json
from unienv_interface.world.env_composer import WorldEnv
from unienv_interface.world.node import WorldNode
from unienv_interface.world.world import World

from . import wire
from .protocol import (ATTRIBUTES, ERROR, EVENT, METHODS, OK, OPTIONAL_METHODS, PRIORITIES, PROTOCOL_VERSION,
                       PUBLISHED_METHODS, RESET_METHODS, SPACES)
from .transport import Listener, MemoryTransport, Transport, listen

if TYPE_CHECKING:
    from .client import RemoteClient

__all__ = ["RemoteServer"]

Hosted = Union[Env, World, WorldNode]
_log = logging.getLogger(__name__)


class _Group:
    """One worker thread per registered root object: all its calls run there, serially."""

    def __init__(self, label: str) -> None:
        self.queue: SimpleQueue = SimpleQueue()
        self.resources: List[_Resource] = []
        self.owned: List[Hosted] = []  # registered objects closed with the server
        self.version = 0  # bumped whenever a space of one of the resources changes
        self.thread = Thread(target=self._run, name=f"unienv-remote:{label}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            item[0](*item[1])


class _Resource:
    def __init__(self, name: str, obj: Hosted, kind: str, group: _Group) -> None:
        self.name, self.obj, self.kind, self.group = name, obj, kind, group
        self.methods = frozenset(m for m in METHODS[kind]
                                 if m not in OPTIONAL_METHODS or callable(getattr(obj, m, None)))
        try:
            self.backend, self.device = obj.backend, obj.device
        except AttributeError:  # a node without a world
            self.backend, self.device = None, None
        self.links: Dict[str, Any] = {}
        self.spaces: Tuple[Any, ...] = ()
        self.documents: Tuple[Any, ...] = ()
        self.version = 0
        self.sequence = 0
        self.subscribers: Tuple[_Connection, ...] = ()
        self.static: Dict[str, Any] = {}
        if kind != "world":
            self.spaces = tuple(getattr(obj, attr, None) for attr in SPACES)
            self.documents = tuple(_document(space) for space in self.spaces)

    def descriptor(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"kind": self.kind, "methods": sorted(self.methods), **self.links}
        attributes = {}
        for attr in ATTRIBUTES:
            try:
                value = getattr(self.obj, attr)
            except Exception:
                continue
            if attr == "metadata" and isinstance(value, dict):
                value = {key: item for key, item in value.items() if _portable(item)}
            if _portable(value):
                attributes[attr] = value
        result["attributes"] = attributes
        if self.kind == "node":
            result["priorities"] = {attr: sorted(getattr(self.obj, attr, ())) for attr in PRIORITIES}
        return result

    def space_update(self) -> List[Any]:
        return [self.version, {attr: None if doc is None else wire.SpaceJSON(doc)
                               for attr, doc in zip(SPACES, self.documents)}]


def _portable(value: Any) -> bool:
    try:
        wire.encode(value)
    except (TypeError, ValueError):
        return False
    return True


def _document(space: Any) -> Optional[Dict[str, Any]]:
    if space is None:
        return None
    try:
        return space_to_json(space)
    except NotImplementedError as exc:
        raise TypeError(f"Cannot serve a space of type {type(space).__name__}: {exc}") from None


def _error(request_id: Any, exc: BaseException) -> Tuple[List[Any], int]:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return wire.encode([request_id, ERROR, [type(exc).__name__, str(exc), text]])


class _Connection:
    """Per-client state. Events go out through a lazily started writer (latest wins)."""

    def __init__(self, server: RemoteServer, transport: Transport) -> None:
        self.server, self.transport = server, transport
        self.compress: Dict[str, Any] = {}
        self.compress_key: Any = None
        self.seen: Dict[_Group, int] = {}
        self.subscriptions: Dict[str, _Resource] = {}
        self.closed = False
        self._events: Dict[str, Tuple[List[Any], int]] = {}
        self._condition = Condition()
        self._writer: Optional[Thread] = None

    def send(self, frame: Tuple[List[Any], int]) -> None:
        try:
            self.transport.send(*frame)
        except (ConnectionError, OSError):
            self.close()

    def post(self, name: str, frame: Tuple[List[Any], int]) -> None:
        with self._condition:
            if self.closed:
                return
            self._events[name] = frame  # replaces an unsent older event
            if self._writer is None:
                self._writer = Thread(target=self._write_events, name="unienv-remote-events", daemon=True)
                self._writer.start()
            self._condition.notify()

    def _write_events(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self.closed or self._events)
                if self.closed:
                    return
                frame = self._events.pop(next(iter(self._events)))
            self.send(frame)

    def unsubscribe(self, name: str) -> None:
        resource = self.subscriptions.pop(name, None)
        if resource is not None:
            with self.server._lock:
                resource.subscribers = tuple(c for c in resource.subscribers if c is not self)
        with self._condition:
            self._events.pop(name, None)

    def close(self) -> None:
        with self._condition:
            if self.closed:
                return
            self.closed = True
            self._events.clear()
            self._condition.notify_all()
        self.transport.close()
        for name in list(self.subscriptions):
            self.unsubscribe(name)


class RemoteServer:
    """Serve registered objects to :class:`RemoteClient` connections.

    Example::

        with RemoteServer() as server:
            server.register("sim", env)
            server.serve("tcp://0.0.0.0:7000")  # blocks; or listen() in the background

    Each registered root runs on its own worker thread; calls on it, its world
    and its nodes are executed there one at a time, in arrival order, whichever
    client sent them. Only the methods in ``protocol.METHODS`` can be called.
    """

    def __init__(self, *, max_frame_size: int = wire.DEFAULT_MAX_FRAME_SIZE) -> None:
        self.max_frame_size = max_frame_size
        self._resources: Dict[str, _Resource] = {}
        self._names: Dict[int, str] = {}  # id(object) -> resource name
        self._groups: Dict[int, _Group] = {}  # id(world or base env) -> group
        self._listeners: List[Listener] = []
        self._connections: List[_Connection] = []
        self._threads: List[Thread] = []
        self._lock = Lock()
        self._closed = Event()

    # ------------------------------------------------------------------ registration

    def register(self, name: str, obj: Hosted, *, close: bool = True) -> str:
        """Host ``obj`` (an ``Env``, ``World`` or ``WorldNode``) under ``name``.

        An env built on a ``WorldEnv`` also exposes ``name/world`` and ``name/node``
        (the root node), and every node its children as ``<node name>/<child>``,
        so clients can compose them in components mode. Objects already exposed
        keep their first name and share their worker. With ``close=True`` the
        server closes ``obj`` (on its worker) when it shuts down.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Resource names must be non-empty strings")
        if not isinstance(obj, (Env, World, WorldNode)):
            raise TypeError(f"Expected an Env, World or WorldNode, got {type(obj).__name__}")
        base = obj.unwrapped if isinstance(obj, Env) else obj
        world = base.world if isinstance(base, (WorldEnv, WorldNode)) else (base if isinstance(base, World) else None)
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("Server is closed")
            group = self._groups.get(id(world)) if world is not None else None
            group = group or self._groups.get(id(base))
            created = group is None
            if created:
                group = _Group(name)
            added: Dict[str, _Resource] = {}
            names: Dict[int, str] = {}
            try:
                self._expose(name, obj, group, added, names)
            except Exception:
                if created:
                    group.queue.put(None)
                raise
            for resource in added.values():
                resource.static = resource.descriptor()
            self._resources.update(added)
            self._names.update(names)
            group.resources.extend(added.values())
            for key in {id(base), id(world)} - {id(None)}:
                self._groups.setdefault(key, group)
            if close and all(owned is not obj for owned in group.owned):
                group.owned.append(obj)
        return self._names[id(obj)]

    def _expose(self, name: str, obj: Hosted, group: _Group, added: Dict[str, _Resource],
                names: Dict[int, str]) -> str:
        existing = self._names.get(id(obj)) or names.get(id(obj))
        if existing is not None:
            return existing
        if name in self._resources or name in added:
            raise ValueError(f"Resource name {name!r} is already registered")
        kind = "env" if isinstance(obj, Env) else "world" if isinstance(obj, World) else "node"
        resource = _Resource(name, obj, kind, group)
        added[name], names[id(obj)] = resource, name
        base = obj.unwrapped if isinstance(obj, Env) else obj
        if isinstance(base, WorldEnv):
            resource.links["world"] = self._expose(name + "/world", base.world, group, added, names)
            resource.links["node"] = self._expose(name + "/node", base.node, group, added, names)
        elif kind == "node":
            world = obj.world
            resource.links["world"] = None if world is None else self._expose(name + "/world", world, group, added, names)
            resource.links["children"] = {
                child.name: self._expose(f"{name}/{quote(child.name, safe='')}", child, group, added, names)
                for child in getattr(obj, "nodes", ())
            }
        return name

    @property
    def resources(self) -> List[str]:
        """Names of every hosted resource."""
        return list(self._resources)

    # ------------------------------------------------------------------ connections

    def listen(self, address: str = "tcp://127.0.0.1:0") -> str:
        """Accept connections on ``address`` in the background; return the bound address.

        ``address`` is ``tcp://host:port``, ``unix:///path`` or ``ws://host:port``
        (the latter requires ``websockets``, from the ``network`` extra).
        """
        if self._closed.is_set():
            raise RuntimeError("Server is closed")
        listener = listen(address, on_connect=self._accept, serve_connection=self._serve,
                          max_frame_size=self.max_frame_size)
        with self._lock:
            self._listeners.append(listener)
        return listener.address

    def serve(self, address: str = "tcp://127.0.0.1:0") -> None:
        """Listen on ``address`` and block until :meth:`close` (or Ctrl-C)."""
        _log.info("Serving %s on %s", ", ".join(self._resources), self.listen(address))
        try:
            self._closed.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def connect(self, **client_options: Any) -> RemoteClient:
        """An in-process client over a :class:`MemoryTransport` (mainly for tests)."""
        from .client import RemoteClient
        client_side, server_side = MemoryTransport.pair(self.max_frame_size)
        self._accept(server_side)
        return RemoteClient(client_side, **client_options)

    def _accept(self, transport: Transport) -> None:
        thread = Thread(target=self._serve, args=(transport,), name="unienv-remote-reader", daemon=True)
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()] + [thread]
        thread.start()

    def _serve(self, transport: Transport) -> None:
        """Read requests from one connection until it closes."""
        connection = _Connection(self, transport)
        with self._lock:
            if self._closed.is_set():
                transport.close()
                return
            self._connections.append(connection)
        try:
            while not connection.closed:
                document, buffers = transport.recv()
                self._dispatch(connection, wire.parse(document), buffers)
        except wire.WireError as exc:
            _log.warning("Closing a connection that sent a malformed frame: %s", exc)
        except (ConnectionError, OSError):
            pass
        finally:
            connection.close()
            with self._lock:
                if connection in self._connections:
                    self._connections.remove(connection)

    def _dispatch(self, connection: _Connection, message: Any, buffers: List[Any]) -> None:
        if (type(message) is not list or len(message) != 5 or type(message[0]) is not int
                or type(message[1]) is not str or type(message[2]) is not str):
            raise wire.WireError("Requests must be [id, target, method, args, kwargs]")
        request_id, target, method = message[:3]
        if target == "":
            try:
                value = self._protocol_call(connection, method, *wire.restore(message[3:], buffers))
                connection.send(wire.encode([request_id, OK, value]))
            except wire.WireError:
                raise
            except Exception as exc:
                connection.send(_error(request_id, exc))
            return
        resource = self._resources.get(target)
        if resource is None or method not in resource.methods:
            exc: Exception = (KeyError(f"No resource named {target!r}") if resource is None else
                              AttributeError(f"{resource.kind} {target!r} does not expose {method!r}"))
            wire.restore(message[3:], buffers)  # still validate the frame
            connection.send(_error(request_id, exc))
            return
        try:
            args, kwargs = wire.restore(message[3:], buffers, resource.backend, resource.device)
        except wire.WireError:
            raise
        except Exception as exc:  # e.g. the hosted backend cannot represent a dtype
            connection.send(_error(request_id, exc))
            return
        if type(args) is not list or type(kwargs) is not dict:
            raise wire.WireError("Request args must be a list and kwargs a dict")
        resource.group.queue.put((self._execute, (connection, resource, request_id, method, args, kwargs)))

    def _protocol_call(self, connection: _Connection, method: str, args: Any, kwargs: Any) -> Any:
        if method == "hello":
            version = kwargs.get("version")
            if version != PROTOCOL_VERSION:
                raise ValueError(f"Protocol version {version!r} is not supported (server: {PROTOCOL_VERSION})")
            if kwargs.get("compress"):
                from .compression import parse_spec
                connection.compress = parse_spec(kwargs["compress"])
                connection.compress_key = repr(sorted(connection.compress.items()))
            resources = {}
            for name, resource in list(self._resources.items()):
                group = resource.group
                connection.seen.setdefault(group, group.version)  # read before the spaces
                resources[name] = dict(resource.static, spaces=resource.space_update())
            return {"version": PROTOCOL_VERSION, "resources": resources}
        if method in ("subscribe", "unsubscribe"):
            name = args[0]
            resource = self._resources.get(name)
            if resource is None or resource.kind != "env":
                raise KeyError(f"No environment named {name!r}")
            if method == "subscribe" and name not in connection.subscriptions:
                connection.subscriptions[name] = resource
                with self._lock:
                    resource.subscribers += (connection,)
            elif method == "unsubscribe":
                connection.unsubscribe(name)
            return None
        raise AttributeError(f"Unknown protocol method {method!r}")

    # ------------------------------------------------------------------ worker side

    def _execute(self, connection: _Connection, resource: _Resource, request_id: int, method: str,
                 args: List[Any], kwargs: Dict[str, Any]) -> None:
        try:
            value = getattr(resource.obj, method)(*args, **kwargs)
            group = resource.group
            if method in RESET_METHODS:
                self._check_spaces(group)
            if resource.subscribers and method in PUBLISHED_METHODS:
                self._publish(resource, method, value)
            message = [request_id, OK, self._compress(connection, resource, method, value)]
            seen = connection.seen.get(group, 0)
            if group.version > seen:
                message.append({r.name: r.space_update() for r in group.resources if r.version > seen})
            frame = wire.encode(message, max_size=connection.transport.max_frame_size)
            connection.seen[group] = group.version
        except Exception as exc:
            frame = _error(request_id, exc)
        connection.send(frame)

    def _check_spaces(self, group: _Group) -> None:
        """Bump the version of resources whose spaces changed (cheap identity check first)."""
        for resource in group.resources:
            if resource.kind == "world":
                continue
            spaces = tuple(getattr(resource.obj, attr, None) for attr in SPACES)
            if all(new is old for new, old in zip(spaces, resource.spaces)):
                continue
            documents = tuple(_document(space) for space in spaces)
            resource.spaces = spaces
            if documents != resource.documents:
                resource.documents = documents
                group.version += 1
                resource.version = group.version

    def _compress(self, connection: _Connection, resource: _Resource, method: str, value: Any) -> Any:
        if not connection.compress:
            return value
        from .compression import compress_result
        return compress_result(value, resource, method, connection.compress)

    def _publish(self, resource: _Resource, method: str, value: Any) -> None:
        resource.sequence += 1
        frames: Dict[Any, Tuple[List[Any], int]] = {}
        for connection in resource.subscribers:
            frame = frames.get(connection.compress_key)
            if frame is None:
                try:
                    payload = self._compress(connection, resource, method, value)
                    frame = wire.encode([resource.name, EVENT, [method, resource.sequence, payload]], copy=True,
                                        max_size=connection.transport.max_frame_size)
                except (TypeError, ValueError) as exc:
                    _log.warning("Cannot publish %s of %r: %s", method, resource.name, exc)
                    return
                frames[connection.compress_key] = frame
            connection.post(resource.name, frame)

    # ------------------------------------------------------------------ shutdown

    def close(self) -> None:
        """Stop listening, drop every connection, close owned objects on their workers."""
        with self._lock:
            if self._closed.is_set():
                return
            self._closed.set()
            listeners, connections, threads = list(self._listeners), list(self._connections), list(self._threads)
        for connection in connections:
            connection.close()
        for listener in listeners:
            listener.close()
        for thread in threads:
            thread.join()
        groups = {id(group): group for group in self._groups.values()}.values()
        for group in groups:
            group.queue.put((self._close_owned, (group,)))
            group.queue.put(None)
        for group in groups:
            group.thread.join()

    @staticmethod
    def _close_owned(group: _Group) -> None:
        for obj in reversed(group.owned):
            try:
                obj.close()
            except Exception:
                _log.exception("Closing a hosted object failed")

    def __enter__(self) -> RemoteServer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
