"""Host environments, worlds, nodes, data transformations and policies for remote clients."""

from __future__ import annotations

import logging
import traceback
from queue import SimpleQueue
from threading import Condition, Event, Lock, Thread, Timer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union
from urllib.parse import quote

from unienv_interface.env_base.env import Env
from unienv_interface.policy import FuncPolicy, Policy
from unienv_interface.space.space_utils.serialization_utils import space_to_json
from unienv_interface.transformations.transformation import DataTransformation
from unienv_interface.world.env_composer import WorldEnv
from unienv_interface.world.node import WorldNode
from unienv_interface.world.world import World

from . import wire
from .protocol import (ATTRIBUTES, ERROR, EVENT, METHODS, OK, OPTIONAL_METHODS, PRIORITIES, PROTOCOL_VERSION,
                       PUBLISHED_METHODS, RESET_METHODS, SPACES)
from .transport import Listener, MemoryTransport, Transport, listen

if TYPE_CHECKING:
    from unienv_interface.backends import ComputeBackend
    from .client import RemoteClient

__all__ = ["RemoteServer"]

Hosted = Union[Env, World, WorldNode, DataTransformation, Policy]
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
            try:
                item[0](*item[1])
            except BaseException:  # the worker must outlive any failure (calls report theirs themselves)
                _log.exception("A remote worker task failed")


class _Resource:
    def __init__(self, name: str, obj: Hosted, kind: str, group: _Group) -> None:
        self.name, self.obj, self.kind, self.group = name, obj, kind, group
        self.methods = frozenset(m for m in METHODS[kind]
                                 if m not in OPTIONAL_METHODS or callable(getattr(obj, m, None)))
        try:
            self.backend, self.device = obj.backend, obj.device
        except AttributeError:  # a node without a world, a transformation
            self.backend, self.device = None, None
        self.related: Dict[str, Any] = _related(obj, kind)  # objects whose names are linked if hosted
        self.spaces: Tuple[Any, ...] = ()
        self.documents: Tuple[Any, ...] = ()
        self.version = 0
        self.sequence = 0
        self.subscribers: Tuple[_Connection, ...] = ()
        self.static: Dict[str, Any] = {}
        self.factory: Optional[_Factory] = None  # set for objects built by a factory
        self.released = False  # a factory-built object that has since been closed
        if kind != "world":
            self.spaces = tuple(getattr(obj, attr, None) for attr in SPACES)
            self.documents = tuple(_document(space) for space in self.spaces)

    def descriptor(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"kind": self.kind, "methods": sorted(self.methods)}
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


class _Factory:
    """An object built on demand (:meth:`RemoteServer.register_factory`) on its own worker."""

    def __init__(self, name: str, make: Callable[[], Hosted], release_after: Optional[float],
                 backend: Optional[ComputeBackend], device: Optional[Any]) -> None:
        self.name, self.make, self.release_after = name, make, release_after
        self.backend, self.device = backend, device
        self.group = _Group(name)
        self.resource: Optional[_Resource] = None  # while the object exists
        self.users: Set[_Connection] = set()  # connections that opened it
        self.generation = 0  # bumped by every open and release; a stale release does nothing
        self.timer: Optional[Timer] = None


def _kind(obj: Any) -> str:
    """The resource kind of a hostable object (``TypeError`` for anything else)."""
    if isinstance(obj, FuncPolicy):
        raise TypeError("A FuncPolicy passes its state through every call; host it as a Policy with "
                        "FuncPolicyBasedPolicy(func_policy)")
    if not isinstance(obj, (Env, World, WorldNode, DataTransformation, Policy)):
        raise TypeError(f"Expected an Env, World, WorldNode, DataTransformation or Policy, got {type(obj).__name__}")
    return ("env" if isinstance(obj, Env) else "world" if isinstance(obj, World)
            else "transformation" if isinstance(obj, DataTransformation)
            else "policy" if isinstance(obj, Policy) else "node")


def _related(obj: Hosted, kind: str) -> Dict[str, Any]:
    """The world / root node / child nodes ``obj`` links to (resolved to names at ``hello``)."""
    base = obj.unwrapped if isinstance(obj, Env) else obj
    if isinstance(base, WorldEnv):
        return {"world": base.world, "node": base.node}
    if kind == "node":
        return {"world": obj.world, "children": {child.name: child for child in getattr(obj, "nodes", ())}}
    return {}


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


def _not_open(name: str) -> Exception:
    return PermissionError(f"{name!r} is built on demand and this connection has not opened it (or released it): "
                           "get it through RemoteClient.resource / env / ... first")


def _restore(message: List[Any], buffers: List[Any], backend: Optional[ComputeBackend] = None,
             device: Optional[Any] = None) -> Any:
    """Decode the ``[args, kwargs]`` of a request (clients never send compressed arrays)."""
    return wire.restore(message[3:], buffers, backend, device, allow_codecs=False)


def _error(request_id: Any, exc: BaseException) -> Tuple[List[Any], int]:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return wire.encode([request_id, ERROR, [type(exc).__name__, str(exc), text]])


class _Connection:
    """Per-client state. Events go out through a lazily started writer (latest wins)."""

    def __init__(self, server: RemoteServer, transport: Transport) -> None:
        self.server, self.transport = server, transport
        # Largest frame sent to this client: the transport's limit, lowered by the client's own
        # limit if it sends one in ``hello`` (older clients do not).
        self.max_frame_size: int = transport.max_frame_size
        self.compress: Dict[str, Any] = {}
        self.compress_key: Any = None
        self.seen: Dict[_Group, int] = {}
        self.subscriptions: Dict[str, _Resource] = {}
        self.factories: Set[_Factory] = set()  # factory-built objects this connection opened
        self.closed = False
        self._events: Dict[str, Tuple[List[Any], int]] = {}
        self._condition = Condition()
        self._writer: Optional[Thread] = None

    def send(self, frame: Tuple[List[Any], int]) -> None:
        try:
            self.transport.send(*frame)
        except TimeoutError as exc:
            _log.warning("Closing a connection that stopped reading: %s", exc)
            self.close()
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
        # ``subscribe`` and ``_open`` check ``closed`` and register this connection under the
        # server lock, so this snapshot cannot miss one that saw the connection still open.
        with self.server._lock:
            names, factories = list(self.subscriptions), list(self.factories)
        for name in names:
            self.unsubscribe(name)
        for factory in factories:
            self.server._release_use(self, factory)


class RemoteServer:
    """Serve registered objects to :class:`RemoteClient` connections.

    Example::

        with RemoteServer() as server:
            server.register("sim", env)  # or register_recursive() to also host its world and nodes
            server.serve("tcp://0.0.0.0:7000")  # blocks; or listen() in the background

    Each registered root runs on its own worker thread, shared by every hosted object
    with the same world; calls on them are executed there one at a time, in arrival
    order, whichever client sent them. Only the methods in ``protocol.METHODS`` can be
    called. :meth:`register_factory` hosts an object that is built when a client first
    opens it and closed once no client uses it.

    Replies are sent from the worker, so a client that stops reading would stall every
    client of that worker: a connection whose send makes no progress for ``send_timeout``
    seconds is closed (TCP and Unix sockets; ``None`` waits forever).
    """

    def __init__(self, *, max_frame_size: int = wire.DEFAULT_MAX_FRAME_SIZE,
                 send_timeout: Optional[float] = 60.0) -> None:
        if send_timeout is not None and not send_timeout > 0:
            raise ValueError(f"send_timeout must be > 0 seconds or None, got {send_timeout!r}")
        self.max_frame_size, self.send_timeout = max_frame_size, send_timeout
        self._resources: Dict[str, _Resource] = {}
        self._factories: Dict[str, _Factory] = {}
        self._names: Dict[int, str] = {}  # id(object) -> resource name
        self._groups: Dict[int, _Group] = {}  # id(world or base env) -> group
        self._listeners: List[Listener] = []
        self._connections: List[_Connection] = []
        self._threads: List[Thread] = []
        self._lock = Lock()
        self._closed = Event()

    # ------------------------------------------------------------------ registration

    def register(self, name: str, obj: Hosted, *, close: bool = True,
                 backend: Optional[ComputeBackend] = None, device: Optional[Any] = None) -> str:
        """Host ``obj`` (an ``Env``, ``World``, ``WorldNode``, ``DataTransformation`` or ``Policy``) under ``name``.

        Only ``obj`` itself is exposed; use :meth:`register_recursive` to also expose
        the world and nodes it is built from. Objects sharing a world (e.g. an env and
        its world registered separately) run on the same worker. Registering an object
        that is already hosted returns its existing name. With ``close=True`` the server
        closes ``obj`` (on its worker) when it shuts down.

        A ``Policy`` runs on its own worker like an env; the observations and contexts a
        client sends are rebuilt on the policy's backend / device. Host a ``FuncPolicy``
        as ``FuncPolicyBasedPolicy(func_policy)``.

        A ``DataTransformation`` has no backend of its own: ``backend`` / ``device``
        (default: numpy, no device) are where the source spaces and data of its
        calls are rebuilt on the server. Other objects use their own.
        """
        return self._register(name, obj, close=close, recursive=False, backend=backend, device=device)

    def register_recursive(self, name: str, obj: Hosted, *, close: bool = True) -> str:
        """Like :meth:`register`, but also expose everything ``obj`` is built from.

        An env built on a ``WorldEnv`` also exposes ``name/world`` and ``name/node``
        (the root node), and every node exposes its world (``<node name>/world``, unless
        already hosted) and its children as ``<node name>/<child>``, so clients can
        compose them in components mode. Objects already hosted keep their first name.
        """
        return self._register(name, obj, close=close, recursive=True)

    def _register(self, name: str, obj: Hosted, *, close: bool, recursive: bool,
                  backend: Optional[ComputeBackend] = None, device: Optional[Any] = None) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError("Resource names must be non-empty strings")
        _kind(obj)
        if (backend is not None or device is not None) and not isinstance(obj, DataTransformation):
            raise ValueError("backend / device are only given for a DataTransformation (others have their own)")
        base = obj.unwrapped if isinstance(obj, (Env, Policy)) else obj
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
                self._expose(name, obj, group, added, names, recursive)
            except Exception:
                if created:
                    group.queue.put(None)
                raise
            for resource in added.values():
                if resource.kind == "transformation":
                    from unienv_interface.backends.numpy import NumpyComputeBackend
                    resource.backend, resource.device = backend or NumpyComputeBackend, device
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
                names: Dict[int, str], recursive: bool) -> str:
        existing = self._names.get(id(obj)) or names.get(id(obj))
        if existing is not None:
            return existing
        if name in self._resources or name in added or name in self._factories:
            raise ValueError(f"Resource name {name!r} is already registered")
        kind = _kind(obj)
        resource = _Resource(name, obj, kind, group)
        added[name], names[id(obj)] = resource, name
        if recursive:
            related = resource.related
            for key in ("world", "node"):
                if related.get(key) is not None:
                    self._expose(f"{name}/{key}", related[key], group, added, names, recursive)
            for child_name, child in related.get("children", {}).items():
                self._expose(f"{name}/{quote(child_name, safe='')}", child, group, added, names, recursive)
        return name

    def register_factory(self, name: str, factory: Callable[[], Hosted], *, release_after: Optional[float] = 60.0,
                         backend: Optional[ComputeBackend] = None, device: Optional[Any] = None) -> str:
        """Host the object ``factory()`` builds, only while clients use it.

        The object is built on its own worker thread when a client first opens ``name``
        (``client.env(name)``, ``client.resource(name)``, ...; its spaces and attributes
        are read then), and closed there ``release_after`` seconds after the last client
        that opened it disconnects or calls ``close()`` on its proxy (``0``: at once,
        ``None``: never). The next client gets a newly built object, which it resets
        first like any fresh env. ``backend`` / ``device`` are given for a factory of
        ``DataTransformation`` s, as in :meth:`register`. Only the object itself is
        hosted (no :meth:`register_recursive` counterpart).
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Resource names must be non-empty strings")
        if not callable(factory):
            raise TypeError(f"factory must be callable, got {type(factory).__name__}")
        if release_after is not None and not release_after >= 0:
            raise ValueError(f"release_after must be >= 0 seconds or None, got {release_after!r}")
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("Server is closed")
            if name in self._resources or name in self._factories:
                raise ValueError(f"Resource name {name!r} is already registered")
            record = _Factory(name, factory, release_after, backend, device)
            self._factories[name] = record
            self._groups[id(record)] = record.group
        return name

    # ------------------------------------------------------------------ factory-built objects

    def _open(self, connection: _Connection, record: _Factory, request_id: int) -> None:
        """Worker side of ``open``: build the object if needed and count ``connection`` as a user."""
        try:
            with self._lock:
                record.generation += 1  # a release queued before this open no longer applies
            if record.resource is None:
                self._build(record)
            with self._lock:
                if connection.closed:  # its close() already released what it used
                    if not record.users:
                        self._schedule_release(record)
                    return
                record.users.add(connection)
                connection.factories.add(record)
                resource = record.resource
            group = record.group
            descriptor = dict(resource.static, **self._links(resource), spaces=resource.space_update(), factory=True)
            frame = wire.encode([request_id, OK, descriptor], max_size=connection.max_frame_size)
            connection.seen[group] = group.version
        except BaseException as exc:  # e.g. SystemExit from the factory: still answer, keep the worker
            frame = _error(request_id, exc)
        connection.send(frame)

    def _build(self, record: _Factory) -> None:
        obj = record.make()
        try:
            kind = _kind(obj)
            if kind != "transformation" and (record.backend is not None or record.device is not None):
                raise ValueError("backend / device are only given for a DataTransformation (others have their own)")
            resource = _Resource(record.name, obj, kind, record.group)
            if kind == "transformation":
                from unienv_interface.backends.numpy import NumpyComputeBackend
                resource.backend, resource.device = record.backend or NumpyComputeBackend, record.device
            resource.static = resource.descriptor()
        except BaseException:
            self._close_quietly(obj)
            raise
        resource.factory = record
        group = record.group
        group.version += 1  # clients see the spaces of a rebuilt object as an update
        resource.version = group.version
        with self._lock:
            record.resource = resource
            group.resources, group.owned = [resource], [obj]
            self._resources[record.name] = resource

    def _release_use(self, connection: _Connection, record: _Factory) -> None:
        """``connection`` stopped using ``record``'s object (``release`` call or disconnect)."""
        with self._lock:
            connection.factories.discard(record)
            if connection in record.users:
                record.users.discard(connection)
                if not record.users:
                    self._schedule_release(record)
        connection.unsubscribe(record.name)  # a connection only follows what it has open

    def _schedule_release(self, record: _Factory) -> None:
        """Queue closing the object in ``release_after`` seconds, unless it is opened again (holds the lock)."""
        record.generation += 1
        if record.timer is not None:
            record.timer.cancel()
            record.timer = None
        if record.resource is None or record.release_after is None or self._closed.is_set():
            return
        task = (self._release_if_idle, (record, record.generation))
        if record.release_after == 0:
            record.group.queue.put(task)
        else:
            record.timer = Timer(record.release_after, record.group.queue.put, args=(task,))
            record.timer.daemon = True
            record.timer.start()

    def _release_if_idle(self, record: _Factory, generation: int) -> None:
        """Worker side of a release: close the object if nobody opened it since the release was queued."""
        with self._lock:
            resource = record.resource
            if record.generation != generation or record.users or resource is None or self._closed.is_set():
                return
            record.resource, record.timer = None, None
            record.group.resources, record.group.owned = [], []
            self._resources.pop(record.name, None)
            resource.released = True
            subscribers, resource.subscribers = resource.subscribers, ()
        for connection in subscribers:
            connection.subscriptions.pop(record.name, None)
        self._close_quietly(resource.obj)

    @staticmethod
    def _close_quietly(obj: Any) -> None:
        if not callable(getattr(obj, "close", None)):
            return
        try:
            obj.close()
        except Exception:
            _log.exception("Closing a hosted object failed")

    def _links(self, resource: _Resource) -> Dict[str, Any]:
        """Names of the hosted objects ``resource`` links to (unhosted ones are left out)."""
        def hosted(obj: Any) -> Optional[str]:
            return None if obj is None else self._names.get(id(obj))
        links: Dict[str, Any] = {}
        related = resource.related
        if "world" in related:
            links["world"] = hosted(related["world"])
        if "node" in related:
            links["node"] = hosted(related["node"])
        if "children" in related:
            links["children"] = {key: hosted(child) for key, child in related["children"].items()
                                 if hosted(child) is not None}
        return links

    @property
    def resources(self) -> List[str]:
        """Names of every hosted resource (factories included, built or not)."""
        return list(dict.fromkeys([*self._resources, *self._factories]))

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
        transport.send_timeout = self.send_timeout
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
        except Exception:  # never let a bad request kill the reader thread uncleanly
            _log.exception("Closing a connection after an unexpected error")
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
        if target == "" and method in ("open", "release"):
            args = _restore(message, buffers)[0]
            valid = type(args) is list and len(args) == 1 and type(args[0]) is str
            record = self._factories.get(args[0]) if valid else None
            if record is None:
                connection.send(_error(request_id, KeyError(f"No factory-hosted resource named {args!r}")))
            elif method == "open":
                record.group.queue.put((self._open, (connection, record, request_id)))
            else:
                self._release_use(connection, record)
                connection.send(wire.encode([request_id, OK, None]))
            return
        if target == "":
            try:
                value = self._protocol_call(connection, method, *_restore(message, buffers))
                connection.send(wire.encode([request_id, OK, value], max_size=connection.max_frame_size))
            except wire.WireError:
                raise
            except Exception as exc:
                connection.send(_error(request_id, exc))
            return
        resource = self._resources.get(target)
        if resource is not None and resource.factory is not None and connection not in resource.factory.users:
            _restore(message, buffers)
            connection.send(_error(request_id, _not_open(target)))
            return
        if resource is None and target in self._factories:
            _restore(message, buffers)
            connection.send(_error(request_id, _not_open(target)))
            return
        if resource is None or method not in resource.methods:
            exc: Exception = (KeyError(f"No resource named {target!r}") if resource is None else
                              AttributeError(f"{resource.kind} {target!r} does not expose {method!r}"))
            _restore(message, buffers)  # still validate the frame
            connection.send(_error(request_id, exc))
            return
        try:
            args, kwargs = _restore(message, buffers, resource.backend, resource.device)
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
            limit = kwargs.get("max_frame_size")
            if type(limit) is int and limit > 0:
                connection.max_frame_size = min(limit, connection.transport.max_frame_size)
            if kwargs.get("compress"):
                from .compression import parse_spec
                connection.compress = parse_spec(kwargs["compress"])
                connection.compress_key = repr(sorted(connection.compress.items()))
            resources = {}
            for name, resource in list(self._resources.items()):
                if resource.factory is not None:
                    continue  # listed below, opened on demand
                group = resource.group
                connection.seen.setdefault(group, group.version)  # read before the spaces
                resources[name] = dict(resource.static, **self._links(resource), spaces=resource.space_update())
            if kwargs.get("factories"):  # clients that know how to open them
                for name in list(self._factories):
                    resources[name] = {"kind": None, "factory": True}
            return {"version": PROTOCOL_VERSION, "resources": resources}
        if method in ("subscribe", "unsubscribe"):
            name = args[0]
            if method == "unsubscribe" and (name in connection.subscriptions or name in self._factories):
                connection.unsubscribe(name)  # allowed after the factory object was released
                return None
            resource = self._resources.get(name)
            if resource is None and name in self._factories or (
                    resource is not None and resource.factory is not None and connection not in resource.factory.users):
                raise _not_open(name)
            if resource is None or resource.kind != "env":
                raise KeyError(f"No environment named {name!r}")
            if method == "subscribe":
                with self._lock:  # see _Connection.close
                    if connection.closed:
                        raise ConnectionError("Connection closed")
                    if name not in connection.subscriptions:
                        connection.subscriptions[name] = resource
                        resource.subscribers += (connection,)
            elif method == "unsubscribe":
                connection.unsubscribe(name)
            return None
        raise AttributeError(f"Unknown protocol method {method!r}")

    # ------------------------------------------------------------------ worker side

    def _execute(self, connection: _Connection, resource: _Resource, request_id: int, method: str,
                 args: List[Any], kwargs: Dict[str, Any]) -> None:
        try:
            if resource.released:
                raise _not_open(resource.name)
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
            frame = wire.encode(message, max_size=connection.max_frame_size)
            connection.seen[group] = group.version
        except BaseException as exc:  # e.g. SystemExit from the hosted object: still answer, keep the worker
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
        frames: Dict[Any, Optional[Tuple[List[Any], int]]] = {}  # by compression: None if it failed
        for connection in resource.subscribers:
            key = connection.compress_key
            if key not in frames:
                try:
                    payload = self._compress(connection, resource, method, value)
                    frames[key] = wire.encode([resource.name, EVENT, [method, resource.sequence, payload]], copy=True)
                except Exception as exc:  # never fails the call that produced the result
                    _log.warning("Cannot publish %s of %r: %s", method, resource.name, exc)
                    frames[key] = None
            frame = frames[key]
            if frame is None:
                continue
            if frame[1] > connection.max_frame_size:
                _log.warning("Cannot publish %s of %r to a client: the %d-byte frame exceeds its %d-byte limit",
                             method, resource.name, frame[1], connection.max_frame_size)
                continue
            connection.post(resource.name, frame)

    # ------------------------------------------------------------------ shutdown

    def close(self) -> None:
        """Stop listening, drop every connection, close owned objects on their workers."""
        with self._lock:
            if self._closed.is_set():
                return
            self._closed.set()
            listeners, connections, threads = list(self._listeners), list(self._connections), list(self._threads)
            for record in self._factories.values():
                if record.timer is not None:
                    record.timer.cancel()
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
