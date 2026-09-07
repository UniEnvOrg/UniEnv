"""Explicit resource registry and serialized execution domains."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
import logging
import math
from threading import Condition, Event, Lock, Thread, current_thread
import time
from urllib.parse import quote
from uuid import uuid4
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING, TypedDict, Union

from unienv_interface.env_base.env import Env
from unienv_interface.world.world import World
from unienv_interface.world.node import WorldNode
from unienv_interface.world.env_composer import WorldEnv
from unienv_interface.space.space_utils.serialization_utils import space_to_json

from .codec import Codec, CodecError, PROTOCOL_VERSION, DEFAULT_MAX_MESSAGE_SIZE, map_arrays
from .errors import RemoteError
from .protocol import FIELDS, METHODS, PRIORITIES
from .transport import MemoryTransport, WebSocketTransport

if TYPE_CHECKING:
    from concurrent.futures import Future
    from unienv_interface.backends import ComputeBackend
    from websockets.sync.server import Server, ServerConnection
    from .client import RemoteClient
    from .protocol import Descriptor, ExecutionMode, OperationKind, ResourceKind, WireMessage
    from .transport import MessageTransport

    class _Operation(TypedDict):
        id: str
        kind: OperationKind
        recovered: bool

HostedResource = Union[Env, World, WorldNode]
ResourceFactory = Callable[[], HostedResource]
DomainKey = Union[int, str, Tuple[str, int]]
SnapshotCaptures = Dict[Tuple[str, str], Tuple[Dict[str, Any], Optional[Exception]]]

try:
    SERVER_VERSION = version("unienv")
except PackageNotFoundError:
    SERVER_VERSION = "unknown"

@dataclass
class _Domain:
    executor: Optional[ThreadPoolExecutor] = None
    resources: List[_Resource] = field(default_factory=list)
    owner: Optional[_Session] = None
    mode: Optional[ExecutionMode] = None
    operation: Optional[_Operation] = None
    faulted: bool = False
    revision: int = 0
    sequence: int = 0
    subscriptions: Dict[Tuple[str, str], Tuple[_Session, _Resource, List[str]]] = field(default_factory=dict)
    factory: Optional[ResourceFactory] = None
    root_id: Optional[str] = None
    instantiated: bool = True
    last_activity: float = field(default_factory=time.monotonic)


@dataclass
class _Resource:
    id: str
    obj: Optional[HostedResource]
    kind: Optional[ResourceKind]
    domain: _Domain
    world_id: Optional[str] = None
    children: Dict[str, str] = field(default_factory=dict)
    node_id: Optional[str] = None
    latest: Dict[str, Any] = field(default_factory=dict)


class _Session:
    """Separate writer: slow observers cannot block the world's worker."""

    def __init__(self, server: RemoteServer, transport: MessageTransport) -> None:
        self.server, self.transport = server, transport
        self.codec = server.codec
        self.id = uuid4().hex
        self.condition = Condition()
        self.responses: Deque[bytes] = deque()
        self.snapshots: Dict[str, bytes] = {}
        self.closed = False
        self.ready = False
        self.writer = Thread(target=self._write, name="unienv-send", daemon=True)
        self.reader = Thread(target=self.run, name="unienv-recv", daemon=True)

    def enqueue(self, payload: bytes, subscription: Optional[str] = None) -> None:
        with self.condition:
            if self.closed:
                return
            if subscription is None:
                if len(self.responses) >= 64:
                    self.close()
                    return
                self.responses.append(payload)
            else:
                self.snapshots[subscription] = payload
            self.condition.notify_all()

    def _write(self) -> None:
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.responses or self.snapshots)
                    if self.closed:
                        return
                    payload = self.responses.popleft() if self.responses else self.snapshots.pop(next(iter(self.snapshots)))
                self.transport.send(payload)
        except (ConnectionError, OSError):
            self.close()

    def close(self) -> None:
        with self.condition:
            if self.closed:
                return
            self.closed = True
            self.responses.clear()
            self.snapshots.clear()
            self.condition.notify_all()
        self.transport.close()

    def run(self) -> None:
        self.writer.start()
        try:
            while not self.closed:
                request = self.codec.decode(self.transport.recv())
                response = self.server._request(self, request)
                self.enqueue(self.codec.encode(response))
        except (ConnectionError, OSError, CodecError):
            pass
        finally:
            self.close()
            self.writer.join()
            for domain in self.server._domains.values():
                domain.executor.submit(self.server._detach, domain, self)
            with self.server._lock:
                self.server._sessions.discard(self)


class RemoteServer:
    """Host objects or lazy root factories. Register resources before connecting.

    Registration doesn't call lifecycle methods. All discovery, lifecycle,
    snapshots, and cleanup execute on the resource's dedicated world worker.
    Only factory registrations are evicted after idle_timeout seconds.
    """

    def __init__(self, *, max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
                 idle_timeout: Optional[float] = 60.0) -> None:
        if idle_timeout is not None and (not math.isfinite(idle_timeout) or idle_timeout <= 0):
            raise ValueError("idle_timeout must be positive and finite, or None")
        self.codec = Codec(max_message_size)
        self.idle_timeout = idle_timeout
        self._resources: Dict[str, _Resource] = {}
        self._objects: Dict[int, str] = {}
        self._domains: Dict[DomainKey, _Domain] = {}
        self._world_owners: Dict[int, _Domain] = {}
        self._disposal_reservations: Set[int] = set()
        self._sessions: Set[_Session] = set()
        self._lock = Lock()
        self._started = False
        self._closed = False
        self._websocket: Optional[Server] = None
        self._websocket_thread: Optional[Thread] = None
        self._reaper_stop = Event()
        self._reaper: Optional[Thread] = None
        if idle_timeout is not None:
            self._reaper = Thread(target=self._reap, name="unienv-reaper", daemon=True)
            self._reaper.start()

    def register(self, resource_id: str, obj_or_factory: Union[HostedResource, ResourceFactory]) -> str:
        """Register an Env, World, WorldNode, or zero-arg root factory.

        WorldEnv exports its world and complete node tree automatically.
        Re-registering the same object returns its existing ID, without aliases.
        World/node relationships and child names must stay fixed after registration.
        Factories run on the world worker on first access (including discovery).
        Their topology must remain identical across idle eviction and revival.
        A root WorldNode with world=None owns a separate domain, including its
        detached children. Each node must declare a backend; device defaults to
        None for RPC coercion. Devices secretly shared by detached nodes have no
        ownership protection, so their construction must be trusted.
        """
        with self._lock:
            if self._started or self._closed:
                raise RuntimeError("Register resources before starting the server")
            if callable(obj_or_factory):
                if not isinstance(resource_id, str) or not resource_id:
                    raise ValueError("resource_id must be a nonempty string")
                if resource_id in self._resources:
                    raise ValueError(f"Resource ID already registered: {resource_id}")
                domain = _Domain(factory=obj_or_factory, root_id=resource_id, instantiated=False,
                                 executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="unienv-world"))
                resource = _Resource(resource_id, None, None, domain)
                domain.resources = [resource]
                self._resources[resource_id] = resource
                self._domains[resource_id] = domain
                return resource_id
            return self._register(resource_id, obj_or_factory)

    def _register(self, resource_id: str, obj: HostedResource, target: Optional[_Domain] = None) -> str:
        resources, objects, domains = self._resources.copy(), self._objects.copy(), self._domains.copy()
        world_owners = self._world_owners.copy()
        members = {key: list(domain.resources) for key, domain in domains.items()}
        if target is not None:
            resources = {key: value for key, value in resources.items() if value.domain is not target}
            members[target.root_id] = []
        root_world: Optional[Union[Env, World]] = None
        detached_root = obj if isinstance(obj, WorldNode) and obj.world is None else None
        detached_key: Optional[DomainKey] = ("node", id(detached_root)) if detached_root is not None else None

        def visit(resource_id: str, obj: HostedResource) -> str:
            nonlocal root_world
            if not isinstance(resource_id, str) or not resource_id:
                raise ValueError("resource_id must be a nonempty string")
            if id(obj) in self._disposal_reservations:
                raise ValueError("Resource is reserved for disposal")
            if resource_id in resources and resources[resource_id].obj is not obj:
                raise ValueError(f"Resource ID already registered: {resource_id}")
            if id(obj) in objects:
                existing = resources[objects[id(obj)]]
                if target is not None and existing.domain is not target:
                    raise ValueError("Factory trees must not share registered objects")
                expected_key = target.root_id if target is not None else detached_key
                if detached_root is not None and obj is not detached_root and existing.domain is not domains.get(expected_key):
                    raise ValueError("Detached node trees must not share registered nodes across domains")
                return objects[id(obj)]
            if not isinstance(obj, (Env, World, WorldNode)):
                raise TypeError("Expected Env, World, or WorldNode")
            kind: ResourceKind = "env" if isinstance(obj, Env) else ("world" if isinstance(obj, World) else "node")
            base = obj.unwrapped if isinstance(obj, Env) else obj
            world = base.world if isinstance(base, (WorldEnv, WorldNode)) else base
            if world is None:
                if detached_root is None:
                    raise ValueError("Only detached node roots and their children may omit a world")
                self._backend_device(obj)
            if id(base) in self._disposal_reservations or id(world) in self._disposal_reservations:
                raise ValueError("Underlying world or environment is reserved for disposal")
            key: DomainKey = detached_key if world is None else id(world)
            if target is not None:
                if world is not None:
                    root_world = world if root_world is None else root_world
                    if world is not root_world:
                        raise ValueError("Factory trees must have their own single world")
                key = target.root_id
            if key not in domains:
                domains[key] = _Domain()
                members[key] = []
            domain = domains[key]
            if world is not None:
                owner = world_owners.get(id(world))
                if owner is not None and owner is not domain:
                    raise ValueError("Underlying world or environment already belongs to another domain")
                world_owners[id(world)] = domain
            resource = _Resource(resource_id, obj, kind, domain)
            resources[resource_id] = resource
            objects[id(obj)] = resource_id
            members[key].append(resource)
            if isinstance(base, (WorldEnv, WorldNode)) and world is not None:
                resource.world_id = visit(resource_id + "/world", world)
            if isinstance(base, WorldEnv):
                resource.node_id = visit(resource_id + "/node", base.node)
            if isinstance(obj, WorldNode):
                for child in getattr(obj, "nodes", ()):
                    if child.world is not world:
                        raise ValueError("All children must retain their server-side world")
                    if child.name in resource.children:
                        raise ValueError("Duplicate child node name")
                    resource.children[child.name] = visit(resource_id + "/" + quote(child.name, safe=""), child)
            return resource_id

        canonical_id = visit(resource_id, obj)
        if target is not None and target.resources[0].kind is not None:
            def topology(items: Iterable[_Resource]) -> Dict[
                str, Tuple[Optional[ResourceKind], Optional[str], Optional[str], Dict[str, str]]
            ]:
                return {r.id: (r.kind, r.world_id, r.node_id, r.children) for r in items}

            if topology(target.resources) != topology(members[target.root_id]):
                raise ValueError("Factory resource topology must remain unchanged on revival")
        # Validation is complete before workers or existing domain state change.
        for domain in domains.values():
            if domain.executor is None:
                domain.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unienv-world")
        for key, domain in domains.items():
            domain.resources = members[key]
        self._resources, self._objects, self._domains = resources, objects, domains
        self._world_owners = world_owners
        return canonical_id

    def _backend_device(self, obj: HostedResource) -> Tuple[ComputeBackend, Optional[object]]:
        if isinstance(obj, WorldNode) and obj.world is None:
            backend = getattr(obj, "backend", None)
            if backend is None:
                raise ValueError("Detached nodes must declare a backend; the RPC device defaults to None")
            return backend, getattr(obj, "device", None)
        return obj.backend, obj.device

    def _instantiate(self, domain: _Domain) -> None:
        if domain.instantiated:
            return
        obj = domain.factory()
        cleanup: List[HostedResource] = []
        reserved: Set[int] = set()
        try:
            with self._lock:
                revived = domain.resources[0].kind is not None
                try:
                    self._register(domain.root_id, obj, target=domain)
                except Exception:
                    # No registration can interleave between validation and reservation.
                    try:
                        cleanup, reserved = self._reserve_rejected(obj)
                    except Exception:
                        logging.getLogger(__name__).exception("Rejected factory cleanup planning failed for %s", domain.root_id)
                    raise
        except Exception:
            try:
                self._dispose_rejected(cleanup)
            except Exception:
                logging.getLogger(__name__).exception("Rejected factory cleanup failed for %s", domain.root_id)
            finally:
                with self._lock:
                    self._disposal_reservations.difference_update(reserved)
            raise
        domain.instantiated = True
        domain.faulted = False
        if revived:
            domain.revision += 1

    def _reserve_rejected(self, obj: HostedResource) -> Tuple[List[HostedResource], Set[int]]:
        """Plan cleanup and reserve its identities while holding the registry lock."""
        protected = set(self._objects) | set(self._world_owners) | self._disposal_reservations
        objects: Dict[int, HostedResource] = {}
        children: Dict[int, List[int]] = {}
        worlds: List[int] = []

        def visit(current: HostedResource) -> None:
            key = id(current)
            if not isinstance(current, (Env, World, WorldNode)) or key in objects:
                return
            objects[key], children[key] = current, []
            if key in protected:
                return
            dependencies: List[HostedResource] = []
            if isinstance(current, WorldEnv):
                dependencies = [current.node, current.world]
            elif isinstance(current, Env):
                wrapped = getattr(current, "env", None)
                base = wrapped if isinstance(wrapped, Env) else current.unwrapped
                if base is not current:
                    dependencies = [base]
            elif isinstance(current, WorldNode):
                dependencies = list(getattr(current, "nodes", ()))
            for child in dependencies:
                visit(child)
                if id(child) in objects:
                    children[key].append(id(child))
            if isinstance(current, WorldNode) and current.world is not None:
                visit(current.world)
                worlds.append(id(current.world))

        def descendants(key: int, seen: Set[int]) -> None:
            if key in seen:
                return
            seen.add(key)
            for child in children[key]:
                descendants(child, seen)

        visited: Set[int] = set()
        reserved: Set[int] = set()
        cleanup: List[HostedResource] = []

        def select(key: int) -> None:
            if key in protected or key in visited:
                return
            covered: Set[int] = set()
            descendants(key, covered)
            if covered & (protected | visited):
                # Composite close hooks cascade; only close independent branches.
                visited.add(key)
                for child in children[key]:
                    select(child)
                return
            visited.update(covered)
            reserved.update(covered)
            cleanup.append(objects[key])

        visit(obj)
        if id(obj) in objects:
            select(id(obj))
        for key in worlds:
            select(key)
        self._disposal_reservations.update(reserved)
        return cleanup, reserved

    def _dispose_rejected(self, objects: Iterable[HostedResource]) -> None:
        for obj in objects:
            try:
                obj.close()
            except Exception:
                logging.getLogger(__name__).exception("Rejected factory object cleanup failed")

    def _discover(self, domain: _Domain) -> List[Descriptor]:
        domain.last_activity = time.monotonic()
        try:
            self._instantiate(domain)
            return self._freeze(self._descriptors(domain))
        finally:
            domain.last_activity = time.monotonic()

    def _reap(self) -> None:
        pending: Dict[DomainKey, Future[None]] = {}
        while not self._reaper_stop.wait(min(self.idle_timeout / 2, 5.0)):
            with self._lock:
                domains = list(self._domains.items())
            for key, domain in domains:
                if domain.factory is not None and (key not in pending or pending[key].done()):
                    pending[key] = domain.executor.submit(self._evict, domain)

    def _evict(self, domain: _Domain) -> None:
        if (not domain.instantiated or domain.owner is not None or domain.subscriptions
                or time.monotonic() - domain.last_activity < self.idle_timeout):
            return
        try:
            self._dispose(domain)
        except Exception:
            logging.getLogger(__name__).exception("Idle disposal failed for %s", domain.root_id)
        finally:
            with self._lock:
                for resource in domain.resources:
                    self._objects.pop(id(resource.obj), None)
                    resource.obj = None
                    resource.latest = {}
            domain.instantiated = False

    def attach(self, transport: MessageTransport) -> _Session:
        """Accept a connected MessageTransport, starting a session in background."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Server is closed")
            self._started = True
            session = _Session(self, transport)
            self._sessions.add(session)
            session.reader.start()
        return session

    def connect(self, **client_options: object) -> RemoteClient:
        """Create an in-memory RemoteClient using the full wire protocol."""
        from .client import RemoteClient
        client_transport, server_transport = MemoryTransport.pair()
        self.attach(server_transport)
        client_options.setdefault("max_message_size", self.codec.max_message_size)
        return RemoteClient(client_transport, **client_options)

    def listen(self, host: str = "127.0.0.1", port: int = 0) -> str:
        """Start the optional WebSocket listener; return its ws:// endpoint."""
        from websockets.sync.server import serve
        if self._closed or self._websocket is not None:
            raise RuntimeError("Server is closed or already listening")

        def handler(connection: ServerConnection) -> None:
            session = self.attach(WebSocketTransport(connection))
            session.reader.join()

        self._started = True
        self._websocket = serve(handler, host, port, max_size=self.codec.max_message_size,
                                max_queue=16, compression=None, close_timeout=1)
        self._websocket_thread = Thread(target=self._websocket.serve_forever, name="unienv-listen", daemon=True)
        self._websocket_thread.start()
        port = self._websocket.socket.getsockname()[1]
        return f"ws://{'[' + host + ']' if ':' in host else host}:{port}"

    def _descriptor(self, resource: _Resource) -> Descriptor:
        obj = resource.obj
        root = resource.domain.resources[0]
        result: Descriptor = {"id": resource.id, "kind": resource.kind, "world_id": resource.world_id,
                   "node_id": resource.node_id, "children": resource.children,
                   "revision": resource.domain.revision, "sequence": resource.domain.sequence,
                   "domain_id": root.world_id or root.id}
        for attr in ("name", "batch_size", "world_timestep", "world_subtimestep", "control_timestep",
                     "update_timestep", "render_mode", "render_fps", "supported_render_modes",
                     "has_reward", "has_termination_signal", "has_truncation_signal", "metadata"):
            if hasattr(obj, attr):
                result[attr] = getattr(obj, attr)
        for attr in ("action_space", "observation_space", "context_space"):
            if resource.kind != "world":
                space = getattr(obj, attr, None)
                result[attr] = space_to_json(space) if space is not None else None
        for attr in PRIORITIES:
            if resource.kind == "node":
                result[attr] = sorted(getattr(obj, attr), reverse=True)
        result["operations"] = sorted(self._methods(resource))
        return result

    def _methods(self, resource: _Resource) -> Set[str]:
        return {method for method, options in METHODS[resource.kind].items()
                if not options.get("optional") or callable(getattr(resource.obj, method, None))}

    def _descriptors(self, domain: _Domain) -> List[Descriptor]:
        return [self._descriptor(resource) for resource in domain.resources]

    def _request(self, session: _Session, request: Any) -> WireMessage:
        response: WireMessage = {"version": PROTOCOL_VERSION, "type": "response", "session_id": session.id,
                    "request_id": request.get("request_id") if isinstance(request, dict) else None,
                    "resource_id": request.get("resource_id") if isinstance(request, dict) else None,
                    "operation_id": request.get("operation_id") if isinstance(request, dict) else None}
        try:
            if not isinstance(request, dict) or request.get("type") != "request" or not isinstance(request.get("request_id"), str):
                raise RemoteError("invalid_request", "Expected a request with a string request_id")
            if request.get("version") != PROTOCOL_VERSION:
                raise RemoteError("version_mismatch", "Only protocol version 1 is supported")
            operation = request.get("op")
            if not isinstance(operation, str):
                raise RemoteError("invalid_request", "op must be a string")
            if operation == "hello":
                if session.ready:
                    raise RemoteError("invalid_session", "hello has already completed")
                limit = request.get("max_message_size", self.codec.max_message_size)
                if type(limit) is not int or limit < 8:
                    raise RemoteError("invalid_request", "Invalid max_message_size")
                session.codec = Codec(min(limit, self.codec.max_message_size))
                session.ready = True
                result: Any = {"session_id": session.id, "max_message_size": session.codec.max_message_size,
                          "server": "unienv", "server_version": SERVER_VERSION,
                          "protocol_versions": [PROTOCOL_VERSION]}
            else:
                if not session.ready or request.get("session_id") != session.id:
                    raise RemoteError("invalid_session", "Complete hello before making requests")
                if operation == "discover":
                    futures = [domain.executor.submit(self._discover, domain)
                               for domain in self._domains.values()]
                    result = []
                    for future in futures:
                        result.extend(future.result())
                else:
                    if not isinstance(request.get("resource_id"), str):
                        raise RemoteError("invalid_request", "resource_id must be a string")
                    resource = self._resources.get(request.get("resource_id"))
                    if resource is None:
                        roots = [d.root_id for d in self._domains.values() if d.factory is not None
                                 and request["resource_id"].startswith(d.root_id + "/")]
                        if roots:
                            resource = self._resources[max(roots, key=len)]
                    if resource is None:
                        raise RemoteError("not_found", "Unknown resource ID")
                    result = resource.domain.executor.submit(self._execute, session, resource, request).result()
            response["result"] = result
            # Freeze mutable array references while still serialized by _execute.
            session.codec.encode(response)
        except Exception as exc:
            response.update(type="error", error={"code": exc.code if isinstance(exc, RemoteError) else
                            ("serialization_error" if isinstance(exc, CodecError) else "execution_error"),
                            "message": str(exc), "uncertain": getattr(exc, "uncertain", False)})
            response.pop("result", None)
        return response

    def _claim(self, session: _Session, domain: _Domain, mode: Optional[ExecutionMode]) -> None:
        if mode not in ("env", "components"):
            raise RemoteError("invalid_request", "Unknown execution mode")
        if domain.owner is not None and (domain.owner is not session or domain.mode != mode):
            raise RemoteError("ownership_conflict", "This world already has a controller or a different execution mode")
        domain.owner, domain.mode = session, mode

    def _execute(self, session: _Session, resource: _Resource, request: WireMessage) -> Any:
        domain = resource.domain
        domain.last_activity = time.monotonic()
        try:
            if session.closed:
                raise RemoteError("invalid_session", "Session has closed")
            if request["op"] not in {"release", "unsubscribe"}:
                self._instantiate(domain)
            # A queued request may still hold a resource from before revival.
            resource = self._resources.get(request["resource_id"])
            if resource is None or resource.domain is not domain:
                raise RemoteError("not_found", "Unknown resource ID")
            return self._execute_active(session, resource, request)
        finally:
            domain.last_activity = time.monotonic()

    def _execute_active(self, session: _Session, resource: _Resource, request: WireMessage) -> Any:
        domain, op = resource.domain, request["op"]
        if session.closed:
            raise RemoteError("invalid_session", "Session has closed")
        if op == "describe":
            return self._freeze(self._descriptors(domain))
        if op == "subscribe":
            fields = request.get("fields", sorted(FIELDS[resource.kind] - {"render"}))
            if not isinstance(fields, list) or any(not isinstance(f, str) or f not in FIELDS[resource.kind] for f in fields):
                raise RemoteError("invalid_request", "Unsupported snapshot fields")
            subscription_id = request.get("subscription_id")
            if not isinstance(subscription_id, str) or not subscription_id:
                raise RemoteError("invalid_request", "subscription_id must be a nonempty string")
            key = (session.id, subscription_id)
            if key in domain.subscriptions:
                raise RemoteError("invalid_request", "Subscription ID already exists")
            domain.subscriptions[key] = (session, resource, fields)
            return subscription_id
        if op == "unsubscribe":
            subscription_id = request.get("subscription_id")
            domain.subscriptions.pop((session.id, subscription_id), None)
            with session.condition:
                session.snapshots.pop(subscription_id, None)
            return None
        if op == "release":
            if domain.owner is session:
                self._release(domain)
            return None
        if op == "acquire":
            if resource.kind == "node" and resource.world_id is None and request.get("mode") == "env":
                raise RemoteError("invalid_request", "Detached nodes support only components mode")
            self._claim(session, domain, request.get("mode"))
            return None
        if op not in {"begin", "complete", "abort", "call"}:
            raise RemoteError("unsupported_operation", "Unknown protocol operation")
        mode = "env" if resource.kind == "env" and op == "call" else "components"
        self._claim(session, domain, mode)
        if op == "begin":
            kind, operation_id = request.get("kind"), request.get("operation_id")
            if kind not in {"reset", "reload", "step"} or not isinstance(operation_id, str) or not operation_id:
                raise RemoteError("invalid_request", "begin requires a kind and operation_id")
            if domain.operation is not None:
                raise RemoteError("operation_in_progress", "A component operation is already active")
            if domain.faulted and kind not in {"reset", "reload"}:
                raise RemoteError("faulted", "Reset or reload the world after an interrupted operation")
            for member in domain.resources:
                member.latest = {}
            domain.operation = {"id": operation_id, "kind": kind, "recovered": False}
            return None
        if op in {"complete", "abort"}:
            self._check_operation(domain, request)
            if op == "abort":
                domain.faulted, domain.operation = True, None
                return None
            operation = domain.operation
            if operation["kind"] in {"reset", "reload"} and not operation["recovered"]:
                root = domain.resources[0]
                scope = "detached-root node" if root.kind == "node" and root.world_id is None else "world"
                raise RemoteError("invalid_operation", f"Recovery requires a successful {scope} reset or reload")
            domain.operation = None
            domain.faulted = False
            descriptors = self._freeze(self._descriptors(domain))
            captured: SnapshotCaptures = {}
            self._publish(domain, operation["id"], operation["kind"], captured=captured)
            self._boundary_descriptors(domain, domain.sequence, descriptors, captured)
            return self._freeze_result(session, request, descriptors)
        method = request.get("method")
        if not isinstance(method, str):
            raise RemoteError("invalid_request", "method must be a string")
        if method not in self._methods(resource):
            raise RemoteError("unsupported_operation", "Method is not exported")
        mutation = METHODS[resource.kind][method]["mutation"]
        if domain.faulted and method not in {"reset", "reload"} and not (domain.operation and domain.operation["kind"] in {"reset", "reload"}):
            raise RemoteError("faulted", "Reset or reload the world after an interrupted operation")
        if resource.kind != "env" and (mutation or domain.operation):
            self._check_operation(domain, request)
            if domain.operation["kind"] == "step" and method in {"reset", "reload", "after_reset", "after_reload"}:
                raise RemoteError("invalid_operation", "Reset lifecycle requires a reset/reload operation")
            if domain.operation["kind"] in {"reset", "reload"} and method in {"step", "set_next_action", "pre_environment_step", "post_environment_step"}:
                raise RemoteError("invalid_operation", "Step lifecycle requires a step operation")
        invoked = False
        try:
            args = request.get("args", ())
            kwargs = request.get("kwargs", {})
            if not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict):
                raise RemoteError("invalid_request", "Expected args sequence and kwargs dictionary")
            backend, device = self._backend_device(resource.obj)
            args, kwargs = map_arrays(args, backend, device), map_arrays(kwargs, backend, device)
            if resource.kind == "env" and mutation:
                for member in domain.resources:
                    member.latest = {}
            invoked = True
            value = getattr(resource.obj, method)(*args, **kwargs)
            if method in {"reset", "reload", "after_reset", "after_reload"}:
                domain.revision += 1
            recovery_root = (resource.kind == "world" or
                             (resource.kind == "node" and resource.world_id is None and resource is domain.resources[0]))
            if recovery_root and method in {"reset", "reload"}:
                domain.operation["recovered"] = True
            if resource.kind == "world" and method == "step":
                resource.latest = {"dt": value}
            if resource.kind == "env" and mutation:
                domain.faulted = False
            result = {"value": value, "descriptors": self._descriptors(domain)}
            captured = {}
            if resource.kind == "env" and mutation:
                # Detach mandatory values before optional getters can refresh
                # buffers still referenced by the method's return tuple.
                result = self._freeze(result)
                fields = (("observation", "reward", "terminated", "truncated", "info") if method == "step"
                          else ("context", "observation", "info"))
                resource.latest = dict(zip(fields, result["value"]))
                self._boundary_descriptors(domain, domain.sequence + 1, result["descriptors"], captured, resource)
            result = self._freeze_result(session, request, result)
            if resource.kind == "env" and mutation:
                self._publish(domain, request["operation_id"], method, env_resource=resource,
                              captured=captured)
            return result
        except Exception as exc:
            if mutation and invoked:
                domain.faulted = True
                domain.operation = None
                raise RemoteError("serialization_error" if isinstance(exc, CodecError) else "execution_error",
                                  str(exc), uncertain=True) from exc
            raise

    def _check_operation(self, domain: _Domain, request: WireMessage) -> None:
        if domain.operation is None or domain.operation["id"] != request.get("operation_id"):
            raise RemoteError("invalid_operation", "An explicit matching component operation is required")

    def _freeze(self, value: Any) -> Any:
        return self.codec.decode(self.codec.encode(value))

    def _boundary_descriptors(self, domain: _Domain, sequence: int, descriptors: List[Descriptor],
                              captured: SnapshotCaptures, env_resource: Optional[_Resource] = None) -> None:
        for resource, descriptor in zip(domain.resources, descriptors):
            descriptor["sequence"] = sequence
            if resource.kind == "env" and resource is not env_resource:
                continue
            try:
                selections = [fields for session, member, fields in domain.subscriptions.values()
                              if session is domain.owner and member is resource]
                allowed = (set(getattr(resource.obj, "remote_snapshot_fields", FIELDS["node"]))
                           if resource.kind == "node" else FIELDS[resource.kind])
                fields = sorted(((set().union(*selections) if selections else FIELDS[resource.kind]) & allowed) - {"render"})
                data = self._capture_snapshot(resource, fields, captured)
                descriptor["snapshot"] = {"sequence": sequence, "revision": domain.revision,
                                          "fields": fields, "data": data}
            except Exception:
                # Optional read caching must not turn a successful mutation into a failure.
                pass

    def _capture_snapshot(self, resource: _Resource, fields: Iterable[str], captured: SnapshotCaptures) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field in sorted(fields):
            key = (resource.id, field)
            if key not in captured:
                try:
                    captured[key] = (self._freeze(self._snapshot(resource, [field])), None)
                except Exception as exc:
                    captured[key] = ({}, exc)
            data, error = captured[key]
            if error is not None:
                raise error
            result.update(data)
        return result

    def _freeze_result(self, session: _Session, request: WireMessage, result: Any) -> Any:
        def encode() -> Any:
            frozen = self._freeze(result)
            session.codec.encode({"version": PROTOCOL_VERSION, "type": "response", "session_id": session.id,
                                  "request_id": request["request_id"], "resource_id": request.get("resource_id"),
                                  "operation_id": request.get("operation_id"), "result": frozen})
            return frozen

        try:
            return encode()
        except CodecError:
            # Keep the original result and its uncertainty semantics when optional
            # snapshots would exceed either peer's negotiated message limit.
            descriptors = result if isinstance(result, list) else result["descriptors"]
            if not any("snapshot" in descriptor for descriptor in descriptors):
                raise
            for descriptor in descriptors:
                descriptor.pop("snapshot", None)
            return encode()

    def _snapshot(self, resource: _Resource, fields: Iterable[str]) -> Dict[str, Any]:
        if resource.kind != "node":
            result: Dict[str, Any] = {key: value for key, value in resource.latest.items() if key in fields}
            if "render" in fields and resource.kind == "env":
                result["render"] = resource.obj.render()
            return result
        obj, result = resource.obj, {}
        getters = {"context": ("context_space", "get_context"), "observation": ("observation_space", "get_observation"),
                   "reward": ("has_reward", "get_reward"), "terminated": ("has_termination_signal", "get_termination"),
                   "truncated": ("has_truncation_signal", "get_truncation"), "info": (None, "get_info"),
                   "render": ("can_render", "render")}
        for name in fields:
            flag, getter = getters[name]
            enabled = flag is None or (getattr(obj, flag, None) is not None if flag.endswith("_space") else bool(getattr(obj, flag, False)))
            if enabled:
                result[name] = getattr(obj, getter)()
        return result

    def _publish(self, domain: _Domain, operation_id: str, kind: str,
                 env_resource: Optional[_Resource] = None, captured: Optional[SnapshotCaptures] = None) -> None:
        domain.sequence += 1
        timestamp = time.time_ns()
        if captured is None:
            captured = {}
        for (_, subscription_id), (session, resource, fields) in list(domain.subscriptions.items()):
            # A client-composed env need not match any registered server Env.
            if resource.kind == "env" and resource is not env_resource:
                continue
            event: WireMessage = {"version": PROTOCOL_VERSION, "type": "snapshot", "session_id": session.id,
                     "subscription_id": subscription_id, "resource_id": resource.id,
                     "operation_id": operation_id, "kind": kind, "sequence": domain.sequence,
                     "timestamp_ns": timestamp, "descriptor_revision": domain.revision}
            try:
                event["data"] = self._capture_snapshot(resource, fields, captured)
                event["descriptor"] = self._descriptor(resource)
                session.enqueue(session.codec.encode(event), subscription_id)
            except Exception as exc:
                # Errors use the reliable response queue, not the lossy snapshots.
                event.pop("data", None)
                event.pop("descriptor", None)
                event.update(type="subscription_error", error={"code": "snapshot_error", "message": str(exc)[:512]})
                try:
                    session.enqueue(session.codec.encode(event))
                except CodecError:
                    session.close()
                domain.subscriptions.pop((session.id, subscription_id), None)

    def _release(self, domain: _Domain) -> None:
        if domain.operation is not None:
            domain.faulted = True
        domain.operation = None
        domain.owner = domain.mode = None

    def _detach(self, domain: _Domain, session: _Session) -> None:
        if domain.owner is session:
            self._release(domain)
            domain.last_activity = time.monotonic()
        for key in list(domain.subscriptions):
            if key[0] == session.id:
                domain.subscriptions.pop(key)
                domain.last_activity = time.monotonic()

    def _dispose(self, domain: _Domain) -> None:
        if not domain.instantiated:
            return
        try:
            self._dispose_resources(domain)
        finally:
            with self._lock:
                self._world_owners = {key: owner for key, owner in self._world_owners.items()
                                      if owner is not domain}

    def _dispose_resources(self, domain: _Domain) -> None:
        nodes = [r.obj for r in domain.resources if r.kind == "node"]
        child_ids = {id(child) for node in nodes for child in getattr(node, "nodes", ())}
        errors: List[Exception] = []
        environments = [r.obj for r in domain.resources if r.kind == "env"]
        covered_nodes: Set[int] = set()

        def cover(node: WorldNode) -> None:
            if id(node) in covered_nodes:
                return
            covered_nodes.add(id(node))
            for child in getattr(node, "nodes", ()):
                cover(child)

        # Prefer the outermost registered wrapper chain, allowing its close hook
        # to release wrapper-owned resources before the underlying env/world.
        wrapped_ids: Set[int] = set()
        for env in environments:
            current = getattr(env, "env", None)
            while isinstance(current, Env) and id(current) not in wrapped_ids:
                wrapped_ids.add(id(current))
                current = getattr(current, "env", None)
        roots = [env for env in environments if id(env) not in wrapped_ids]
        objects = roots[:1]
        world_covered = False
        if roots:
            base = roots[0].unwrapped
            if isinstance(base, WorldEnv):
                cover(base.node)
                world_covered = True
        objects += [node for node in nodes if id(node) not in child_ids and id(node) not in covered_nodes]
        if not world_covered:
            objects += [r.obj for r in domain.resources if r.kind == "world"]
        for obj in objects:
            try:
                obj.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sessions = list(self._sessions)
        self._reaper_stop.set()
        if self._reaper is not None:
            self._reaper.join()
        if self._websocket is not None:
            self._websocket.shutdown()
        for session in sessions:
            session.close()
        for session in sessions:
            if current_thread() is not session.reader:
                session.reader.join()
            if current_thread() is not session.writer:
                session.writer.join()
        errors: List[Exception] = []
        for domain in self._domains.values():
            try:
                domain.executor.submit(self._dispose, domain).result()
            except Exception as exc:
                errors.append(exc)
            finally:
                domain.executor.shutdown(wait=True)
                domain.factory = None
        if self._websocket_thread is not None:
            self._websocket_thread.join()
        self._resources.clear()
        self._objects.clear()
        self._domains.clear()
        self._world_owners.clear()
        if errors:
            raise errors[0]

    def __enter__(self) -> RemoteServer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
