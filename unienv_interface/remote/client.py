"""Synchronous requests with an independent response/event receiver."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from threading import Condition, Event, Lock, RLock, Thread, current_thread
from queue import Queue, Full
from uuid import uuid4
import weakref
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TYPE_CHECKING, TypedDict, Union

from unienv_interface.backends.numpy import NumpyComputeBackend

from .codec import Codec, PROTOCOL_VERSION, DEFAULT_MAX_MESSAGE_SIZE, map_arrays
from .errors import RemoteError, UncertainOutcomeError
from .protocol import METHODS
from .transport import WebSocketTransport

if TYPE_CHECKING:
    from unienv_interface.backends import ComputeBackend
    from .protocol import BoundarySnapshot, Descriptor, ExecutionMode, OperationKind, WireMessage
    from .proxies import _Proxy, RemoteEnv, RemoteWorld, RemoteWorldNode
    from .transport import MessageTransport

    class _Pending(TypedDict, total=False):
        event: Event
        response: WireMessage
        failure: BaseException


class Subscription:
    """Closeable latest-snapshot iterator. next(timeout=...) is also supported."""

    def __init__(self, client: RemoteClient, resource_id: str, subscription_id: str) -> None:
        self.client, self.resource_id, self.id = client, resource_id, subscription_id
        self._condition = Condition()
        self._latest: Optional[WireMessage] = None
        self._error: Optional[BaseException] = None
        self._closed = False

    def _push(self, event: WireMessage) -> None:
        with self._condition:
            if self._closed:
                return
            if event["type"] == "subscription_error":
                self._error = RemoteError(**event["error"])
            else:
                self._latest = event
            self._condition.notify_all()

    def _finish(self, error: Optional[BaseException] = None) -> None:
        with self._condition:
            self._closed = True
            self._error = error
            self._latest = None
            self._condition.notify_all()

    def next(self, timeout: Optional[float] = None) -> WireMessage:
        with self._condition:
            if not self._condition.wait_for(lambda: self._closed or self._error or self._latest is not None, timeout):
                raise TimeoutError("No snapshot arrived before the deadline")
            if self._error is not None:
                raise self._error
            if self._closed:
                raise StopIteration
            result, self._latest = self._latest, None
        self.client._update([result["descriptor"]])
        result["data"] = map_arrays(result["data"], self.client.backend, self.client.device)
        return result

    def __iter__(self) -> Subscription:
        return self

    def __next__(self) -> WireMessage:
        return self.next()

    def close(self) -> None:
        if self._closed:
            return
        self._finish()
        self.client._subscriptions.pop(self.id, None)
        if not self.client.closed:
            self.client._request("unsubscribe", self.resource_id, subscription_id=self.id)

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class RemoteClient:
    """A connection, descriptor cache, and shared control session.

    Requests are serialized, including entire component operation contexts.
    A timeout closes the connection: late replies can never be confused with a
    later request. Mutations aren't retried and may have executed remotely.
    """

    def __init__(self, transport: MessageTransport, *, backend: ComputeBackend = NumpyComputeBackend,
                 device: Optional[object] = None, timeout: Optional[float] = 30.0,
                 max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE) -> None:
        self.transport, self.backend, self.device = transport, backend, device
        self.timeout = timeout
        self.codec = Codec(max_message_size)
        self.session_id: Optional[str] = None
        self.closed = False
        self.descriptors: Dict[str, Descriptor] = {}
        self._proxies: weakref.WeakValueDictionary[str, _Proxy] = weakref.WeakValueDictionary()
        self._subscriptions: Dict[str, Subscription] = {}
        self._operations: Dict[str, str] = {}
        self._snapshots: Dict[str, BoundarySnapshot] = {}
        self._boundary_versions: Dict[str, Tuple[int, int]] = {}
        self._request_lock = RLock()
        self._state_lock = Lock()
        self._pending: Dict[str, _Pending] = {}
        self._outgoing: Queue[Optional[bytes]] = Queue(maxsize=1)
        self._receiver = Thread(target=self._receive, name="unienv-client", daemon=True)
        self._sender = Thread(target=self._send, name="unienv-client-send", daemon=True)
        self._sender.start()
        self._receiver.start()
        try:
            hello = self._request("hello", max_message_size=self.codec.max_message_size)
            self.session_id = hello["session_id"]
            self.codec.max_message_size = min(self.codec.max_message_size, hello["max_message_size"])
            self.discover()
        except Exception:
            self.close()
            raise

    @classmethod
    def connect(cls, uri: str, **options: object) -> RemoteClient:
        transport = WebSocketTransport.connect(uri, max_message_size=options.get("max_message_size", DEFAULT_MAX_MESSAGE_SIZE), close_timeout=1)
        return cls(transport, **options)

    def _receive(self) -> None:
        failure: Optional[Exception] = None
        try:
            while not self.closed:
                message = self.codec.decode(self.transport.recv())
                if not isinstance(message, dict) or message.get("version") != PROTOCOL_VERSION:
                    raise RemoteError("protocol_error", "Invalid response version or envelope")
                if self.session_id is not None and message.get("session_id") != self.session_id:
                    raise RemoteError("protocol_error", "Response has the wrong session ID")
                if message.get("type") in {"snapshot", "subscription_error"}:
                    with self._state_lock:
                        resource_id = message["resource_id"]
                        if message["type"] == "subscription_error":
                            self._drop_snapshots(resource_id)
                        elif resource_id in self.descriptors:
                            self._observe_boundary(self._domain_id(resource_id),
                                                   (message["descriptor_revision"], message["sequence"]))
                    subscription = self._subscriptions.get(message.get("subscription_id"))
                    if subscription is not None:
                        subscription._push(message)
                elif message.get("type") in {"response", "error"}:
                    with self._state_lock:
                        pending = self._pending.get(message.get("request_id"))
                        if pending is not None:
                            pending["response"] = message
                            pending["event"].set()
                else:
                    raise RemoteError("protocol_error", "Unknown server message type")
        except Exception as exc:
            failure = exc
        finally:
            self._disconnect(failure or ConnectionError("Client closed"))

    def _send(self) -> None:
        try:
            while not self.closed:
                payload = self._outgoing.get()
                if payload is None or self.closed:
                    return
                self.transport.send(payload)
        except Exception as exc:
            self._disconnect(exc)

    def _disconnect(self, failure: BaseException) -> None:
        with self._state_lock:
            self.closed = True
            self._snapshots.clear()
            self._boundary_versions.clear()
            for pending in self._pending.values():
                pending.setdefault("failure", failure)
                pending["event"].set()
        for subscription in list(self._subscriptions.values()):
            subscription._finish(failure)
        try:
            self._outgoing.put_nowait(None)
        except Full:
            pass
        self.transport.close()

    def _request(self, op: str, resource_id: Optional[str] = None, *, mutation: bool = False,
                 **parameters: Any) -> Any:
        with self._request_lock:
            if self.closed:
                raise ConnectionError("Client is closed")
            request_id = uuid4().hex
            message: WireMessage = {"version": PROTOCOL_VERSION, "type": "request", "request_id": request_id,
                       "session_id": self.session_id, "op": op, "resource_id": resource_id, **parameters}
            payload = self.codec.encode(message)
            pending: _Pending = {"event": Event()}
            with self._state_lock:
                if self.closed:
                    raise ConnectionError("Client is closed")
                descriptor = self.descriptors.get(resource_id, {})
                changes_state = (op == "call" and METHODS.get(descriptor.get("kind"), {}).get(
                    parameters.get("method"), {}).get("mutation", False))
                if resource_id is not None and (mutation or changes_state or op in {"begin", "abort", "release", "acquire"}):
                    self._drop_snapshots(resource_id)
                self._pending[request_id] = pending
            try:
                try:
                    self._outgoing.put_nowait(payload)
                    if not pending["event"].wait(self.timeout):
                        raise TimeoutError("Remote request timed out")
                    # A successfully received reply remains authoritative if a
                    # disconnect follows it immediately.
                    if "response" not in pending:
                        raise pending.get("failure", ConnectionError("Connection lost"))
                except Exception as exc:
                    self._disconnect(exc)
                    if mutation:
                        raise UncertainOutcomeError() from exc
                    raise
                response = pending["response"]
                if response["type"] == "error":
                    with self._state_lock:
                        self._drop_snapshots(resource_id)
                    # Only the server knows whether invocation actually started.
                    raise RemoteError(**response["error"])
                return response["result"]
            finally:
                with self._state_lock:
                    self._pending.pop(request_id, None)

    def _update(self, descriptors: Iterable[Descriptor]) -> None:
        with self._state_lock:
            accepted: List[Descriptor] = []
            for descriptor in descriptors:
                resource_id = descriptor["id"]
                domain_id = descriptor.get("domain_id") or descriptor["world_id"] or resource_id
                version = (descriptor["revision"], descriptor.get("sequence", 0))
                if not self._observe_boundary(domain_id, version):
                    continue
                schema = {key: value for key, value in descriptor.items() if key != "snapshot"}
                self.descriptors[resource_id] = schema
                proxy = self._proxies.get(resource_id)
                if proxy is not None:
                    proxy._refresh(schema)
                accepted.append(descriptor)
            if not self.closed:
                for descriptor in accepted:
                    snapshot = descriptor.get("snapshot")
                    if snapshot is not None:
                        self._snapshots[descriptor["id"]] = snapshot

    def _drop_snapshots(self, resource_id: Optional[str] = None) -> None:
        """Invalidate a domain while holding _state_lock."""
        if resource_id is None:
            self._snapshots.clear()
            return
        domain_id = self._domain_id(resource_id) if resource_id in self.descriptors else resource_id
        self._snapshots = {key: value for key, value in self._snapshots.items()
                           if self._domain_id(key) != domain_id}

    def _observe_boundary(self, domain_id: str, version: Tuple[int, int]) -> bool:
        """Track response/event ordering while holding _state_lock."""
        previous = self._boundary_versions.get(domain_id)
        if previous is not None and previous > version:
            return False
        if previous != version:
            self._drop_snapshots(domain_id)
            self._boundary_versions[domain_id] = version
        return True

    def _read_snapshot(self, resource_id: str, field: str, method: str) -> Any:
        with self._request_lock:
            with self._state_lock:
                if self.closed:
                    raise ConnectionError("Client is closed")
                snapshot = self._snapshots.get(resource_id)
                if (self._domain_id(resource_id) not in self._operations and snapshot is not None
                        and field in snapshot["data"]):
                    return map_arrays(deepcopy(snapshot["data"][field]), self.backend, self.device)
            return self.call(resource_id, method)

    def discover(self) -> List[Descriptor]:
        descriptors = self._request("discover")
        self._update(descriptors)
        return descriptors

    def describe(self, resource_id: str) -> Descriptor:
        self._update(self._request("describe", resource_id))
        return self.descriptors[resource_id]

    def resource(self, resource_id: str) -> Union[RemoteEnv, RemoteWorld, RemoteWorldNode]:
        from .proxies import RemoteEnv, RemoteWorld, RemoteWorldNode
        if resource_id in self._proxies:
            proxy = self._proxies[resource_id]
            if not proxy._closed:
                return proxy
        if resource_id not in self.descriptors:
            self.describe(resource_id)
        cls = {"env": RemoteEnv, "world": RemoteWorld, "node": RemoteWorldNode}[self.descriptors[resource_id]["kind"]]
        proxy = cls(self, resource_id)
        self._proxies[resource_id] = proxy
        return proxy

    def env(self, resource_id: str) -> RemoteEnv:
        proxy = self.resource(resource_id)
        if self.descriptors[resource_id]["kind"] != "env":
            raise TypeError("Resource is not an environment")
        return proxy

    def world(self, resource_id: str) -> RemoteWorld:
        proxy = self.resource(resource_id)
        if self.descriptors[resource_id]["kind"] != "world":
            raise TypeError("Resource is not a world")
        return proxy

    def node(self, resource_id: str) -> RemoteWorldNode:
        proxy = self.resource(resource_id)
        if self.descriptors[resource_id]["kind"] != "node":
            raise TypeError("Resource is not a node")
        return proxy

    def _domain_id(self, resource_id: str) -> str:
        descriptor = self.descriptors[resource_id]
        return descriptor.get("domain_id") or descriptor["world_id"] or resource_id

    def acquire(self, resource_id: str, mode: Optional[ExecutionMode] = None) -> None:
        mode = mode or ("env" if self.descriptors[resource_id]["kind"] == "env" else "components")
        self._request("acquire", resource_id, mode=mode, mutation=True)

    def release(self, resource_id: str, *, close_subscriptions: bool = False) -> None:
        """Release shared world control, retaining subscriptions by default.

        Set close_subscriptions=True to also close this client's domain streams.
        """
        if close_subscriptions:
            domain_id = self._domain_id(resource_id)
            for subscription in list(self._subscriptions.values()):
                if self._domain_id(subscription.resource_id) == domain_id:
                    subscription.close()
        if not self.closed:
            self._request("release", resource_id, mutation=True)

    @contextmanager
    def operation(self, resource_id: str, kind: OperationKind = "step") -> Iterator[str]:
        """Mark one client-coordinated reset/reload/step, without rollback.

        Use a world or node ID. Every mutating component call must belong to an
        operation. The context publishes snapshots only after successful exit.
        """
        with self._request_lock:
            domain_id = self._domain_id(resource_id)
            if domain_id in self._operations:
                raise RuntimeError("Nested component operations are not supported")
            operation_id = uuid4().hex
            self._request("begin", domain_id, kind=kind, operation_id=operation_id, mutation=True)
            self._operations[domain_id] = operation_id
            try:
                yield operation_id
                descriptors = self._request("complete", domain_id, operation_id=operation_id, mutation=True)
                self._update(descriptors)
            except BaseException:
                if not self.closed:
                    try:
                        self._request("abort", domain_id, operation_id=operation_id, mutation=True)
                    except Exception:
                        pass
                raise
            finally:
                self._operations.pop(domain_id, None)

    def call(self, resource_id: str, method: str, *args: Any, **kwargs: Any) -> Any:
        with self._request_lock:
            descriptor = self.descriptors[resource_id]
            mutation = METHODS[descriptor["kind"]].get(method, {}).get("mutation", False)
            operation_id = self._operations.get(self._domain_id(resource_id))
            if descriptor["kind"] == "env":
                operation_id = uuid4().hex
            result = self._request("call", resource_id, method=method, args=args, kwargs=kwargs,
                                   operation_id=operation_id, mutation=mutation)
            self._update(result["descriptors"])
            return map_arrays(result["value"], self.backend, self.device)

    def subscribe(self, resource_id: str, fields: Optional[Iterable[str]] = None) -> Subscription:
        if resource_id not in self.descriptors:
            self.describe(resource_id)
        subscription = Subscription(self, resource_id, uuid4().hex)
        self._subscriptions[subscription.id] = subscription
        try:
            options = {} if fields is None else {"fields": list(fields)}
            self._request("subscribe", resource_id, subscription_id=subscription.id, **options)
        except Exception:
            self._subscriptions.pop(subscription.id, None)
            subscription._finish()
            raise
        return subscription

    def close(self) -> None:
        self._disconnect(ConnectionError("Client closed"))
        if current_thread() is not self._receiver:
            self._receiver.join()
        if current_thread() is not self._sender:
            self._sender.join()

    def __enter__(self) -> RemoteClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
