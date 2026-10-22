"""Connect to a :class:`RemoteServer` and call hosted objects."""

from __future__ import annotations

from itertools import count
from threading import Condition, Lock, RLock, Thread, current_thread
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union

from unienv_interface.backends.numpy import NumpyComputeBackend

from . import wire
from .errors import RemoteError
from .protocol import ERROR, EVENT, PROTOCOL_VERSION
from .transport import Transport, connect

if TYPE_CHECKING:
    from unienv_interface.backends import ComputeBackend
    from .proxies import RemoteDataTransformation, RemoteEnv, RemoteWorld, RemoteWorldNode

__all__ = ["RemoteClient", "Subscription", "SubscriptionEvent"]

_LOST = ("Connection to the server was lost; a call in flight may or may not have run on the server")


class _Pending:
    __slots__ = ("lock", "message", "buffers", "error")

    def __init__(self) -> None:
        self.lock = Lock()
        self.lock.acquire()
        self.message: Any = None
        self.buffers: List[Any] = []
        self.error: Optional[BaseException] = None


class SubscriptionEvent(NamedTuple):
    """The result of a ``reset`` / ``reload`` / ``step`` of a subscribed env."""

    kind: str
    """Name of the method that produced ``value``."""
    sequence: int
    """Per-env call counter; gaps mean intermediate results were skipped."""
    value: Any
    """The method's return value, converted to the client's backend and device."""


class Subscription:
    """Iterator over the latest results of a hosted env (latest wins, never queued).

    Use it as a context manager or call :meth:`close`; ``next(timeout=...)``
    raises ``TimeoutError`` if no new result arrives in time.
    """

    def __init__(self, client: RemoteClient, name: str) -> None:
        self.client, self.name = client, name
        self._condition = Condition()
        self._latest: Optional[Tuple[Any, List[Any]]] = None
        self._error: Optional[BaseException] = None
        self.closed = False

    def _push(self, message: Any, buffers: List[Any]) -> None:
        with self._condition:
            self._latest = (message, buffers)
            self._condition.notify_all()

    def _fail(self, error: BaseException) -> None:
        with self._condition:
            self._error = error
            self._condition.notify_all()

    def next(self, timeout: Optional[float] = None) -> SubscriptionEvent:
        with self._condition:
            if not self._condition.wait_for(lambda: self.closed or self._error or self._latest, timeout):
                raise TimeoutError(f"No new result from {self.name!r} within {timeout} s")
            if self.closed:
                raise StopIteration
            if self._latest is None:
                raise self._error
            (message, buffers), self._latest = self._latest, None
        kind, sequence, value = wire.restore(message, buffers, self.client.backend, self.client.device)
        return SubscriptionEvent(kind, sequence, value)

    def __iter__(self) -> Iterator[SubscriptionEvent]:
        return self

    def __next__(self) -> SubscriptionEvent:
        return self.next()

    def close(self) -> None:
        with self._condition:
            if self.closed:
                return
            self.closed = True
            self._condition.notify_all()
        self.client._unsubscribe(self)

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class RemoteClient:
    """A connection to a :class:`RemoteServer`. Thread-safe: calls may be concurrent.

    Results are converted to ``backend`` arrays on ``device``. A remote exception
    is re-raised as :class:`RemoteError`; a lost connection raises
    ``ConnectionError``, in which case a call that was in flight may or may not
    have run on the server. ``timeout`` (seconds, ``None`` = wait forever) bounds
    each call; a timed-out call keeps running on the server.
    """

    def __init__(self, transport: Transport, *, backend: ComputeBackend = NumpyComputeBackend,
                 device: Optional[Any] = None, timeout: Optional[float] = None,
                 compress: Optional[Dict[str, str]] = None) -> None:
        if compress:
            from .compression import parse_spec
            parse_spec(compress)  # fail early on a bad spec
        self.transport, self.backend, self.device, self.timeout = transport, backend, device, timeout
        self.closed = False
        self._ids = count(1)
        self._pending: Dict[int, _Pending] = {}
        self._subscriptions: Dict[str, List[Subscription]] = {}
        self._proxies: Dict[str, Any] = {}
        self._lock = Lock()
        self._subscribe_lock = RLock()
        self._error: BaseException = ConnectionError("Client is closed")
        self._reader = Thread(target=self._read, name="unienv-remote-client", daemon=True)
        self._reader.start()
        try:
            hello = self._request("", "hello", [], {"version": PROTOCOL_VERSION, "compress": compress},
                                  timeout=30.0 if timeout is None else timeout)
        except BaseException:
            self.close()
            raise
        self.resources: Dict[str, Dict[str, Any]] = hello["resources"]
        """Descriptor of every hosted resource, by name."""
        for descriptor in self.resources.values():
            version, spaces = descriptor.pop("spaces")
            descriptor.update(version=version, spaces=spaces)

    @classmethod
    def connect(cls, address: str, *, timeout: Optional[float] = None, connect_timeout: float = 10.0,
                backend: ComputeBackend = NumpyComputeBackend, device: Optional[Any] = None,
                compress: Optional[Dict[str, str]] = None,
                max_frame_size: int = wire.DEFAULT_MAX_FRAME_SIZE) -> RemoteClient:
        """Connect to ``tcp://host:port``, ``unix:///path`` or ``ws://host:port``.

        ``compress`` opts into lossy compression of server-to-client data, e.g.
        ``{"image": "jpeg:90"}`` JPEG-encodes uint8 image leaves (spaces whose
        ``metadata["kind"] == "image"``) at quality 90.
        """
        transport = connect(address, timeout=connect_timeout, max_frame_size=max_frame_size)
        return cls(transport, backend=backend, device=device, timeout=timeout, compress=compress)

    # ------------------------------------------------------------------ requests

    def _read(self) -> None:
        try:
            while True:
                document, buffers = self.transport.recv()
                message = wire.parse(document)
                if type(message) is not list or len(message) < 3:
                    raise wire.WireError("Invalid message")
                if message[1] == EVENT:
                    for subscription in self._subscriptions.get(message[0], ()):
                        subscription._push(message[2], buffers)
                    continue
                pending = self._pending.pop(message[0], None)
                if pending is not None:
                    pending.message, pending.buffers = message, buffers
                    pending.lock.release()
        except wire.WireError as exc:
            self._shutdown(ConnectionError(f"Closed the connection after a malformed frame: {exc}"))
        except BaseException:
            self._shutdown(ConnectionError(_LOST))

    def _shutdown(self, error: BaseException) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed, self._error = True, error
            pending, self._pending = self._pending, {}
            subscriptions = [s for group in self._subscriptions.values() for s in group]
        for item in pending.values():
            item.error = error
            item.lock.release()
        for subscription in subscriptions:
            subscription._fail(error)
        self.transport.close()

    def _request(self, target: str, method: str, args: List[Any], kwargs: Dict[str, Any],
                 timeout: Union[float, None, str] = "default") -> Any:
        request_id = next(self._ids)
        parts, size = wire.encode([request_id, target, method, args, kwargs], max_size=self.transport.max_frame_size)
        pending = _Pending()
        with self._lock:
            if self.closed:
                raise ConnectionError(str(self._error))
            self._pending[request_id] = pending
        try:
            self.transport.send(parts, size)
        except (ConnectionError, OSError):
            self._shutdown(ConnectionError(_LOST))
        timeout = self.timeout if timeout == "default" else timeout
        if not pending.lock.acquire(timeout=-1 if timeout is None else timeout):
            self._pending.pop(request_id, None)
            raise TimeoutError(f"{target or 'server'}.{method} did not answer within {timeout} s "
                               "(the call may still run on the server)")
        if pending.error is not None:
            raise ConnectionError(str(pending.error))
        message = pending.message
        if message[1] == ERROR:
            error = message[2]
            if type(error) is not list or len(error) != 3 or not all(type(e) is str for e in error):
                raise ConnectionError("Malformed error response")
            raise RemoteError(*error)
        result = wire.restore(message[2:], pending.buffers, self.backend, self.device)
        if len(result) > 1:
            self._apply_updates(result[1])
        return result[0]

    def _apply_updates(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            for name, (version, spaces) in updates.items():
                descriptor = self.resources.get(name)
                if descriptor is None or version <= descriptor["version"]:
                    continue  # unknown, or an older update than one applied already
                descriptor.update(version=version, spaces=spaces)
                proxy = self._proxies.get(name)
                if proxy is not None:
                    proxy._set_spaces(spaces)

    def call(self, name: str, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call an allowed ``method`` of the hosted resource ``name``."""
        return self._request(name, method, list(args), kwargs)

    # ------------------------------------------------------------------ proxies

    def resource(self, name: str) -> Union[RemoteEnv, RemoteWorld, RemoteWorldNode, RemoteDataTransformation]:
        """The proxy of hosted resource ``name`` (one shared proxy per name)."""
        proxy = self._proxies.get(name)
        if proxy is None:
            from .proxies import RemoteDataTransformation, RemoteEnv, RemoteWorld, RemoteWorldNode
            if name not in self.resources:
                raise KeyError(f"The server hosts no resource named {name!r}; it has {sorted(self.resources)}")
            kind = self.resources[name]["kind"]
            proxy = {"env": RemoteEnv, "world": RemoteWorld, "node": RemoteWorldNode,
                     "transformation": RemoteDataTransformation}[kind](self, name)
            proxy = self._proxies.setdefault(name, proxy)
        return proxy

    def _typed(self, name: str, kind: str) -> Any:
        proxy = self.resource(name)
        if self.resources[name]["kind"] != kind:
            raise TypeError(f"{name!r} is a {self.resources[name]['kind']}, not a {kind}")
        return proxy

    def env(self, name: str) -> RemoteEnv:
        """Proxy of a hosted ``Env``."""
        return self._typed(name, "env")

    def world(self, name: str) -> RemoteWorld:
        """Proxy of a hosted ``World`` (e.g. ``"sim/world"`` for a ``WorldEnv`` registered recursively as ``"sim"``)."""
        return self._typed(name, "world")

    def node(self, name: str) -> RemoteWorldNode:
        """Proxy of a hosted ``WorldNode`` (e.g. ``"sim/node"``, ``"sim/node/camera"``)."""
        return self._typed(name, "node")

    def transformation(self, name: str) -> RemoteDataTransformation:
        """Proxy of a hosted ``DataTransformation``."""
        return self._typed(name, "transformation")

    # ------------------------------------------------------------------ observers

    def subscribe(self, name: str) -> Subscription:
        """Follow the ``reset`` / ``reload`` / ``step`` results of hosted env ``name``.

        Results of calls by any client are delivered, latest wins: a slow
        subscriber skips results rather than slowing the server down.
        """
        subscription = Subscription(self, name)
        with self._subscribe_lock:
            with self._lock:
                group = self._subscriptions.get(name, [])
                self._subscriptions[name] = group + [subscription]
            if not group:
                try:
                    self._request("", "subscribe", [name], {})
                except BaseException:
                    with self._lock:
                        self._subscriptions.pop(name, None)
                    raise
        return subscription

    def _unsubscribe(self, subscription: Subscription) -> None:
        with self._subscribe_lock:
            with self._lock:
                group = [s for s in self._subscriptions.get(subscription.name, ()) if s is not subscription]
                if group:
                    self._subscriptions[subscription.name] = group
                else:
                    self._subscriptions.pop(subscription.name, None)
            if not group and not self.closed:
                try:
                    self._request("", "unsubscribe", [subscription.name], {})
                except ConnectionError:
                    pass

    # ------------------------------------------------------------------ lifetime

    def close(self) -> None:
        """Close the connection (hosted objects stay open on the server)."""
        self._shutdown(ConnectionError("Client is closed"))
        if current_thread() is not self._reader:
            self._reader.join()

    def __enter__(self) -> RemoteClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
