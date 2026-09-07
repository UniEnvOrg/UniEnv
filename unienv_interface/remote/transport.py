"""Ordered, reliable binary message transports. WebSocket is an optional extra."""

from __future__ import annotations

from collections import deque
from threading import Condition
from typing import Deque, Protocol, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from websockets.sync.connection import Connection

from .codec import DEFAULT_MAX_MESSAGE_SIZE


class TransportClosed(ConnectionError):
    pass


class MessageTransport(Protocol):
    """One reader and one writer may run concurrently; close unblocks both.

    Implementations preserve complete message boundaries and order. They must
    bound buffering and raise ConnectionError/OSError on transport failure.
    """

    def send(self, payload: bytes) -> None: ...
    def recv(self) -> bytes: ...
    def close(self) -> None: ...


class _Channel:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.messages: Deque[bytes] = deque()
        self.closed = False
        self.condition = Condition()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.messages.clear()
            self.condition.notify_all()


class MemoryTransport:
    """Bounded, byte-copying transport for tests and embedded deployments."""

    def __init__(self, incoming: _Channel, outgoing: _Channel) -> None:
        self._incoming, self._outgoing = incoming, outgoing

    @classmethod
    def pair(cls, capacity: int = 16) -> Tuple[MemoryTransport, MemoryTransport]:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        left, right = _Channel(capacity), _Channel(capacity)
        return cls(left, right), cls(right, left)

    def send(self, payload: bytes) -> None:
        channel = self._outgoing
        with channel.condition:
            channel.condition.wait_for(lambda: channel.closed or len(channel.messages) < channel.capacity)
            if channel.closed:
                raise TransportClosed("Connection closed")
            channel.messages.append(bytes(payload))
            channel.condition.notify_all()

    def recv(self) -> bytes:
        channel = self._incoming
        with channel.condition:
            channel.condition.wait_for(lambda: channel.closed or channel.messages)
            if channel.closed:
                raise TransportClosed("Connection closed")
            message = channel.messages.popleft()
            channel.condition.notify_all()
            return message

    def close(self) -> None:
        self._incoming.close()
        self._outgoing.close()


class WebSocketTransport:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    @classmethod
    def connect(cls, uri: str, *, max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
                **kwargs: object) -> WebSocketTransport:
        from websockets.sync.client import connect
        return cls(connect(uri, max_size=max_message_size, max_queue=16, compression=None, proxy=None, **kwargs))

    def send(self, payload: bytes) -> None:
        from websockets.exceptions import ConnectionClosed
        try:
            self.connection.send(payload)
        except ConnectionClosed as exc:
            raise TransportClosed(str(exc)) from exc

    def recv(self) -> bytes:
        from websockets.exceptions import ConnectionClosed
        try:
            payload = self.connection.recv()
        except ConnectionClosed as exc:
            raise TransportClosed(str(exc)) from exc
        if not isinstance(payload, bytes):
            raise TransportClosed("UniEnv requires binary messages")
        return payload

    def close(self) -> None:
        self.connection.close()
