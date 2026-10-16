"""Frame transports: TCP and Unix stream sockets, WebSocket, and in-memory pairs.

Addresses are URLs: ``tcp://host:port``, ``unix:///path/to.sock`` and
``ws://host:port``. Port ``0`` picks a free port. Every transport sends one
frame per :meth:`Transport.send` (thread-safe) and returns
``(json, buffers)`` from :meth:`Transport.recv` (one reader thread).
"""

from __future__ import annotations

import os
import socket
import stat
import struct
from queue import SimpleQueue
from threading import Lock, Thread
from typing import Any, Callable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import numpy as np

from .wire import DEFAULT_MAX_FRAME_SIZE, WireError, read_header, split

__all__ = ["Transport", "StreamTransport", "MemoryTransport", "WebSocketTransport", "connect", "listen",
           "parse_address"]

_MAX_IOV = 512
Frame = Tuple[Any, List[Any]]


class Transport:
    """A reliable, ordered connection carrying whole frames."""

    max_frame_size: int = DEFAULT_MAX_FRAME_SIZE

    def send(self, parts: Sequence[Any], size: int) -> None:
        """Send the frame ``parts`` (``size`` bytes in total). Thread-safe."""
        raise NotImplementedError

    def recv(self) -> Frame:
        """Block for the next frame; raise ``ConnectionError`` once closed."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the connection, unblocking ``recv`` (idempotent)."""
        raise NotImplementedError


class StreamTransport(Transport):
    """Frames over a connected stream socket; each frame goes out in one ``sendmsg``."""

    def __init__(self, sock: socket.socket, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> None:
        self.sock, self.max_frame_size = sock, max_frame_size
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._file = sock.makefile("rb", buffering=1 << 16)
        self._send_lock = Lock()
        self._closed = False

    def send(self, parts: Sequence[Any], size: int) -> None:
        if len(parts) > _MAX_IOV:
            parts = [b"".join(parts)]
        with self._send_lock:
            if self._closed:
                raise ConnectionError("Connection closed")
            sent = self.sock.sendmsg(parts)
            if sent < size:  # partial write: send the rest part by part
                for part in parts:
                    view = memoryview(part).cast("B")
                    if sent >= len(view):
                        sent -= len(view)
                        continue
                    self.sock.sendall(view[sent:])
                    sent = 0

    def _read_into(self, buffer: Any) -> None:
        view = memoryview(buffer).cast("B")
        while view:
            count = self._file.readinto(view)
            if not count:
                raise ConnectionError("Connection closed by peer")
            view = view[count:]

    def _read(self, size: int) -> bytes:
        data = self._file.read(size)
        if len(data) < size:
            raise ConnectionError("Connection closed by peer")
        return data

    def recv(self) -> Frame:
        try:
            count, length = read_header(self._read(8), self.max_frame_size)
            sizes = struct.unpack(f"<{count}Q", self._read(8 * count)) if count else ()
            if length + sum(sizes) > self.max_frame_size:
                raise WireError(f"Frame exceeds the {self.max_frame_size}-byte limit")
            document = self._read(length)
            buffers = []
            for size in sizes:
                buffer = np.empty(size, np.uint8)  # writable; not zero-filled
                self._read_into(buffer)
                buffers.append(buffer)
            return document, buffers
        except (OSError, ValueError) as exc:
            self._file.close()
            if self._closed:
                raise ConnectionError("Connection closed") from None
            if isinstance(exc, (ConnectionError, WireError)):
                raise
            raise ConnectionError(str(exc)) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class _Queue:
    def __init__(self) -> None:
        self.items: SimpleQueue = SimpleQueue()
        self.closed = False


class MemoryTransport(Transport):
    """In-process pair (tests); frames are copied, as on a real connection."""

    def __init__(self, incoming: _Queue, outgoing: _Queue, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> None:
        self._incoming, self._outgoing, self.max_frame_size = incoming, outgoing, max_frame_size

    @classmethod
    def pair(cls, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> Tuple[MemoryTransport, MemoryTransport]:
        left, right = _Queue(), _Queue()
        return cls(left, right, max_frame_size), cls(right, left, max_frame_size)

    def send(self, parts: Sequence[Any], size: int) -> None:
        if self._outgoing.closed:
            raise ConnectionError("Connection closed")
        self._outgoing.items.put(bytearray(b"".join(parts)))

    def recv(self) -> Frame:
        data = self._incoming.items.get()
        if data is None:
            raise ConnectionError("Connection closed")
        return split(data, self.max_frame_size)

    def close(self) -> None:
        for queue in (self._incoming, self._outgoing):
            if not queue.closed:
                queue.closed = True
                queue.items.put(None)


class WebSocketTransport(Transport):
    """One frame per binary WebSocket message (requires ``websockets``)."""

    def __init__(self, connection: Any, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> None:
        self.connection, self.max_frame_size = connection, max_frame_size
        self._send_lock = Lock()

    def send(self, parts: Sequence[Any], size: int) -> None:
        from websockets.exceptions import ConnectionClosed
        with self._send_lock:
            try:
                self.connection.send(b"".join(parts))
            except ConnectionClosed as exc:
                raise ConnectionError(str(exc)) from None

    def recv(self) -> Frame:
        from websockets.exceptions import ConnectionClosed
        try:
            message = self.connection.recv()
        except ConnectionClosed as exc:
            raise ConnectionError(str(exc)) from None
        if not isinstance(message, bytes):
            raise WireError("Expected a binary WebSocket message")
        return split(bytearray(message), self.max_frame_size)

    def close(self) -> None:
        self.connection.close()


def parse_address(address: str) -> Tuple[str, str, Optional[int]]:
    """Return ``(scheme, host or path, port)`` for a transport URL."""
    parts = urlsplit(address)
    if parts.scheme == "unix":
        path = parts.netloc + parts.path
        if not path:
            raise ValueError(f"Missing socket path in {address!r}")
        return "unix", path, None
    if parts.scheme in ("tcp", "ws"):
        if parts.hostname is None or parts.port is None:
            raise ValueError(f"Expected {parts.scheme}://host:port, got {address!r}")
        return parts.scheme, parts.hostname, parts.port
    raise ValueError(f"Unsupported address {address!r}; use tcp://, unix:// or ws://")


def _format(scheme: str, host: str, port: int) -> str:
    return f"{scheme}://{'[' + host + ']' if ':' in host else host}:{port}"


def connect(address: str, *, timeout: Optional[float] = 10.0,
            max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> Transport:
    """Open a client transport to ``address``."""
    scheme, host, port = parse_address(address)
    if scheme == "ws":
        from websockets.sync.client import connect as ws_connect
        connection = ws_connect(address, max_size=max_frame_size, compression=None, open_timeout=timeout,
                                close_timeout=1, proxy=None)
        return WebSocketTransport(connection, max_frame_size)
    if scheme == "tcp":
        sock = socket.create_connection((host, port), timeout=timeout)
    else:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(host)
        except OSError:
            sock.close()
            raise
    sock.settimeout(None)
    return StreamTransport(sock, max_frame_size)


class Listener:
    """A background accept loop; ``close`` stops accepting (open connections stay)."""

    address: str

    def close(self) -> None:
        raise NotImplementedError


class _SocketListener(Listener):
    def __init__(self, sock: socket.socket, address: str, on_connect: Callable[[Transport], None],
                 max_frame_size: int, path: Optional[str] = None) -> None:
        self.sock, self.address, self.path = sock, address, path
        self._on_connect, self._max_frame_size = on_connect, max_frame_size
        self._closed = False
        self._thread = Thread(target=self._accept, name="unienv-remote-accept", daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while True:
            try:
                connection, _ = self.sock.accept()
            except OSError:
                return
            if self._closed:
                connection.close()
                return
            self._on_connect(StreamTransport(connection, self._max_frame_size))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self._thread.join()
        if self.path is not None:
            try:
                os.unlink(self.path)
            except OSError:
                pass


class _WebSocketListener(Listener):
    def __init__(self, host: str, port: int, serve_connection: Callable[[Transport], None],
                 max_frame_size: int) -> None:
        from websockets.sync.server import serve

        def handler(connection: Any) -> None:
            serve_connection(WebSocketTransport(connection, max_frame_size))

        self._server = serve(handler, host, port, max_size=max_frame_size, compression=None, close_timeout=1)
        bound = self._server.socket.getsockname()
        self.address = _format("ws", host, bound[1])
        self._thread = Thread(target=self._server.serve_forever, name="unienv-remote-ws", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join()


def listen(address: str, *, on_connect: Callable[[Transport], None], serve_connection: Callable[[Transport], None],
           max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> Listener:
    """Start accepting connections on ``address``.

    Stream connections are handed to ``on_connect`` (which must not block);
    WebSocket connections run ``serve_connection`` on the library's handler thread.
    """
    scheme, host, port = parse_address(address)
    if scheme == "ws":
        return _WebSocketListener(host, port, serve_connection, max_frame_size)
    if scheme == "tcp":
        sock = socket.create_server((host, port))
        return _SocketListener(sock, _format("tcp", host, sock.getsockname()[1]), on_connect, max_frame_size)
    if os.path.exists(host) and stat.S_ISSOCK(os.stat(host).st_mode):
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(host)
        except OSError:
            os.unlink(host)  # stale socket file left by an earlier server
        else:
            raise OSError(f"Another server is listening on {host}")
        finally:
            probe.close()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(host)
        sock.listen()
    except OSError:
        sock.close()
        raise
    return _SocketListener(sock, "unix://" + host, on_connect, max_frame_size, path=host)
