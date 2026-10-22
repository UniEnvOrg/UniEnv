"""Lightweight RPC for UniEnv environments, worlds, nodes and data transformations.

Host objects with :class:`RemoteServer` and use them through
:class:`RemoteClient` proxies that implement the normal ``Env`` / ``World`` /
``WorldNode`` / ``DataTransformation`` interfaces. Transports: TCP, Unix sockets, WebSocket (optional
``websockets`` dependency) and in-memory. See ``docs/guides/remote-execution.md``.
"""

from typing import List

from .client import RemoteClient, Subscription, SubscriptionEvent
from .errors import RemoteError
from .protocol import PROTOCOL_VERSION
from .proxies import RemoteDataTransformation, RemoteEnv, RemoteWorld, RemoteWorldNode
from .server import RemoteServer
from .transport import MemoryTransport, StreamTransport, Transport, WebSocketTransport
from .wire import WireError

__all__: List[str] = [
    "RemoteServer", "RemoteClient", "RemoteEnv", "RemoteWorld", "RemoteWorldNode", "RemoteDataTransformation", "Subscription",
    "SubscriptionEvent", "RemoteError", "WireError", "PROTOCOL_VERSION", "Transport", "StreamTransport",
    "MemoryTransport", "WebSocketTransport",
]
