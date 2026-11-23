"""Serve what a UniEnv ``RemoteServer`` hosts to MCP clients."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from unienv_interface.remote import RemoteClient

from .server import UniEnvMCPServer

__all__ = ["bridge"]


def bridge(address: str, *, names: Optional[Sequence[str]] = None, server: Optional[UniEnvMCPServer] = None,
           physical: Sequence[str] = (), **client_options: Any) -> UniEnvMCPServer:
    """An MCP server exposing the environments and policies hosted at ``address`` (``tcp://``, ``unix://``, ``ws://``).

    The simulator or robot stack keeps running in its own process (and Python environment); this
    process only needs ``unienv[mcp]``. Remote environments are shared instances, so every MCP
    client acts on the same one. Objects the remote server builds on demand (``register_factory``)
    are opened here, and stay open while the bridge runs. ``names`` limits what is exposed;
    ``physical`` lists environments that are real systems. Worlds, nodes and transformations are
    not exposed.
    """
    client = RemoteClient.connect(address, **client_options)
    server = server or UniEnvMCPServer()
    for name, descriptor in sorted(client.resources.items()):
        if names is not None and name not in names:
            continue
        kind = descriptor.get("kind")
        if descriptor.get("factory") and kind is None:
            client.resource(name)  # opens it, which reveals its kind
            kind = client.resources[name]["kind"]
        tool_name = name.replace("/", "_")
        if kind == "env":
            server.add_env(tool_name, client.env(name), physical=tool_name in physical or name in physical)
        elif kind == "policy":
            server.add_policy(tool_name, client.policy(name))
    server._bridge_client = client  # keep the connection open as long as the server
    return server
