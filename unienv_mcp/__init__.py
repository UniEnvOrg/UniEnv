"""UniEnv environments and policies as MCP servers, for LLM agents.

Needs the ``mcp`` extra: ``pip install "unienv[mcp]"``. See :class:`UniEnvMCPServer`, and
``python -m unienv_mcp --help`` (or the ``unienv-mcp`` command) to serve from the command line.
"""

from .content import ContentOptions
from .server import UniEnvMCPServer
from .bridge import bridge

__all__ = ["UniEnvMCPServer", "ContentOptions", "bridge"]
