"""``unienv-mcp``: serve UniEnv environments and policies over MCP.

    unienv-mcp serve mypkg.envs:make_env --name sim --policy mypkg.policies:make_policy=pick
    unienv-mcp serve mypkg.mcp:server                  # a UniEnvMCPServer configured in Python
    unienv-mcp bridge tcp://sim-box:7000               # what a RemoteServer hosts
    ... --transport http --host 127.0.0.1 --port 8000  # Streamable HTTP instead of stdio

A target ``module:attr`` is a ``UniEnvMCPServer`` (or a function returning one), an ``Env`` (one
shared instance) or a function building one (an instance per open); policies likewise.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any, Optional

from unienv_interface.env_base.env import Env
from unienv_interface.policy import Policy

from .server import UniEnvMCPServer


def _load(target: str) -> Any:
    module, _, attr = target.partition(":")
    if not attr:
        raise SystemExit(f"Targets look like 'package.module:attribute', got {target!r}")
    sys.path.insert(0, "")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _server(target: str, name: Optional[str], policies: list, physical: bool) -> UniEnvMCPServer:
    obj = _load(target)
    if isinstance(obj, UniEnvMCPServer):
        return obj
    server = UniEnvMCPServer()
    env_name = name or target.rpartition(":")[2].split(".")[-1].removeprefix("make_") or "env"
    if isinstance(obj, Env):
        server.add_env(env_name, obj, physical=physical)
    elif callable(obj):
        built = obj()
        if isinstance(built, UniEnvMCPServer):
            server = built
        elif isinstance(built, Env):
            first = [built]
            server.add_env(env_name, lambda: first.pop() if first else obj(), physical=physical)
        else:
            raise SystemExit(f"{target} returned {type(built).__name__}; expected an Env or a UniEnvMCPServer")
    else:
        raise SystemExit(f"{target} is neither an Env, a UniEnvMCPServer nor a callable")
    for spec in policies:
        policy_target, _, policy_name = spec.partition("=")
        policy = _load(policy_target)
        if not (isinstance(policy, Policy) or callable(policy)):
            raise SystemExit(f"{policy_target} is neither a Policy nor a callable")
        server.add_policy(policy_name or policy_target.rpartition(":")[2].removeprefix("make_"), policy)
    return server


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(prog="unienv-mcp", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="serve an environment (and policies) from Python code")
    serve.add_argument("target", help="module:attr of a UniEnvMCPServer, an Env or an env factory")
    serve.add_argument("--name", help="the environment's name (tool prefix)")
    serve.add_argument("--policy", action="append", default=[], metavar="MODULE:ATTR[=NAME]",
                       help="a Policy or policy factory to expose (repeatable)")
    serve.add_argument("--physical", action="store_true", help="the environment is a real system")
    remote = sub.add_parser("bridge", help="serve what a UniEnv RemoteServer hosts")
    remote.add_argument("address", help="tcp://host:port, unix:///path or ws://host:port")
    remote.add_argument("--only", action="append", default=None, help="expose only these names (repeatable)")
    remote.add_argument("--physical", action="append", default=[], help="names of real systems (repeatable)")
    for command in (serve, remote):
        command.add_argument("--transport", choices=["stdio", "http"], default="stdio")
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.command == "serve":
        server = _server(args.target, args.name, args.policy, args.physical)
    else:
        from .bridge import bridge
        server = bridge(args.address, names=args.only, physical=args.physical)
    server.run(args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
