"""Serve UniEnv environments and policies to MCP clients (LLM agents)."""

from __future__ import annotations

import itertools
import logging
import math
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import anyio
import numpy as np

import mcp_types as types
from mcp.server.lowlevel import Server
from unienv_interface.env_base.env import Env
from unienv_interface.policy import Policy
from unienv_interface.space.space_utils.json_utils import (bound_violations, describe_metadata, from_json,
                                                          space_to_json_schema, to_json)

from .hosting import DEFAULT_ADDRESS, HostedEnv, HostedPolicy, Hosting, on_thread
from .content import (ContentOptions, Rendered, error_result, image_paths, json_safe, render_image, render_observation,
                      result_schema, to_result, with_images)

__all__ = ["UniEnvMCPServer"]

_log = logging.getLogger(__name__)
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
EnvSource = Union[Env, Callable[[], Env]]
PolicySource = Union[Policy, Callable[[], Policy]]


class _ToolError(Exception):
    """A problem the model caused and can fix (unknown instance, bad action, ...): returned as a tool error."""


# ---------------------------------------------------------------------------- registrations and instances


@dataclass
class _EnvEntry:
    name: str
    factory: Optional[Callable[[], Env]]  # None: a shared object
    description: Optional[str]
    max_instances: int
    idle_timeout: Optional[float]
    dt: Optional[float]
    max_steps_per_call: int
    physical: bool
    clip_actions: bool
    spaces: Dict[str, Any] = field(default_factory=dict)  # observation / action / context space
    batch_size: Optional[int] = None
    spare: Optional["_Instance"] = None  # the instance built at registration, until the first open
    remote: Optional[Tuple[str, str]] = None  # (address, name) where a bridged env is already hosted


@dataclass
class _PolicyEntry:
    name: str
    factory: Optional[Callable[[], Policy]]
    shared: Optional[Policy]
    description: Optional[str]
    envs: Optional[Tuple[str, ...]]
    max_steps: int
    lock: threading.Lock = field(default_factory=threading.Lock)  # one run at a time on a shared policy
    remote: Optional[Tuple[str, str]] = None  # (address, name) where a bridged policy is already hosted


class _Instance:
    """One env object with its own worker thread (simulators often need thread affinity)."""

    def __init__(self, entry: _EnvEntry, instance_id: str, worker: ThreadPoolExecutor, env: Env) -> None:
        self.entry, self.id, self.worker, self.env = entry, instance_id, worker, env
        self.thread_ident = worker.submit(threading.get_ident).result()
        self.hosting: Optional[Hosting] = None  # the remote server hosting it, if any
        self.context: Any = None
        self.observation: Any = None
        self.reset_done = False
        self.episode_steps = 0
        self.policies: Dict[str, Policy] = {}  # factory-built policies paired with this instance
        self.last_used = time.monotonic()
        self.busy = 0  # calls and runs in flight; idle closing waits for 0
        self.active_run: Optional[str] = None  # the policy run in progress, if any
        # The outcome of the last reset / step / run, for observe
        self.last_rewards: List[Any] = []
        self.terminated = self.truncated = False
        self.info: Any = None
        self.closed = False

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future:
        return self.worker.submit(fn, *args)


@dataclass
class _Run:
    id: str
    policy: str
    instance: str
    max_steps: int
    status: str = "running"  # running, finished, cancelled, failed
    steps: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    result: Optional[Rendered] = None
    error: Optional[str] = None
    future: Optional[Future] = None
    finished_at: Optional[float] = None


async def _wait(future: Future) -> Any:
    return await anyio.to_thread.run_sync(future.result, abandon_on_cancel=True)


# ---------------------------------------------------------------------------- the server


class UniEnvMCPServer:
    """An MCP server exposing UniEnv environments and policies as tools.

    Example::

        server = UniEnvMCPServer("lab")
        server.add_env("sim", make_env)                    # a factory: one env per opened instance
        server.add_policy("pick", make_policy)             # runs closed-loop on an env instance
        server.run("stdio")                                # or "http"

    Each environment ``E`` gets the tools ``E.describe``, ``E.open``, ``E.reset``, ``E.step``,
    ``E.render`` and ``E.close``; each policy ``P`` gets ``P.run`` (a closed-loop run on an env
    instance, the way to use policies as skills) and ``P.suggest`` (one action, not executed); with
    any policy registered, ``runs.status`` and ``runs.cancel`` manage runs started with
    ``wait=false``. State lives in instances addressed by id (MCP itself may be stateless), and
    each instance runs on its own thread.

    Environments given as factories are built per ``open`` (one is built at registration to read
    the spaces for the tool schemas, and handed to the first ``open``); instances are closed by
    ``E.close`` or after ``idle_timeout`` seconds unused. An environment given as an object is one
    shared instance whose id is its name. Policies given as factories are built per env instance;
    a shared policy object runs one run at a time.

    Actions are JSON (see ``unienv_interface.space.space_utils.json_utils``): the tool schemas are
    generated from the action spaces, vectors with ``names`` metadata are objects keyed by name, and
    out-of-bounds actions are tool errors listing the violations (or clipped, with ``clip_actions``).
    Image leaves of observations (space metadata ``kind="image"`` / ``"depth"``) are returned as
    images; the rest as structured JSON.

    There is no access control between clients: anyone connected may call any tool. For physical
    systems (``physical=True``) the tools that act are annotated as destructive and open-world, so
    MCP hosts ask before running them; bind HTTP servers to trusted interfaces only.
    """

    def __init__(self, name: str = "unienv", *, instructions: Optional[str] = None,
                 content: Optional[ContentOptions] = None, max_finished_runs: int = 32, hosting: bool = True) -> None:
        self.name = name
        self.hosting = hosting
        self.content = content or ContentOptions()
        self.max_finished_runs = max_finished_runs
        self._instructions = instructions
        self._envs: Dict[str, _EnvEntry] = {}
        self._policies: Dict[str, _PolicyEntry] = {}
        self._instances: Dict[str, _Instance] = {}
        self._runs: Dict[str, _Run] = {}
        self._counter = itertools.count(1)
        self._lock = threading.RLock()
        self._closed = False
        self._server: Optional[Server] = None
        self._sweeper: Optional[threading.Thread] = None
        self._sweep_stop = threading.Event()

    # ------------------------------------------------------------------ registration

    def add_env(self, name: str, env: EnvSource, *, description: Optional[str] = None, max_instances: int = 4,
                idle_timeout: Optional[float] = 600.0, dt: Optional[float] = None, max_steps_per_call: int = 100,
                physical: bool = False, clip_actions: bool = False) -> None:
        """Expose an environment as tools ``<name>.*``.

        ``env`` is an ``Env`` (one shared instance) or a callable building one (an instance per
        ``open``, at most ``max_instances`` at once, closed after ``idle_timeout`` seconds unused;
        ``None``: never). ``dt`` is the duration of a step in seconds, passed to policies (default:
        the env's ``control_timestep``). ``max_steps_per_call`` bounds ``step``'s ``repeat``.
        ``physical`` marks a real system: acting tools are annotated as destructive.
        """
        self._check_name(name)
        if max_instances < 1:
            raise ValueError("max_instances must be at least 1")
        entry = _EnvEntry(name, None if isinstance(env, Env) else env, description, max_instances, idle_timeout, dt,
                          max_steps_per_call, physical, clip_actions)
        if isinstance(env, Env):
            instance = self._make_instance(entry, instance_id=name, env=env)
        elif callable(env):
            instance = self._make_instance(entry, instance_id=None, env=None)
            entry.spare = instance
        else:
            raise TypeError(f"env must be an Env or a callable building one, got {type(env).__name__}")
        built = instance.env
        entry.spaces = {"observation": built.observation_space, "action": built.action_space,
                        "context": getattr(built, "context_space", None)}
        entry.batch_size = getattr(built, "batch_size", None)
        if entry.dt is None:
            entry.dt = getattr(built, "control_timestep", None)
        space_to_json_schema(built.action_space)  # fail at registration if the action has no JSON form
        with self._lock:
            self._envs[name] = entry
            if entry.factory is None:
                self._instances[name] = instance

    def add_policy(self, name: str, policy: PolicySource, *, description: Optional[str] = None,
                   envs: Optional[Sequence[str]] = None, max_steps: int = 1000) -> None:
        """Expose a policy as tools ``<name>.run`` and ``<name>.suggest``.

        ``policy`` is a ``Policy`` (shared: one run at a time) or a callable building one (one per
        env instance, built on that instance's thread). ``envs`` limits the environments it may run
        on (default: all); ``max_steps`` caps a run's length.
        """
        self._check_name(name)
        if name == "runs":
            raise ValueError("'runs' is reserved for the run tools")
        entry = _PolicyEntry(name, None if isinstance(policy, Policy) else policy,
                             policy if isinstance(policy, Policy) else None, description,
                             tuple(envs) if envs is not None else None, max_steps)
        if entry.factory is None and entry.shared is None:
            if not callable(policy):
                raise TypeError(f"policy must be a Policy or a callable building one, got {type(policy).__name__}")
        with self._lock:
            self._policies[name] = entry

    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError(f"Names must be 1-48 letters, digits, '_' or '-' (they prefix tool names), got {name!r}")
        if name in self._envs or name in self._policies:
            raise ValueError(f"{name!r} is already registered")

    def _make_instance(self, entry: _EnvEntry, *, instance_id: Optional[str], env: Optional[Env]) -> _Instance:
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"unienv-mcp:{entry.name}")
        try:
            built = env if env is not None else worker.submit(entry.factory).result()
            if not isinstance(built, Env):
                raise TypeError(f"The factory of {entry.name!r} returned {type(built).__name__}, not an Env")
        except BaseException:
            worker.shutdown(wait=False)
            raise
        if instance_id is None:
            instance_id = f"{entry.name}-{next(self._counter)}"
        return _Instance(entry, instance_id, worker, built)

    # ------------------------------------------------------------------ tool list

    def _tools(self) -> List[types.Tool]:
        tools: List[types.Tool] = []
        instance = {"type": "string", "description": "the instance id returned by open (describe lists them)"}
        images = {"description": 'which observation images to return: "all", "none", or a list of camera paths '
                                 '(describe lists them); default: the server\'s choice',
                  "anyOf": [{"type": "string", "enum": ["all", "none"]}, {"type": "array", "items": {"type": "string"}}]}
        for name, entry in self._envs.items():
            acting = types.ToolAnnotations(read_only_hint=False, destructive_hint=entry.physical,
                                           idempotent_hint=False, open_world_hint=entry.physical)
            reading = types.ToolAnnotations(read_only_hint=True, open_world_hint=entry.physical)
            about = entry.description or describe_metadata(entry.spaces["observation"]) or "an environment"
            action_schema = space_to_json_schema(entry.spaces["action"])
            shared = entry.factory is None
            tools += [
                types.Tool(name=f"{name}.describe", description=f"Describe {name} ({about}): its observation, "
                           "action and context as JSON Schemas, its cameras, its timing, the policies that run on "
                           "it, and its open instances.",
                           input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                           annotations=types.ToolAnnotations(read_only_hint=True, idempotent_hint=True)),
                types.Tool(name=f"{name}.open", description=(
                    f"Get the id of the shared {name} instance." if shared else
                    f"Create a new {name} instance and return its id (close it with {name}.close)."),
                    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                    annotations=acting),
                types.Tool(name=f"{name}.reset", description=f"Reset a {name} instance: start a new episode. "
                           "Returns the context (e.g. the task) and the first observation; images follow as image "
                           "content, referenced from the JSON by number.",
                           input_schema={"type": "object", "properties": {
                               "instance": instance, "seed": {"type": "integer", "minimum": 0}, "images": images},
                               "required": ["instance"], "additionalProperties": False},
                           annotations=acting),
                types.Tool(name=f"{name}.step", description=(
                    f"Apply an action to a {name} instance, `repeat` times (stopping early when the episode "
                    "ends). Returns the rewards, whether the episode terminated or was truncated, and the last "
                    "observation."), input_schema={"type": "object", "properties": {
                        "instance": instance, "action": action_schema,
                        "repeat": {"type": "integer", "minimum": 1, "maximum": entry.max_steps_per_call,
                                   "default": 1}, "images": images},
                        "required": ["instance", "action"], "additionalProperties": False},
                    annotations=acting),
                types.Tool(name=f"{name}.observe", description=(
                    f"The current state of a {name} instance without acting: the context (task), the latest "
                    "observation, the episode's progress and whether it ended, and any policy run in progress. "
                    "Use it to look again, e.g. after your context was compacted."),
                    input_schema={"type": "object", "properties": {"instance": instance, "images": images},
                                  "required": ["instance"], "additionalProperties": False},
                    annotations=types.ToolAnnotations(read_only_hint=True, idempotent_hint=True)),
                types.Tool(name=f"{name}.render", description=f"Render a {name} instance as an image.",
                           input_schema={"type": "object", "properties": {"instance": instance},
                                         "required": ["instance"], "additionalProperties": False},
                           annotations=reading),
                *([types.Tool(name=f"{name}.host", description=(
                    f"Host a {name} instance with UniEnv remote execution, so other processes (a training script, "
                    "a viewer, a policy on another machine) can use it: RemoteClient.connect(address).env(name). "
                    "Its calls run on the instance's thread like yours, interleaved with them; remote subscribers "
                    "see every reset and step, yours and policy runs' included. `address`: tcp://host:port, "
                    "unix:///path or ws://host:port (default tcp://127.0.0.1 on a free port); call again to add "
                    "addresses. Anyone who can reach an address can control the instance."),
                    input_schema={"type": "object", "properties": {"instance": instance, "address": {"type": "string"}},
                                  "required": ["instance"], "additionalProperties": False},
                    annotations=types.ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                                      idempotent_hint=True, open_world_hint=True)),
                   types.Tool(name=f"{name}.unhost", description=(
                       f"Stop hosting a {name} instance (and the policies hosted with it); remote clients are "
                       "disconnected."),
                       input_schema={"type": "object", "properties": {"instance": instance},
                                     "required": ["instance"], "additionalProperties": False},
                       annotations=types.ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                                         idempotent_hint=True))] if self.hosting else []),
                types.Tool(name=f"{name}.close", description=(
                    f"Close a {name} instance." if not shared else f"No-op: the shared {name} instance stays open."),
                    input_schema={"type": "object", "properties": {"instance": instance},
                                  "required": ["instance"], "additionalProperties": False},
                    annotations=types.ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                                      idempotent_hint=True)),
            ]
        for name, entry in self._policies.items():
            envs = entry.envs or tuple(self._envs)
            physical = any(self._envs[e].physical for e in envs if e in self._envs)
            about = entry.description or "a policy"
            tools += [
                types.Tool(name=f"{name}.run", description=(
                    f"Run {name} ({about}) closed-loop on an instance of {', '.join(envs)}, from its current state, "
                    "until the episode ends or max_steps. `task` replaces the env's context (e.g. the instruction) "
                    "for the policy. Returns the outcome, the final observation and keyframes; with wait=false it "
                    "returns a run id at once (see runs.status)."),
                    input_schema={"type": "object", "properties": {
                        "instance": instance, "task": {"type": "string"},
                        "max_steps": {"type": "integer", "minimum": 1, "maximum": entry.max_steps},
                        "keyframes": {"type": "integer", "minimum": 0, "maximum": 8, "default": 3},
                        "wait": {"type": "boolean", "default": True}, "images": images},
                        "required": ["instance"], "additionalProperties": False},
                    annotations=types.ToolAnnotations(read_only_hint=False, destructive_hint=physical,
                                                      idempotent_hint=False, open_world_hint=physical)),
                *([types.Tool(name=f"{name}.host", description=(
                    f"Host {name} as paired with an env instance (the policy object its runs use), with UniEnv "
                    "remote execution: RemoteClient.connect(address).policy(name). It is served with the "
                    "instance (see <env>.host for `address`); <env>.unhost stops both."),
                    input_schema={"type": "object", "properties": {"instance": instance, "address": {"type": "string"}},
                                  "required": ["instance"], "additionalProperties": False},
                    annotations=types.ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                                      idempotent_hint=True, open_world_hint=True))]
                  if self.hosting else []),
                types.Tool(name=f"{name}.suggest", description=(
                    f"The action {name} would take now on an env instance, without executing it (apply it with "
                    "<env>.step, possibly edited). `task` replaces the env's context for the policy."),
                    input_schema={"type": "object", "properties": {"instance": instance, "task": {"type": "string"}},
                                  "required": ["instance"], "additionalProperties": False},
                    annotations=types.ToolAnnotations(read_only_hint=True)),
            ]
        if self._policies:
            tools += [
                types.Tool(name="runs.status", description="The status of a policy run; its outcome once finished.",
                           input_schema={"type": "object", "properties": {"run": {"type": "string"}},
                                         "required": ["run"], "additionalProperties": False},
                           annotations=types.ToolAnnotations(read_only_hint=True)),
                types.Tool(name="runs.cancel", description="Stop a running policy run after its current step.",
                           input_schema={"type": "object", "properties": {"run": {"type": "string"}},
                                         "required": ["run"], "additionalProperties": False},
                           annotations=types.ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                                             idempotent_hint=True)),
            ]
        return tools

    def _server_instructions(self) -> str:
        if self._instructions:
            return self._instructions
        lines = ["UniEnv environments and policies. For an environment E: E.describe explains it (and lists open "
                 "instances), E.open gives an instance id, E.reset starts an episode, E.step acts, E.observe shows "
                 "the current state again without acting, E.close frees the instance."]
        if self._policies:
            lines.append("Policies are skills: P.run executes one on an env instance from its current state, "
                         "optionally with a task (instruction) for the policy; chain runs to follow a plan.")
        lines.append("Observation images are returned as image content after the JSON, which refers to them by "
                     "number; pass images=\"none\" or a list of cameras to get fewer.")
        return " ".join(lines)

    # ------------------------------------------------------------------ MCP plumbing

    @property
    def mcp_server(self) -> Server:
        """The underlying MCP server (e.g. for ``mcp.Client(server.mcp_server)`` in tests)."""
        if self._server is None:
            self._server = Server(self.name, instructions=self._server_instructions(),
                                  on_list_tools=self._on_list_tools, on_call_tool=self._on_call_tool)
            self._start_sweeper()
        return self._server

    async def _on_list_tools(self, ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=self._tools())

    async def _on_call_tool(self, ctx: Any, params: Any) -> types.CallToolResult:
        name = params.name
        arguments = dict(params.arguments or {})
        target, _, verb = name.rpartition(".")
        try:
            if target == "runs" and self._policies:
                return await self._runs_tool(verb, arguments)
            if target in self._envs:
                return await self._env_tool(self._envs[target], verb, arguments)
            if target in self._policies:
                return await self._policy_tool(ctx, self._policies[target], verb, arguments)
            return error_result(f"Unknown tool {name!r}")
        except _ToolError as exc:
            return error_result(str(exc))
        except Exception as exc:  # the environment or policy failed: tell the model instead of failing the request
            _log.exception("Tool %s failed", name)
            return error_result(f"{name} failed: {type(exc).__name__}: {exc}")

    def run(self, transport: str = "stdio", *, host: str = "127.0.0.1", port: int = 8000, path: str = "/mcp") -> None:
        """Serve until interrupted: ``"stdio"`` (for local MCP hosts) or ``"http"`` (Streamable HTTP)."""
        server = self.mcp_server
        try:
            if transport == "stdio":
                from mcp.server.stdio import stdio_server

                async def main() -> None:
                    async with stdio_server() as (read, write):
                        await server.run(read, write, server.create_initialization_options())

                anyio.run(main)
            elif transport == "http":
                import uvicorn

                app = server.streamable_http_app(streamable_http_path=path, host=host)
                uvicorn.run(app, host=host, port=port, log_level="warning")
            else:
                raise ValueError(f"transport must be 'stdio' or 'http', got {transport!r}")
        finally:
            self.close()

    # ------------------------------------------------------------------ environments

    def _instance(self, arguments: Dict[str, Any], entry: Optional[_EnvEntry] = None) -> _Instance:
        instance_id = arguments.get("instance")
        with self._lock:
            instance = self._instances.get(instance_id) if isinstance(instance_id, str) else None
        if instance is None or instance.closed:
            raise _ToolError(f"No open instance {instance_id!r}" + (f"; call {entry.name}.open first" if entry else ""))
        if entry is not None and instance.entry is not entry:
            raise _ToolError(f"{instance_id!r} is an instance of {instance.entry.name}, not {entry.name}")
        instance.last_used = time.monotonic()
        return instance

    async def _call(self, instance: _Instance, fn: Callable[..., Any], *args: Any) -> Any:
        with self._lock:
            instance.busy += 1
        try:
            return await _wait(instance.submit(fn, *args))
        finally:
            with self._lock:
                instance.busy -= 1
                instance.last_used = time.monotonic()

    async def _env_tool(self, entry: _EnvEntry, verb: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        if verb == "describe":
            return self._describe(entry)
        if verb == "open":
            return await self._open(entry)
        if verb == "close":
            return await self._close_instance(self._instance(arguments, entry))
        instance = self._instance(arguments, entry)
        if verb == "host" and self.hosting:
            return await self._host(instance, arguments.get("address"))
        if verb == "unhost" and self.hosting:
            return await self._unhost(instance)
        options = self._options(arguments)
        if verb == "observe":
            return self._observe(instance, options)
        if verb in ("reset", "step") and instance.active_run is not None:
            raise _ToolError(f"{instance.id} is running {instance.active_run}; wait for it or cancel it (runs.cancel)")
        if verb == "reset":
            return await self._call(instance, self._reset, instance, arguments.get("seed"), options)
        if verb == "step":
            if not instance.reset_done:
                raise _ToolError(f"Reset {instance.id} first ({entry.name}.reset)")
            action = self._action(entry, arguments.get("action"))
            repeat = arguments.get("repeat", 1)
            if not isinstance(repeat, int) or not 1 <= repeat <= entry.max_steps_per_call:
                raise _ToolError(f"repeat must be an integer in [1, {entry.max_steps_per_call}]")
            return await self._call(instance, self._step, instance, action, repeat, options)
        if verb == "render":
            frame = await self._call(instance, instance.env.render)
            out = Rendered({"instance": instance.id})
            if frame is None or render_image(frame, f"{instance.id} render", out, self.content) is None:
                raise _ToolError(f"{entry.name} renders no image (render_mode {getattr(instance.env, 'render_mode', None)!r})")
            out.structured["render"] = {"image": 1}
            return to_result(out)
        raise _ToolError(f"Unknown tool {entry.name}.{verb}")

    def _reset(self, instance: _Instance, seed: Optional[int], options: ContentOptions) -> types.CallToolResult:
        """A reset on the instance's thread."""
        entry = instance.entry
        context, observation, info = instance.env.reset(seed=seed)
        self._after_reset(instance, context, observation, info)
        out = Rendered({"instance": instance.id})
        render_observation(entry.spaces["context"], context, out, "context", batched=False, options=options)
        render_observation(entry.spaces["observation"], observation, out, "observation",
                           batched=entry.batch_size is not None, options=options)
        out.structured["info"] = instance.info
        return to_result(out)

    # Every reset and step of an instance, from tools, runs or remote clients, goes through these two, on the
    # instance's thread: they keep the state observe reports, and publish to remote subscribers in order.

    def _after_reset(self, instance: _Instance, context: Any, observation: Any, info: Any, *, full: bool = True,
                     mask: Any = None) -> None:
        if not full and mask is not None and instance.observation is not None:
            env = instance.env
            observation = env.update_observation_post_reset(instance.observation, observation, mask)
            if context is not None and instance.context is not None:
                context = env.update_context_post_reset(instance.context, context, mask)
        instance.context, instance.observation = context, observation
        instance.reset_done = True
        if full:
            instance.episode_steps = 0
        instance.last_rewards, instance.terminated, instance.truncated = [], False, False
        instance.info = json_safe(info, self.content.digits)
        instance.last_used = time.monotonic()
        self._publish(instance, "reset", (context, observation, info))

    def _after_step(self, instance: _Instance, observation: Any, reward: Any, terminated: Any, truncated: Any,
                    info: Any) -> Tuple[Any, bool, bool]:
        """Record a step; returns (the reward as JSON, terminated, truncated)."""
        instance.observation = observation
        instance.episode_steps += 1
        reward_json = json_safe(reward, self.content.digits)
        instance.last_rewards = [reward_json]
        instance.terminated, instance.truncated = _any(terminated), _any(truncated)
        instance.info = json_safe(info, self.content.digits)
        instance.last_used = time.monotonic()
        self._publish(instance, "step", (observation, reward, terminated, truncated, info))
        return reward_json, instance.terminated, instance.truncated

    def _publish(self, instance: _Instance, method: str, value: Any) -> None:
        hosting = instance.hosting
        if hosting is None or hosting.env is None:
            return
        try:
            hosting.server.publish(hosting.env, method, value)
        except Exception:
            _log.exception("Publishing %s of %s failed", method, instance.id)

    # ------------------------------------------------------------------ hosting

    async def _host(self, instance: _Instance, address: Optional[str],
                    policy: Optional[_PolicyEntry] = None) -> types.CallToolResult:
        if address is not None and not isinstance(address, str):
            raise _ToolError("address must be a string such as tcp://127.0.0.1:7000")
        remote = (policy.remote if policy is not None else instance.entry.remote)
        if remote is not None:
            origin, name = remote
            kind = "policy" if policy is not None else "env"
            return to_result(Rendered({"instance": instance.id, "addresses": [origin], kind: name,
                                       "note": "already hosted by the remote server this MCP server bridges",
                                       "connect": f'RemoteClient.connect("{origin}").{kind}("{name}")'}))
        try:
            hosting = await anyio.to_thread.run_sync(self._host_sync, instance, address, policy)
        except (OSError, ValueError, RuntimeError) as exc:
            raise _ToolError(f"Cannot host {instance.id}{f' on {address}' if address else ''}: {exc}") from None
        kind, name = ("policy", hosting.policies[policy.name]) if policy is not None else ("env", hosting.env)
        return to_result(Rendered({
            "instance": instance.id, "addresses": list(hosting.addresses), kind: name, "hosted": hosting.describe(),
            "connect": f'RemoteClient.connect("{hosting.addresses[0]}").{kind}("{name}")',
            "note": "anyone who can reach these addresses can control it; remote subscribers see every reset and "
                    "step, including yours and policy runs'",
        }))

    def _host_sync(self, instance: _Instance, address: Optional[str], policy: Optional[_PolicyEntry]) -> Hosting:
        from unienv_interface.remote import RemoteServer

        with self._lock:
            if instance.closed:
                raise RuntimeError("the instance is closed")
            hosting = instance.hosting
            if hosting is None:
                hosting = instance.hosting = Hosting(RemoteServer())
        if policy is None:
            if hosting.env is None:
                hosting.server.register(instance.id, HostedEnv(self, instance), close=False, publish_results=False)
                hosting.env = instance.id
        elif policy.name not in hosting.policies:
            built = on_thread(instance, lambda: self._policy_for(policy, instance))
            name = f"{policy.name}-{instance.id}"
            hosting.server.register(name, HostedPolicy(instance, built), close=False)
            hosting.policies[policy.name] = name
        if address is not None and address not in hosting.addresses:
            hosting.addresses.append(hosting.server.listen(address))
        elif not hosting.addresses:
            hosting.addresses.append(hosting.server.listen(DEFAULT_ADDRESS))
        return hosting

    async def _unhost(self, instance: _Instance) -> types.CallToolResult:
        with self._lock:
            hosting, instance.hosting = instance.hosting, None
        if hosting is None:
            return to_result(Rendered({"instance": instance.id, "hosted": None, "note": "it was not hosted"}))
        await anyio.to_thread.run_sync(hosting.server.close)
        return to_result(Rendered({"instance": instance.id, "hosted": None, "stopped": hosting.addresses}))

    def _options(self, arguments: Dict[str, Any]) -> ContentOptions:
        try:
            return with_images(self.content, arguments.get("images"))
        except ValueError as exc:
            raise _ToolError(str(exc)) from None

    def _describe(self, entry: _EnvEntry) -> types.CallToolResult:
        spaces = entry.spaces
        batched = entry.batch_size is not None
        about: Dict[str, Any] = {
            "name": entry.name, "description": entry.description,
            "shared": entry.factory is None, "physical": entry.physical,
            "batch_size": entry.batch_size, "step_seconds": entry.dt,
            "action": space_to_json_schema(spaces["action"]),
            # As results show them: image leaves are image references, large arrays summaries
            "observation": result_schema(spaces["observation"], batched, self.content),
            "context": result_schema(spaces["context"], False, self.content),
            "cameras": image_paths(spaces["observation"], batched),
            "images": {"shown_by_default": "all" if self.content.cameras is None else list(self.content.cameras),
                       "max_per_result": self.content.max_images,
                       "batched_envs": self.content.batch_images if batched else None},
        }
        about["policies"] = [name for name, p in self._policies.items() if p.envs is None or entry.name in p.envs]
        with self._lock:
            instances = [i for i in self._instances.values() if i.entry is entry and not i.closed]
        about["instances"] = [{"instance": i.id, "episode_steps": i.episode_steps if i.reset_done else None,
                               "episode_ended": i.terminated or i.truncated, "run": i.active_run,
                               "hosted": i.hosting.describe() if i.hosting else None} for i in instances]
        return to_result(Rendered(about))

    def _observe(self, instance: _Instance, options: ContentOptions) -> types.CallToolResult:
        """The instance's state as the last reset / step / run left it; the env is not called."""
        entry = instance.entry
        out = Rendered({"instance": instance.id, "episode_started": instance.reset_done,
                        "episode_steps": instance.episode_steps if instance.reset_done else None,
                        "terminated": instance.terminated, "truncated": instance.truncated,
                        "last_rewards": instance.last_rewards, "run": instance.active_run,
                        "hosted": instance.hosting.describe() if instance.hosting else None})
        if not instance.reset_done:
            out.structured["next"] = f"{entry.name}.reset to start an episode"
            return to_result(out)
        render_observation(entry.spaces["context"], instance.context, out, "context", batched=False, options=options)
        render_observation(entry.spaces["observation"], instance.observation, out, "observation",
                           batched=entry.batch_size is not None, options=options)
        out.structured["info"] = instance.info
        if instance.terminated or instance.truncated:
            out.structured["next"] = f"the episode ended: {entry.name}.reset to start another"
        return to_result(out)

    async def _open(self, entry: _EnvEntry) -> types.CallToolResult:
        if entry.factory is None:
            return to_result(Rendered({"instance": entry.name, "shared": True,
                                       "next": f"{entry.name}.reset to start an episode"}))
        with self._lock:
            count = sum(1 for i in self._instances.values() if i.entry is entry and not i.closed)
            if count >= entry.max_instances:
                raise _ToolError(f"{entry.name} already has {count} open instances (the maximum); close one first")
            spare, entry.spare = entry.spare, None
        instance = spare
        if instance is None:
            instance = await anyio.to_thread.run_sync(lambda: self._make_instance(entry, instance_id=None, env=None))
        with self._lock:
            self._instances[instance.id] = instance
        instance.last_used = time.monotonic()
        return to_result(Rendered({"instance": instance.id, "shared": False,
                                   "next": f"{entry.name}.reset to start an episode"}))

    async def _close_instance(self, instance: _Instance) -> types.CallToolResult:
        if instance.entry.factory is None:
            return to_result(Rendered({"instance": instance.id, "closed": False,
                                       "note": "shared instances stay open"}))
        with self._lock:
            if instance.busy:
                raise _ToolError(f"{instance.id} is busy (a run or call is in progress); cancel the run first")
            instance.closed = True
            self._instances.pop(instance.id, None)
            hosting, instance.hosting = instance.hosting, None
        if hosting is not None:
            await anyio.to_thread.run_sync(hosting.server.close)
        await anyio.to_thread.run_sync(self._dispose, instance)
        return to_result(Rendered({"instance": instance.id, "closed": True}))

    def _dispose(self, instance: _Instance) -> None:
        """Close a factory-built instance (and its policies) on its own thread, then stop the thread."""
        def close_all() -> None:
            for policy in instance.policies.values():
                _close_quietly(policy)
            _close_quietly(instance.env)

        try:
            instance.submit(close_all).result()
        finally:
            instance.worker.shutdown(wait=False)

    def _action(self, entry: _EnvEntry, value: Any) -> Any:
        space = entry.spaces["action"]
        if value is None:
            raise _ToolError("action is required")
        try:
            action = from_json(space, value)
        except ValueError as exc:
            raise _ToolError(f"Invalid action: {exc}. See {entry.name}.describe for its schema.") from None
        violations = bound_violations(space, action)
        if violations:
            if entry.clip_actions and hasattr(space, "clip"):
                return space.clip(action)
            raise _ToolError("Action out of bounds: " + "; ".join(violations))
        return action

    def _step(self, instance: _Instance, action: Any, repeat: int, options: ContentOptions) -> types.CallToolResult:
        """``repeat`` steps on the instance's thread."""
        entry = instance.entry
        rewards, info, terminated, truncated, observation = [], {}, False, False, instance.observation
        for _ in range(repeat):
            observation, reward, term, trunc, info = instance.env.step(action)
            reward_json, terminated, truncated = self._after_step(instance, observation, reward, term, trunc, info)
            rewards.append(reward_json)
            if terminated or truncated:
                break
        instance.last_rewards = rewards
        out = Rendered({"instance": instance.id, "steps": len(rewards), "episode_steps": instance.episode_steps,
                        "rewards": rewards, "terminated": terminated, "truncated": truncated})
        render_observation(entry.spaces["observation"], observation, out, "observation",
                           batched=entry.batch_size is not None, options=options)
        out.structured["info"] = instance.info
        if terminated or truncated:
            out.structured["next"] = f"the episode ended: {entry.name}.reset to start another"
        return to_result(out)

    # ------------------------------------------------------------------ policies

    async def _policy_tool(self, ctx: Any, entry: _PolicyEntry, verb: str,
                           arguments: Dict[str, Any]) -> types.CallToolResult:
        instance = self._instance(arguments)
        if entry.envs is not None and instance.entry.name not in entry.envs:
            raise _ToolError(f"{entry.name} runs on {', '.join(entry.envs)}, not {instance.entry.name}")
        if verb == "host" and self.hosting:
            return await self._host(instance, arguments.get("address"), policy=entry)
        if not instance.reset_done:
            raise _ToolError(f"Reset {instance.id} first ({instance.entry.name}.reset)")
        task = arguments.get("task")
        if task is not None and not isinstance(task, str):
            raise _ToolError("task must be a string")
        if verb == "suggest":
            if instance.active_run is not None:
                raise _ToolError(f"{instance.id} is running {instance.active_run}; wait for it or cancel it (runs.cancel)")
            return await self._call(instance, self._suggest, entry, instance, task)
        if verb != "run":
            raise _ToolError(f"Unknown tool {entry.name}.{verb}")
        dt = instance.entry.dt
        if dt is None:
            raise _ToolError(f"{instance.entry.name} has no step duration for policies: register it with dt=...")
        max_steps = arguments.get("max_steps", entry.max_steps)
        keyframes = arguments.get("keyframes", 3)
        if not isinstance(max_steps, int) or not 1 <= max_steps <= entry.max_steps:
            raise _ToolError(f"max_steps must be an integer in [1, {entry.max_steps}]")
        if not isinstance(keyframes, int) or not 0 <= keyframes <= 8:
            raise _ToolError("keyframes must be an integer in [0, 8]")
        options = self._options(arguments)
        if options.cameras is not None and len(options.cameras) == 0:
            keyframes = 0  # images="none"
        if entry.shared is not None and not entry.lock.acquire(blocking=False):
            raise _ToolError(f"{entry.name} is a shared policy and already running; wait for that run or cancel it")
        run = _Run(f"run-{next(self._counter)}", entry.name, instance.id, max_steps)
        with self._lock:
            if instance.active_run is not None:
                if entry.shared is not None:
                    entry.lock.release()
                raise _ToolError(f"{instance.id} is running {instance.active_run}; wait for it or cancel it")
            self._runs[run.id] = run
            instance.busy += 1
            instance.active_run = run.id
        run.future = instance.submit(self._run, entry, instance, run, task, dt, keyframes, options)
        run.future.add_done_callback(lambda _: self._run_done(entry, instance))
        if not arguments.get("wait", True):
            return to_result(Rendered({"run": run.id, "status": "running",
                                       "next": "runs.status to follow it, runs.cancel to stop it"}))
        try:
            reported = -1
            while not run.future.done():  # poll: no thread waits on the run
                if run.steps != reported:
                    reported = run.steps
                    await ctx.session.report_progress(run.steps, run.max_steps, f"{run.steps} steps")
                await anyio.sleep(0.1)
        except anyio.get_cancelled_exc_class():
            run.cancel.set()  # the client cancelled the call: stop the run too
            raise
        return self._run_result(run)

    def _run_done(self, entry: _PolicyEntry, instance: _Instance) -> None:
        with self._lock:
            instance.busy -= 1
            instance.active_run = None
            instance.last_used = time.monotonic()
        if entry.shared is not None:
            entry.lock.release()
        self._prune_runs()

    def _policy_for(self, entry: _PolicyEntry, instance: _Instance) -> Policy:
        """The policy object for runs on ``instance`` (called on its thread)."""
        if entry.shared is not None:
            return entry.shared
        policy = instance.policies.get(entry.name)
        if policy is None:
            policy = entry.factory()
            if not isinstance(policy, Policy):
                raise TypeError(f"The factory of {entry.name!r} returned {type(policy).__name__}, not a Policy")
            instance.policies[entry.name] = policy
        return policy

    def _suggest(self, entry: _PolicyEntry, instance: _Instance, task: Optional[str]) -> types.CallToolResult:
        policy = self._policy_for(entry, instance)
        policy.reset(task if task is not None else instance.context, instance.observation)
        action = policy.get_action()
        space = instance.entry.spaces["action"]
        return to_result(Rendered({"instance": instance.id, "policy": entry.name,
                                   "action": to_json(space, action, digits=self.content.digits),
                                   "next": f"{instance.entry.name}.step to apply it"}))

    def _run(self, entry: _PolicyEntry, instance: _Instance, run: _Run, task: Optional[str], dt: float,
             keyframes: int, options: ContentOptions) -> None:
        """A closed-loop run on the instance's thread."""
        env_entry = instance.entry
        out = Rendered()
        frames: List[Tuple[int, Any]] = []
        stride = 1  # frames are kept at multiples of a stride that doubles as the run grows
        total_reward = 0.0
        terminated = truncated = success = False
        info: Any = {}
        rewards: List[Any] = []
        try:
            policy = self._policy_for(entry, instance)
            policy.reset(task if task is not None else instance.context, instance.observation)
            observation = instance.observation
            for step in range(run.max_steps):
                if run.cancel.is_set():
                    run.status = "cancelled"
                    break
                action = policy.get_action()
                observation, reward, term, trunc, info = instance.env.step(action)
                self._after_step(instance, observation, reward, term, trunc, info)
                run.steps = step + 1
                rewards = [json_safe(reward, self.content.digits)]
                total_reward += float(np.sum(np.asarray(json_safe(reward), dtype=float)))
                terminated, truncated = _any(term), _any(trunc)
                success = success or _any(info.get("success", False)) if isinstance(info, dict) else success
                if keyframes and run.steps % stride == 0:
                    frames.append((run.steps, self._keyframe(instance, observation)))
                    if len(frames) > 2 * keyframes:
                        stride *= 2
                        frames = [(s, f) for s, f in frames if s % stride == 0]
                if terminated or truncated or success:
                    break
                policy.observe_transition(action, observation, dt)
            else:
                run.status = "finished"
            if run.status == "running":
                run.status = "finished"
        except Exception as exc:  # report to the model instead of crashing the server
            _log.exception("Run %s failed", run.id)
            run.status, run.error = "failed", f"{type(exc).__name__}: {exc}"
        out.structured.update({"run": run.id, "policy": entry.name, "instance": instance.id, "status": run.status,
                               "task": task, "steps": run.steps, "episode_steps": instance.episode_steps,
                               "terminated": terminated, "truncated": truncated, "success": success,
                               "total_reward": round(total_reward, self.content.digits),
                               "info": json_safe(info, self.content.digits)})
        if run.error:
            out.structured["error"] = run.error
        if run.steps:
            instance.last_rewards, instance.terminated, instance.truncated = rewards, terminated, truncated
            instance.info = json_safe(info, self.content.digits)
        if keyframes and run.steps and (not frames or frames[-1][0] != run.steps):
            frames.append((run.steps, self._keyframe(instance, instance.observation)))
        for step, frame in _spread(frames, keyframes):
            number = render_image(frame, f"keyframe at step {step}", out, options) if frame is not None else None
            out.structured.setdefault("keyframes", []).append({"step": step, "image": number})
        render_observation(env_entry.spaces["observation"], instance.observation, out, "observation",
                           batched=env_entry.batch_size is not None, options=options)
        if terminated or truncated:
            out.structured["next"] = f"the episode ended: {env_entry.name}.reset to start another"
        run.result, run.finished_at = out, time.monotonic()

    def _keyframe(self, instance: _Instance, observation: Any) -> Any:
        """A frame to show: the env's render if it renders images, else the first image in the observation."""
        env = instance.env
        if getattr(env, "render_mode", None) == "rgb_array":
            try:
                return env.render()
            except Exception:
                pass
        from .images import image_kind  # the first image leaf
        space = instance.entry.spaces["observation"]
        stack = [(space, observation)]
        while stack:
            sub, data = stack.pop(0)
            if hasattr(sub, "spaces") and isinstance(getattr(sub, "spaces"), dict):
                stack += [(s, data[k]) for k, s in sub.spaces.items()]
            elif image_kind(sub, instance.entry.batch_size is not None) == "image":
                array = np.asarray(sub.backend.to_numpy(data))
                return array if instance.entry.batch_size is None else array.reshape(-1, *array.shape[-3:])[0]
        return None

    def _run_result(self, run: _Run) -> types.CallToolResult:
        if run.result is None:
            out = Rendered({"run": run.id, "policy": run.policy, "instance": run.instance, "status": run.status,
                            "steps": run.steps, "max_steps": run.max_steps})
            return to_result(out)
        return to_result(run.result)

    async def _runs_tool(self, verb: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        with self._lock:
            run = self._runs.get(arguments.get("run"))
        if run is None:
            raise _ToolError(f"No run {arguments.get('run')!r} (finished runs are kept for a while)")
        if verb == "status":
            return self._run_result(run)
        if verb == "cancel":
            run.cancel.set()
            return to_result(Rendered({"run": run.id, "status": run.status if run.result else "cancelling"}))
        raise _ToolError(f"Unknown tool runs.{verb}")

    def _prune_runs(self) -> None:
        with self._lock:
            finished = sorted((r for r in self._runs.values() if r.finished_at is not None),
                              key=lambda r: r.finished_at)
            for run in finished[:max(0, len(finished) - self.max_finished_runs)]:
                del self._runs[run.id]

    # ------------------------------------------------------------------ lifetime

    def _start_sweeper(self) -> None:
        if self._sweeper is None:
            self._sweeper = threading.Thread(target=self._sweep, name="unienv-mcp-sweeper", daemon=True)
            self._sweeper.start()

    def _sweep(self) -> None:
        """Close factory-built instances unused for longer than their idle timeout."""
        while not self._sweep_stop.wait(1.0):
            now = time.monotonic()
            with self._lock:
                stale = [i for i in self._instances.values()
                         if i.entry.factory is not None and i.entry.idle_timeout is not None and not i.busy
                         and i.hosting is None and now - i.last_used > i.entry.idle_timeout]
                for instance in stale:
                    instance.closed = True
                    del self._instances[instance.id]
            for instance in stale:
                _log.info("Closing idle instance %s", instance.id)
                try:
                    self._dispose(instance)
                except Exception:
                    _log.exception("Closing %s failed", instance.id)

    def close(self) -> None:
        """Stop runs and close every factory-built instance (shared objects belong to the caller)."""
        if self._closed:
            return
        self._closed = True
        self._sweep_stop.set()
        with self._lock:
            runs = list(self._runs.values())
            instances = [i for i in self._instances.values() if i.entry.factory is not None]
            spares = [e.spare for e in self._envs.values() if e.spare is not None]
            shared = [i for i in self._instances.values() if i.entry.factory is None]
            self._instances.clear()
        for run in runs:
            run.cancel.set()
        for instance in instances + shared:
            hosting, instance.hosting = instance.hosting, None
            if hosting is not None:
                try:
                    hosting.server.close()
                except Exception:
                    _log.exception("Stopping the hosting of %s failed", instance.id)
        for instance in instances + spares:
            try:
                self._dispose(instance)
            except Exception:
                _log.exception("Closing %s failed", instance.id)
        for instance in shared:
            instance.worker.shutdown(wait=False)

    def __enter__(self) -> "UniEnvMCPServer":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _spread(frames: List[Tuple[int, Any]], count: int) -> List[Tuple[int, Any]]:
    """``count`` frames spread evenly over ``frames``, the last one included."""
    if count <= 0 or not frames:
        return []
    if len(frames) <= count:
        return frames
    picks = sorted({round((i + 1) * len(frames) / count) - 1 for i in range(count)})
    return [frames[i] for i in picks]


def _any(value: Any) -> bool:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return bool(np.any(np.asarray(value)))


def _close_quietly(obj: Any) -> None:
    try:
        obj.close()
    except Exception:
        _log.exception("Closing %r failed", obj)
