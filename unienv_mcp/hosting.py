"""Hosting MCP instances with UniEnv's remote execution, so other processes can use them.

The hosted objects are thin wrappers: every call runs on the instance's own thread (like the MCP
server's own calls), and every reset and step, whoever made it, goes through the MCP server's
bookkeeping, which records the instance's state and publishes the result to remote subscribers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from unienv_interface.env_base.env import Env
from unienv_interface.policy import Policy

if TYPE_CHECKING:
    from unienv_interface.remote import RemoteServer
    from .server import UniEnvMCPServer, _Instance

__all__ = ["HostedEnv", "HostedPolicy"]

DEFAULT_ADDRESS = "tcp://127.0.0.1:0"


@dataclass
class Hosting:
    """The remote server hosting one MCP instance (its env and the policies paired with it)."""

    server: "RemoteServer"
    addresses: List[str] = field(default_factory=list)
    env: Optional[str] = None  # resource name of the env, once hosted
    policies: Dict[str, str] = field(default_factory=dict)  # policy name -> resource name

    def describe(self) -> Dict[str, Any]:
        return {"addresses": list(self.addresses), "env": self.env, "policies": dict(self.policies)}


def on_thread(instance: "_Instance", fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on the instance's thread (directly if already there) and return its result."""
    if threading.get_ident() == instance.thread_ident:
        return fn()
    return instance.submit(fn).result()


class HostedEnv(Env):
    """The env of an MCP instance as seen by remote clients."""

    def __init__(self, mcp: "UniEnvMCPServer", instance: "_Instance") -> None:
        self._mcp, self._instance = mcp, instance
        env = instance.env
        self.backend, self.device = env.backend, env.device
        self.rng = getattr(env, "rng", None)

    # Live views, so the remote server notices spaces an env rebuilds on reset
    @property
    def _env(self) -> Env:
        return self._instance.env

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def context_space(self):
        return getattr(self._env, "context_space", None)

    @property
    def batch_size(self):
        return getattr(self._env, "batch_size", None)

    @property
    def render_mode(self):
        return getattr(self._env, "render_mode", None)

    @property
    def render_fps(self):
        return getattr(self._env, "render_fps", None)

    @property
    def metadata(self):
        return getattr(self._env, "metadata", {})

    def reset(self, *, mask: Any = None, seed: Optional[int] = None, **kwargs: Any):
        def run():
            kw = dict(kwargs)
            if mask is not None:
                kw["mask"] = mask
            result = self._env.reset(seed=seed, **kw)
            self._mcp._after_reset(self._instance, *result, full=mask is None, mask=mask)
            return result
        return on_thread(self._instance, run)

    def step(self, action: Any):
        def run():
            result = self._env.step(action)
            self._mcp._after_step(self._instance, *result)
            return result
        return on_thread(self._instance, run)

    def render(self):
        return on_thread(self._instance, self._env.render)

    def close(self) -> None:
        """No-op: the MCP server owns the env."""


class HostedPolicy(Policy):
    """A policy paired with an MCP instance, as seen by remote clients (calls run on the instance's thread)."""

    def __init__(self, instance: "_Instance", policy: Policy) -> None:
        self._instance, self._policy = instance, policy
        self.backend, self.device = policy.backend, policy.device
        self.batch_size = policy.batch_size
        self.rng = getattr(policy, "rng", None)

    @property
    def observation_space(self):
        return self._policy.observation_space

    @property
    def action_space(self):
        return self._policy.action_space

    @property
    def context_space(self):
        return getattr(self._policy, "context_space", None)

    @property
    def metadata(self):
        return getattr(self._policy, "metadata", {})

    def reset(self, context: Any, observation: Any, *, seed: Optional[int] = None, mask: Any = None,
              **kwargs: Any) -> None:
        return on_thread(self._instance, lambda: self._policy.reset(context, observation, seed=seed, mask=mask,
                                                                   **kwargs))

    def observe_transition(self, action_taken: Any, new_observation: Any, dt: float) -> None:
        return on_thread(self._instance, lambda: self._policy.observe_transition(action_taken, new_observation, dt))

    def get_action(self, **kwargs: Any) -> Any:
        return on_thread(self._instance, lambda: self._policy.get_action(**kwargs))

    def visualize_action(self, environment_render: Any, observation: Any, action: Any, **kwargs: Any) -> Any:
        return on_thread(self._instance, lambda: self._policy.visualize_action(environment_render, observation,
                                                                              action, **kwargs))

    def close(self) -> None:
        """No-op: the MCP server owns the policy."""
