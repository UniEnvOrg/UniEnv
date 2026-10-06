"""``WorldNode`` adapter that exposes an unbatched :class:`Env` as a node."""

from typing import Any, Dict, Optional, Tuple, Union

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType

from ..node import WorldNode
from ..world import World


class EnvAsWorldNode(WorldNode[ContextType, ObsType, ActType, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """Adapt an existing unbatched :class:`Env` so it can be composed as a ``WorldNode``.

    Design
    ------
    ``Env.step(action)`` is atomic: it produces the next observation, reward,
    termination/truncation signals, and info in a single call.  A ``WorldNode``
    instead splits the step lifecycle into::

        set_next_action(action)
          -> pre_environment_step(dt, priority=...)
          -> World.step()
          -> post_environment_step(dt, priority=...)
          -> get_observation() / get_reward() / ... / get_info()

    This adapter bridges the mismatch by performing the atomic inner
    ``Env.step`` inside ``pre_environment_step`` (using the action cached by the
    preceding ``set_next_action`` call) and caching its five-tuple result for
    the ``get_*`` accessors to expose.  ``reset`` maps directly to
    ``Env.reset``, and the base ``WorldNode.reload`` default (which delegates to
    ``reset``) is exactly right for a pre-built environment, hence the
    non-empty ``reset_priorities`` / ``reload_priorities`` sets.

    The ``dt`` argument of ``pre_environment_step`` is intentionally ignored:
    the wrapped environment is self-paced (e.g. real hardware running at its own
    control rate), while the outer world is expected to be a bookkeeping-only
    world such as :class:`RealWorld`.  ``control_timestep`` / ``update_timestep``
    can still be set for step-ratio bookkeeping by the composer.

    No schema transformation / re-keying is performed: the inner environment's
    action, observation, and context spaces (and the raw observation / context
    layouts) pass through unchanged.

    Only **unbatched** environments (``batch_size is None``) are supported.  The
    optional ``world`` is only used for backend/device delegation; when provided
    its backend type must match the inner environment's backend, since the node
    serves the inner environment's arrays.
    """

    # The wrapped environment is pre-built, so both reset and reload map onto
    # ``Env.reset`` at priority 0.  The priority sets must be non-empty for the
    # composer to actually dispatch the lifecycle calls.
    reset_priorities = {0}
    reload_priorities = {0}
    pre_environment_step_priorities = {0}
    after_reset_priorities = set()
    after_reload_priorities = set()
    post_environment_step_priorities = set()

    def __init__(
        self,
        env: Env,
        name: str,
        *,
        world: Optional[World] = None,
        control_timestep: Optional[float] = None,
        update_timestep: Optional[float] = None,
        hold_last_action: bool = False,
        expose_reward: bool = True,
        expose_termination: bool = True,
        expose_truncation: bool = True,
        forward_info: bool = True,
        reset_seed: Optional[int] = None,
        pre_step_priority: int = 0,
    ):
        if env.batch_size is not None:
            raise ValueError(
                f"EnvAsWorldNode only supports unbatched environments, but "
                f"{type(env).__name__} has batch_size={env.batch_size}. "
                f"Unbatch the environment before composing it as a WorldNode."
            )
        if world is not None and type(world.backend) is not type(env.backend):
            raise ValueError(
                f"Backend mismatch: world backend type is {type(world.backend).__name__} "
                f"but env backend type is {type(env.backend).__name__}. "
                f"EnvAsWorldNode serves the inner environment's arrays, so both must "
                f"use the same backend type."
            )

        self.env = env
        self.name = name
        self.world = world

        self._hold_last_action = hold_last_action
        self._forward_info = forward_info
        self.reset_seed = reset_seed

        if control_timestep is None and env.render_fps is not None:
            control_timestep = 1.0 / env.render_fps
        self.control_timestep = control_timestep
        self.update_timestep = update_timestep if update_timestep is not None else control_timestep

        # Spaces pass through unchanged (no re-keying / re-batching).
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.context_space = env.context_space
        self.has_reward = bool(expose_reward)
        self.has_termination_signal = bool(expose_termination)
        self.has_truncation_signal = bool(expose_truncation)
        self.render_mode = env.render_mode
        self.supported_render_modes = tuple(env.metadata.get("render_modes", ()))

        # Instance-level priority override so users can order multiple env-nodes.
        self.pre_environment_step_priorities = {pre_step_priority}

        self._next_action: Optional[ActType] = None
        self._last_step_result: Optional[Tuple[Any, Any, Any, Any, Dict[str, Any]]] = None
        self._reset_context: Optional[ContextType] = None
        self._reset_obs: Optional[ObsType] = None
        self._reset_info: Optional[Dict[str, Any]] = None
        self._closed = False

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        """Backend of the attached world, falling back to the inner env's backend."""
        if self.world is not None:
            return self.world.backend
        return self.env.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        """Device of the attached world, falling back to the inner env's device."""
        if self.world is not None:
            return self.world.device
        return self.env.device

    # ========== Lifecycle (step flow) ==========

    def set_next_action(self, action: ActType) -> None:
        """Cache the action consumed by the next ``pre_environment_step`` call.

        ``action=None`` is only accepted when ``hold_last_action=True`` (the
        previously cached action is kept); in that case a node that has never
        received an action still raises from ``pre_environment_step``.
        """
        if action is None:
            if self._hold_last_action:
                # Keep whatever was cached before (possibly nothing).
                return
            raise ValueError(
                "EnvAsWorldNode requires an action every step unless hold_last_action=True"
            )
        if self.action_space is not None and not self.action_space.contains(action):
            raise ValueError(
                f"Action {action!r} is not contained in the action space "
                f"{self.action_space!r}."
            )
        self._next_action = action

    def pre_environment_step(self, dt: Union[float, BArrayType], *, priority: int = 0) -> None:
        """Run one atomic ``Env.step`` with the cached action.

        The world-provided ``dt`` is ignored: the wrapped environment is
        self-paced, so it advances by exactly one of its own steps per call.
        """
        action = self._next_action
        if action is None:
            raise RuntimeError(
                f"EnvAsWorldNode '{self.name}' has no cached action to step with. "
                f"Call set_next_action() with an action before pre_environment_step()."
            )
        if not self._hold_last_action:
            self._next_action = None

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_step_result = (obs, reward, terminated, truncated, info)

    # ========== Lifecycle (reset / reload flow) ==========

    def reset(
        self,
        *,
        priority: int = 0,
        seed: Optional[int] = None,
        mask: Optional[BArrayType] = None,
        **kwargs,
    ) -> None:
        """Reset the inner environment (also used as the default ``reload``)."""
        assert mask is None, (
            "EnvAsWorldNode wraps an unbatched environment; masked resets are not supported."
        )
        context, obs, info = self.env.reset(
            seed=seed if seed is not None else self.reset_seed,
            **kwargs,
        )
        self._reset_context = context
        self._reset_obs = obs
        self._reset_info = info
        self._last_step_result = None
        self._next_action = None

    # ========== Accessors ==========

    def get_context(self) -> Optional[ContextType]:
        """Return the context produced by the most recent inner reset."""
        return self._reset_context

    def get_observation(self) -> ObsType:
        """Return the latest step observation, or the reset observation before the first step."""
        if self._last_step_result is not None:
            return self._last_step_result[0]
        return self._reset_obs

    def get_reward(self) -> Union[float, BArrayType]:
        """Return the cached step reward (``0.0`` before the first step)."""
        if self._last_step_result is not None:
            return self._last_step_result[1]
        return 0.0

    def get_termination(self) -> Union[bool, BArrayType]:
        """Return the cached termination signal (``False`` before the first step)."""
        if self._last_step_result is not None:
            return self._last_step_result[2]
        return False

    def get_truncation(self) -> Union[bool, BArrayType]:
        """Return the cached truncation signal (``False`` before the first step)."""
        if self._last_step_result is not None:
            return self._last_step_result[3]
        return False

    def get_info(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the latest info dict when ``forward_info`` is enabled."""
        if not self._forward_info:
            return None
        info = self._last_step_result[4] if self._last_step_result is not None else self._reset_info
        return dict(info) if info is not None else None

    def render(self):
        """Render the inner environment."""
        return self.env.render()

    def close(self) -> None:
        """Close the inner environment once (idempotent, ``__del__``-safe)."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        self.env.close()
