"""ReplayBufferCollectionWrapper – transparently records transitions into replay buffers.

This wrapper sits around any ``Env`` and writes every *step* transition into one
or more ``ReplayBuffer`` instances according to the chosen ``replay_mode``.
"""

from __future__ import annotations

import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    Union,
)

import numpy as np

from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.space import DictSpace, Space
from unienv_interface.transformations import DataTransformation

from .replay_buffer import ReplayBuffer


PathLike = Union[str, "os.PathLike[str]"]


# ===================================================================
# ReplayBufferCollectionWrapper
# ===================================================================


class ReplayBufferCollectionWrapper(
    Wrapper[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
    ]
):
    """Automatically record every transition into one or more replay buffers.

    Parameters
    ----------
    env : Env
        The environment to wrap.
    replay_buffers : ReplayBuffer | Sequence[ReplayBuffer]
        One replay buffer (shared mode) or one per environment slot (per‑env
        mode).
    replay_mode : ``"auto"``, ``"shared"`` or ``"per_env"``
        * ``"shared"`` - a single flat replay buffer.
        * ``"per_env"`` - one buffer per batched slot, with episode segments.
        * ``"auto"`` - resolve based on *batch_size* and buffer count.
    transition_builder : callable or DataTransformation, optional
        Either a callable ``(cached_obs, cached_context, action, next_obs,
        reward, terminated, truncated, env_index, rb_single_space) ->
        transition``, or a ``DataTransformation`` that maps a canonical
        raw transition dict to the replay buffer's format.  When ``None``
        the default DictSpace‑based builder is used.
    validate_transition : bool
        When ``True`` every built transition is validated against the target
        replay buffer's space (single‑instance for append / batched for
        shared ``extend``).  Off by default for speed.
    dump_on_episode_end : bool
        When ``True``, automatically call ``dumps(path)`` on the replay
        buffer(s) after an episode ends (i.e., when ``terminated or truncated``
        is true).  Requires ``dump_path`` to be set.
    dump_on_reset_boundary : bool
        When ``True``, automatically call ``dumps(path)`` on reset boundaries
        (full reset or masked reset).  Requires ``dump_path`` to be set.
    dump_path : str | os.PathLike | None
        Filesystem path where the replay buffer(s) should be dumped when
        auto-dump is triggered.  May be reassigned after construction to
        redirect subsequent dumps.  Required if ``dump_on_episode_end`` or
        ``dump_on_reset_boundary`` is ``True``.  Repeated dumps overwrite
        the same path.
    """

    # ---- helper constructors ------------------------------------------------

    @classmethod
    def shared(
        cls,
        env: Env,
        replay_buffer: ReplayBuffer,
        **kwargs: Any,
    ) -> "ReplayBufferCollectionWrapper":
        """Create a wrapper in **shared** mode with a single replay buffer."""
        return cls(env, replay_buffer, replay_mode="shared", **kwargs)

    @classmethod
    def per_env(
        cls,
        env: Env,
        replay_buffers: Sequence[ReplayBuffer],
        **kwargs: Any,
    ) -> "ReplayBufferCollectionWrapper":
        """Create a wrapper in **per‑env** mode with one buffer per slot."""
        return cls(env, replay_buffers, replay_mode="per_env", **kwargs)

    # ---- init ---------------------------------------------------------------

    def __init__(
        self,
        env: Env[
            BArrayType, ContextType, ObsType, ActType, RenderFrame,
            BDeviceType, BDtypeType, BRNGType,
        ],
        replay_buffers: Union[ReplayBuffer, Sequence[ReplayBuffer]],
        replay_mode: Literal["auto", "shared", "per_env"] = "auto",
        transition_builder: Optional[Union[Callable[..., Any], DataTransformation]] = None,
        validate_transition: bool = False,
        dump_on_episode_end: bool = False,
        dump_on_reset_boundary: bool = False,
        dump_path: Optional[PathLike] = None,
    ) -> None:
        super().__init__(env)

        self._replay_buffers: List[ReplayBuffer] = self._normalize_replay_buffers(
            replay_buffers
        )
        self._replay_mode: Literal["shared", "per_env"] = self._resolve_replay_mode(
            replay_mode
        )
        self._validate_mode_layout()

        self._transition_builder_is_transform: bool = isinstance(
            transition_builder, DataTransformation
        )
        self._transition_builder: Union[Callable[..., Any], DataTransformation] = (
            transition_builder or self._default_transition_from_space
        )
        self._validate_transition_flag: bool = validate_transition

        # Auto-dump configuration
        self._dump_on_episode_end: bool = dump_on_episode_end
        self._dump_on_reset_boundary: bool = dump_on_reset_boundary
        self.dump_path: Optional[PathLike] = dump_path

        # Validate dump configuration
        if (dump_on_episode_end or dump_on_reset_boundary) and dump_path is None:
            raise ValueError(
                "dump_path must be provided when dump_on_episode_end or "
                "dump_on_reset_boundary is True."
            )

        # Per-step caches
        self._cached_context: Optional[ContextType] = None
        self._cached_obs: Optional[ObsType] = None
        self._initialized: bool = False

    # ===================================================================
    # Core lifecycle
    # ===================================================================

    def reset(
        self,
        *args: Any,
        mask: Optional[BArrayType] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        context, obs, info = self.env.reset(*args, mask=mask, seed=seed, **kwargs)

        # Mark segment boundaries in per‑env mode before caching the new state.
        if self._replay_mode == "per_env" and self._initialized:
            self._mark_reset_boundaries(mask)

        # Handle reset-boundary dumping for shared mode (unbatched or B=1).
        # For per_env mode, dumping is handled in _mark_reset_boundaries.
        if (
            self._replay_mode == "shared"
            and self._dump_on_reset_boundary
            and self._initialized
        ):
            self._dump_shared_on_reset_boundary()

        # Merge compact reset results for masked batched resets.
        if mask is not None and self._initialized:
            if self._cached_context is not None:
                context = self.env.update_context_post_reset(
                    self._cached_context, context, mask
                )
            if self._cached_obs is not None:
                obs = self.env.update_observation_post_reset(
                    self._cached_obs, obs, mask
                )

        self._cached_context = context
        self._cached_obs = obs
        self._initialized = True
        return context, obs, info

    def step(
        self,
        action: ActType,
    ) -> Tuple[
        ObsType,
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
        Dict[str, Any],
    ]:
        self._require_initialized()

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._record_transition(
            prev_obs=self._cached_obs,
            prev_context=self._cached_context if self.context_space is not None else None,
            action=action,
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        self._cached_obs = obs
        return obs, reward, terminated, truncated, info

    # ---- async bridge (when supported by the wrapped env) ----------------

    def reset_async(
        self,
        *args: Any,
        mask: Optional[BArrayType] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if not hasattr(self.env, "reset_async"):
            raise AttributeError("The wrapped env does not support reset_async")
        # Segment boundaries are handled in reset_wait where we have the results.
        self.env.reset_async(*args, mask=mask, seed=seed, **kwargs)
        # Stash the mask so reset_wait can merge.
        self._async_reset_mask: Optional[BArrayType] = mask

    def reset_wait(
        self,
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        if not hasattr(self.env, "reset_wait"):
            raise AttributeError("The wrapped env does not support reset_wait")

        mask = getattr(self, "_async_reset_mask", None)
        context, obs, info = self.env.reset_wait()

        if self._replay_mode == "per_env" and self._initialized:
            self._mark_reset_boundaries(mask)

        # Handle reset-boundary dumping for shared mode (unbatched or B=1).
        # For per_env mode, dumping is handled in _mark_reset_boundaries.
        if (
            self._replay_mode == "shared"
            and self._dump_on_reset_boundary
            and self._initialized
        ):
            self._dump_shared_on_reset_boundary()

        if mask is not None and self._initialized:
            if self._cached_context is not None:
                context = self.env.update_context_post_reset(
                    self._cached_context, context, mask
                )
            if self._cached_obs is not None:
                obs = self.env.update_observation_post_reset(
                    self._cached_obs, obs, mask
                )

        self._cached_context = context
        self._cached_obs = obs
        self._initialized = True
        self._async_reset_mask = None
        return context, obs, info

    def step_async(self, action: ActType) -> None:
        if not hasattr(self.env, "step_async"):
            raise AttributeError("The wrapped env does not support step_async")
        self.env.step_async(action)
        # Stash the action for recording after step_wait.
        self._async_action: ActType = action

    def step_wait(
        self,
    ) -> Tuple[
        ObsType,
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
        Dict[str, Any],
    ]:
        if not hasattr(self.env, "step_wait"):
            raise AttributeError("The wrapped env does not support step_wait")

        self._require_initialized()
        action: ActType = self._async_action

        obs, reward, terminated, truncated, info = self.env.step_wait()
        self._record_transition(
            prev_obs=self._cached_obs,
            prev_context=self._cached_context if self.context_space is not None else None,
            action=action,
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        self._cached_obs = obs
        self._async_action = None  # type: ignore[assignment]
        return obs, reward, terminated, truncated, info

    # ===================================================================
    # Public helpers
    # ===================================================================

    @property
    def replay_buffers(self) -> List[ReplayBuffer]:
        """The list of managed replay buffers."""
        return self._replay_buffers

    @property
    def replay_mode(self) -> str:
        """The resolved replay mode (``"shared"`` or ``"per_env"``)."""
        return self._replay_mode

    # ===================================================================
    # Internal – recording dispatch
    # ===================================================================

    def _record_transition(
        self,
        prev_obs: ObsType,
        prev_context: Optional[ContextType],
        action: ActType,
        obs: ObsType,
        reward: Union[SupportsFloat, BArrayType],
        terminated: Union[bool, BArrayType],
        truncated: Union[bool, BArrayType],
    ) -> None:
        """Build and write transitions according to the resolved mode."""
        if self._replay_mode == "shared":
            self._record_shared(
                prev_obs=prev_obs,
                prev_context=prev_context,
                action=action,
                obs=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
        else:
            self._record_per_env(
                prev_obs=prev_obs,
                prev_context=prev_context,
                action=action,
                obs=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

    def _record_shared(
        self,
        prev_obs: ObsType,
        prev_context: Optional[ContextType],
        action: ActType,
        obs: ObsType,
        reward: Union[SupportsFloat, BArrayType],
        terminated: Union[bool, BArrayType],
        truncated: Union[bool, BArrayType],
    ) -> None:
        rb = self._replay_buffers[0]
        B = self.batch_size

        if B is None:
            # Single, non-batched env – append one transition.
            transition = self._build_transition(
                cached_obs=prev_obs,
                cached_context=prev_context,
                action=action,
                next_obs=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                env_index=None,
                rb_single_space=rb.single_space,
            )
            self._validate(transition, rb.single_space)
            rb.append(transition)

            # Check if episode ended
            done = self._done(terminated, truncated)
            if done:
                rb.mark_segment_end()
                if self._dump_on_episode_end:
                    self._dump_buffer(rb)

        elif B == 1:
            # Batched with one slot – unbatch slot 0 and append.
            term_0 = self._unbatch_scalar(terminated, 0)
            trunc_0 = self._unbatch_scalar(truncated, 0)
            transition = self._build_transition(
                cached_obs=self._unbatch_generic(self.observation_space, prev_obs, 0),
                cached_context=(
                    self._unbatch_context(prev_context, 0)
                    if prev_context is not None
                    else None
                ),
                action=self._unbatch_generic(self.action_space, action, 0),
                next_obs=self._unbatch_generic(self.observation_space, obs, 0),
                reward=self._unbatch_scalar(reward, 0),
                terminated=term_0,
                truncated=trunc_0,
                env_index=None,
                rb_single_space=rb.single_space,
            )
            self._validate(transition, rb.single_space)
            rb.append(transition)

            # Check if episode ended
            done = self._done(term_0, trunc_0)
            if done:
                rb.mark_segment_end()
                if self._dump_on_episode_end:
                    self._dump_buffer(rb)

        else:
            # B > 1 – use extend for the whole batch.
            # In shared batched mode env_index is a batched field of shape
            # (B,) so that downstream builders / transforms can record which
            # env slot each transition came from.
            env_index_batched: Any = self.backend.arange(
                B, dtype=self.backend.default_integer_dtype,
            )
            transition = self._build_transition(
                cached_obs=prev_obs,
                cached_context=prev_context,
                action=action,
                next_obs=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                env_index=env_index_batched,
                rb_single_space=rb.single_space,
            )
            # In B > 1 shared-extend mode all builders (callable, default,
            # and DataTransformation) produce batched transitions, so
            # validate against the *batched* replay-buffer space.
            target_space = sbu.batch_space(rb.single_space, B)
            self._validate(transition, target_space)
            rb.extend(transition)

            # B > 1 shared mode: flat transition replay, no segment semantics.
            # If any slot is done, dump once for the shared buffer.
            if self._dump_on_episode_end:
                done_any = self._done_any_batched(terminated, truncated)
                if done_any:
                    self._dump_buffer(rb)

    def _record_per_env(
        self,
        prev_obs: ObsType,
        prev_context: Optional[ContextType],
        action: ActType,
        obs: ObsType,
        reward: Union[SupportsFloat, BArrayType],
        terminated: Union[bool, BArrayType],
        truncated: Union[bool, BArrayType],
    ) -> None:
        B = self.batch_size
        assert B is not None and B > 1, "per_env mode requires batch_size > 1"

        for i in range(B):
            rb = self._replay_buffers[i]
            term_i = self._unbatch_scalar(terminated, i)
            trunc_i = self._unbatch_scalar(truncated, i)
            transition = self._build_transition(
                cached_obs=self._unbatch_generic(self.observation_space, prev_obs, i),
                cached_context=(
                    self._unbatch_context(prev_context, i)
                    if prev_context is not None
                    else None
                ),
                action=self._unbatch_generic(self.action_space, action, i),
                next_obs=self._unbatch_generic(self.observation_space, obs, i),
                reward=self._unbatch_scalar(reward, i),
                terminated=term_i,
                truncated=trunc_i,
                env_index=i,
                rb_single_space=rb.single_space,
            )
            self._validate(transition, rb.single_space)
            rb.append(transition)

            done = self._done(term_i, trunc_i)
            if done:
                rb.mark_segment_end()
                if self._dump_on_episode_end:
                    self._dump_buffer(rb)

    # ===================================================================
    # Internal – transition building
    # ===================================================================

    def _build_transition(
        self,
        *,
        cached_obs: ObsType,
        cached_context: Optional[ContextType],
        action: ActType,
        next_obs: ObsType,
        reward: Union[SupportsFloat, BArrayType],
        terminated: Union[bool, BArrayType],
        truncated: Union[bool, BArrayType],
        env_index: Union[None, int, BArrayType],
        rb_single_space: Space,
    ) -> Any:
        """Build a transition using the configured builder (callable or
        ``DataTransformation``)."""
        if self._transition_builder_is_transform:
            canonical = self._build_canonical_transition_dict(
                cached_obs=cached_obs,
                cached_context=cached_context,
                action=action,
                next_obs=next_obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                env_index=env_index,
                rb_single_space=rb_single_space,
            )
            # Non-DictSpace fallback – skip transformation, return as-is.
            if not isinstance(canonical, dict):
                return canonical

            # Determine batching by inspecting the (potentially batched)
            # observation data, not the canonical dict.
            is_batched = sbu.batch_size_data(cached_obs) is not None
            source_space = self._build_canonical_source_space(
                rb_single_space, is_batched=is_batched
            )
            transform = self._transition_builder  # type: DataTransformation
            return transform.transform(source_space, canonical)

        # Callable path.
        builder = self._transition_builder  # type: Callable[..., Any]
        return builder(
            cached_obs=cached_obs,
            cached_context=cached_context,
            action=action,
            next_obs=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            env_index=env_index,
            rb_single_space=rb_single_space,
        )

    # Canonical keys that the default builder and DataTransformation path can
    # supply from the env transition/context/config.
    _SUPPORTED_CANONICAL_KEYS = frozenset({
        "obs", "next_obs", "action", "reward",
        "terminated", "truncated", "done",
        "context", "env_index",
    })

    def _build_canonical_transition_dict(
        self,
        *,
        cached_obs: Any,
        cached_context: Optional[Any],
        action: Any,
        next_obs: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        env_index: Union[None, int, BArrayType],
        rb_single_space: Space,
    ) -> Dict[str, Any]:
        """Return a canonical raw transition dict for use with a
        ``DataTransformation`` builder.

        Only includes keys that are present in *rb_single_space* (when it
        is a ``DictSpace``).  Scalar values (reward, terminated, …) are
        coerced to match the corresponding subspace dtype so that identity
        transforms produce valid transitions out of the box.

        Raises ``KeyError`` if the space requests a key that cannot be
        supplied from the current env transition/context/config.

        Falls back to returning *next_obs* unchanged for non-DictSpace
        targets.
        """
        if not isinstance(rb_single_space, DictSpace):
            return next_obs  # type: ignore[return-value]

        result: Dict[str, Any] = {}
        for key, subspace in rb_single_space.spaces.items():
            if key == "obs":
                result[key] = cached_obs
            elif key == "next_obs":
                result[key] = next_obs
            elif key == "action":
                result[key] = action
            elif key == "reward":
                result[key] = self._coerce_scalar(reward, subspace)
            elif key == "terminated":
                result[key] = self._coerce_scalar(terminated, subspace)
            elif key == "truncated":
                result[key] = self._coerce_scalar(truncated, subspace)
            elif key == "done":
                done_val = self._done(terminated, truncated)
                result[key] = self._coerce_scalar(done_val, subspace)
            elif key == "context":
                if cached_context is None:
                    raise KeyError(
                        f"Replay buffer space requests key 'context' but the "
                        f"environment has no context_space (context_space is None). "
                        f"Remove 'context' from the replay buffer space or use an "
                        f"environment that provides context."
                    )
                result[key] = cached_context
            elif key == "env_index":
                if env_index is None:
                    raise KeyError(
                        f"Replay buffer space requests key 'env_index' but it is "
                        f"not available in '{self._replay_mode}' replay mode with "
                        f"batch_size={self.batch_size}. "
                        f"'env_index' is available in 'per_env' mode and in "
                        f"'shared' mode when batch_size > 1. "
                        f"Remove 'env_index' from the replay buffer space or "
                        f"switch to per_env mode."
                    )
                result[key] = self._coerce_scalar(env_index, subspace)
            else:
                raise KeyError(
                    f"Replay buffer space requests unknown key '{key}' which "
                    f"cannot be supplied by the default builder or "
                    f"DataTransformation canonical path. Supported keys: "
                    f"{sorted(self._SUPPORTED_CANONICAL_KEYS)}. "
                    f"Use a custom callable transition_builder to handle "
                    f"non-canonical keys."
                )
        return result

    def _build_canonical_source_space(
        self,
        rb_single_space: Space,
        *,
        is_batched: bool = False,
    ) -> Space:
        """Build the source ``DictSpace`` that matches the canonical
        transition dict for the given batching status.

        Only includes subspace keys that are present in *rb_single_space*.

        Env-side spaces (``obs``, ``next_obs``, ``action``, ``context``)
        are used **as-is** from the wrapped environment.  When the env is
        batched those spaces are already batched, so we must NOT apply
        ``batch_space`` on top of them (that would double-batch).

        Scalar / flag fields (``reward``, ``terminated``, ``truncated``,
        ``done``) are derived from the replay-buffer subspace template:
        for an unbatched env they stay scalar; for a batched env (when the
        canonical data carries a batch dimension, i.e. *is_batched* is
        ``True``) they are batched by the environment's ``batch_size`` to
        match the backend-array outputs.
        """
        backend = self.backend
        B = self.batch_size

        # Determine which canonical keys to include.
        if not isinstance(rb_single_space, DictSpace):
            return rb_single_space

        rb_subspaces = rb_single_space.spaces

        # --- env-side subspaces (obs, action, context) ---
        # Use the env spaces as-is.  If the env is batched these are
        # already batched; if unbatched they are single-instance spaces.
        # We must NOT call batch_space() on them – that would double-batch
        # when the env is already batched.
        obs_space = self.observation_space
        act_space = self.action_space

        spaces: Dict[str, Space] = {}

        if "obs" in rb_subspaces:
            spaces["obs"] = obs_space
        if "next_obs" in rb_subspaces:
            spaces["next_obs"] = obs_space
        if "action" in rb_subspaces:
            spaces["action"] = act_space

        # Scalar fields – derive shape/dtype from the rb subspace, but
        # add a batch dimension when the canonical data is batched
        # (shared B>1 mode).  For unbatched envs or unbatched data
        # (shared B=1, per_env) the scalar subspace is used as-is.
        for scalar_key in ("reward", "terminated", "truncated", "done"):
            if scalar_key not in rb_subspaces:
                continue
            sub = rb_subspaces[scalar_key]
            if is_batched and B is not None and B > 0:
                spaces[scalar_key] = sbu.batch_space(sub, B)
            else:
                spaces[scalar_key] = sub

        # --- optional context ---
        if "context" in rb_subspaces:
            if self.context_space is None:
                raise KeyError(
                    f"Replay buffer space requests key 'context' but the "
                    f"environment has no context_space (context_space is None). "
                    f"Remove 'context' from the replay buffer space or use an "
                    f"environment that provides context."
                )
            # Use the env context_space as-is (already batched if env is batched).
            spaces["context"] = self.context_space

        # --- optional env_index ---
        if "env_index" in rb_subspaces:
            # env_index is available in per_env mode (scalar per-slot) and
            # in shared mode when batch_size > 1 (batched field of shape (B,)).
            # It is NOT available in shared mode with batch_size None or 1.
            if self._replay_mode == "per_env":
                sub = rb_subspaces["env_index"]
                # In per_env mode the data is always unbatched (per-slot), so
                # env_index is a scalar.
                spaces["env_index"] = sub
            elif (
                self._replay_mode == "shared"
                and B is not None
                and B > 1
                and is_batched
            ):
                sub = rb_subspaces["env_index"]
                spaces["env_index"] = sbu.batch_space(sub, B)
            else:
                raise KeyError(
                    f"Replay buffer space requests key 'env_index' but it is "
                    f"not available in '{self._replay_mode}' replay mode with "
                    f"batch_size={B}. "
                    f"'env_index' is available in 'per_env' mode and in "
                    f"'shared' mode when batch_size > 1. "
                    f"Remove 'env_index' from the replay buffer space or "
                    f"switch to per_env mode."
                )

        return DictSpace(backend, spaces)

    def _default_transition_from_space(
        self,
        *,
        cached_obs: Any,
        cached_context: Optional[Any],
        action: Any,
        next_obs: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        env_index: Union[None, int, BArrayType],
        rb_single_space: Space,
    ) -> Any:
        """Default DictSpace‑oriented transition builder.

        Populates the canonical keys that exist in the replay buffer's
        single‑instance ``DictSpace``, converting scalar values (reward,
        terminated, etc.) to match the space's dtype when needed.

        Raises ``KeyError`` if the space requests a key that cannot be
        supplied from the current env transition/context/config (e.g.
        ``context`` when the env has no context space, ``env_index`` in
        shared mode with batch_size None or 1, or any unknown key).

        If the space is not a ``DictSpace`` the raw *next_obs* is returned
        unchanged as a fallback.
        """
        if not isinstance(rb_single_space, DictSpace):
            return next_obs

        result: Dict[str, Any] = {}

        for key, subspace in rb_single_space.spaces.items():
            if key == "obs":
                result[key] = cached_obs
            elif key == "action":
                result[key] = action
            elif key == "reward":
                result[key] = self._coerce_scalar(reward, subspace)
            elif key == "terminated":
                result[key] = self._coerce_scalar(terminated, subspace)
            elif key == "truncated":
                result[key] = self._coerce_scalar(truncated, subspace)
            elif key == "done":
                done_val = self._done(terminated, truncated)
                result[key] = self._coerce_scalar(done_val, subspace)
            elif key == "next_obs":
                result[key] = next_obs
            elif key == "context":
                if cached_context is None:
                    raise KeyError(
                        f"Replay buffer space requests key 'context' but the "
                        f"environment has no context_space (context_space is None). "
                        f"Remove 'context' from the replay buffer space or use an "
                        f"environment that provides context."
                    )
                result[key] = cached_context
            elif key == "env_index":
                if env_index is None:
                    raise KeyError(
                        f"Replay buffer space requests key 'env_index' but it is "
                        f"not available in '{self._replay_mode}' replay mode with "
                        f"batch_size={self.batch_size}. "
                        f"'env_index' is available in 'per_env' mode and in "
                        f"'shared' mode when batch_size > 1. "
                        f"Remove 'env_index' from the replay buffer space or "
                        f"switch to per_env mode."
                    )
                result[key] = self._coerce_scalar(env_index, subspace)
            else:
                raise KeyError(
                    f"Replay buffer space requests unknown key '{key}' which "
                    f"cannot be supplied by the default builder. Supported keys: "
                    f"{sorted(self._SUPPORTED_CANONICAL_KEYS)}. "
                    f"Use a custom callable transition_builder to handle "
                    f"non-canonical keys."
                )

        return result

    @staticmethod
    def _coerce_scalar(value: Any, space: Space) -> Any:
        """Convert a scalar (float, bool, array) to match the space's dtype."""
        from unienv_interface.space.spaces.box import BoxSpace

        if not isinstance(space, BoxSpace):
            return value

        target_dtype = space.dtype

        # Python scalar or numpy scalar – cast to target dtype
        return space.backend.asarray(value, dtype=target_dtype)

    # ===================================================================
    # Internal – helpers
    # ===================================================================

    @staticmethod
    def _normalize_replay_buffers(
        replay_buffers: Union[ReplayBuffer, Sequence[ReplayBuffer]],
    ) -> List[ReplayBuffer]:
        if isinstance(replay_buffers, ReplayBuffer):
            return [replay_buffers]
        return list(replay_buffers)

    def _resolve_replay_mode(
        self, mode: Literal["auto", "shared", "per_env"]
    ) -> Literal["shared", "per_env"]:
        if mode != "auto":
            return mode

        B = self.batch_size
        K = len(self._replay_buffers)

        if K == 1:
            if B is None or B == 1:
                return "shared"
            if B > 1:
                if self._has_segment_metadata(self._replay_buffers[0]):
                    raise ValueError(
                        f"Cannot auto-resolve replay mode: batch_size={B}, "
                        f"K=1 but the replay buffer is segment-aware. "
                        f"Use per_env mode with K={B} buffers or a "
                        f"non-segment-aware buffer."
                    )
                return "shared"

        if B is not None and B > 1 and K == B:
            return "per_env"

        raise ValueError(
            f"Cannot auto-resolve replay mode: batch_size={B}, K={K}. "
            f"Specify replay_mode explicitly or adjust the layout."
        )

    def _validate_mode_layout(self) -> None:
        B = self.batch_size
        K = len(self._replay_buffers)

        if self._replay_mode == "shared":
            if K != 1:
                raise ValueError(
                    f"Shared mode requires exactly 1 replay buffer, got {K}."
                )
            if B is not None and B > 1:
                if self._has_segment_metadata(self._replay_buffers[0]):
                    raise ValueError(
                        f"Shared mode with batch_size={B} > 1 requires a "
                        f"segment-unaware replay buffer."
                    )

        elif self._replay_mode == "per_env":
            if B is None:
                raise ValueError(
                    "Per-env mode requires a batched environment (batch_size > 1)."
                )
            if B == 1:
                raise ValueError(
                    "Per-env mode requires batch_size > 1, got batch_size=1."
                )
            if K != B:
                raise ValueError(
                    f"Per-env mode requires exactly one replay buffer per "
                    f"environment slot: got {K} buffers for batch_size={B}."
                )

    @staticmethod
    def _has_segment_metadata(rb: ReplayBuffer) -> bool:
        """Return ``True`` if the buffer maintains segment information."""
        if rb.maintain_segment_metadata:
            return True
        segments = rb.get_segments()
        return segments is not None

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "Cannot call step() before reset(). "
                "Call reset() first to initialize the environment."
            )

    def _mark_reset_boundaries(self, mask: Optional[BArrayType]) -> None:
        """Close per-env segments for slots that are being reset."""
        B = self.batch_size
        if B is None:
            return
        for i in range(B):
            if mask is None or bool(mask[i]):
                rb = self._replay_buffers[i]
                # Close any open segment before starting a new episode.
                if rb.storage is not None and rb.storage.has_open_segment:
                    rb.mark_segment_end()
                    # Dump on reset boundary if enabled (per_env mode).
                    if self._dump_on_reset_boundary:
                        self._dump_buffer(rb)

    def _done(
        self,
        terminated: Union[bool, BArrayType],
        truncated: Union[bool, BArrayType],
    ) -> Union[bool, BArrayType]:
        """Combine terminated and truncated into a single boolean."""
        if self.backend.is_backendarray(terminated):
            return self.backend.logical_or(terminated, truncated)  # type: ignore[arg-type]
        return bool(terminated) or bool(truncated)

    @staticmethod
    def _unbatch_generic(space: Space, data: Any, index: int) -> Any:
        """Extract slot *index* from batched data."""
        return sbu.get_at(space, data, index)

    def _unbatch_context(self, context: ContextType, index: int) -> Any:
        """Extract slot *index* from a potentially-None context batch."""
        if context is None or self.context_space is None:
            return None
        return sbu.get_at(self.context_space, context, index)

    def _unbatch_scalar(
        self,
        value: Union[SupportsFloat, BArrayType, bool],
        index: int,
    ) -> Any:
        """Extract slot *index* from a batched scalar (reward / done flags).

        In an unbatched environment reward, terminated and truncated are
        always plain scalars.  In a batched environment they are always
        backend arrays, so simple indexing is sufficient.
        """
        if self.batch_size is None:
            return value
        return value[index]  # type: ignore[index]

    def _validate(self, transition: Any, target_space: Space) -> None:
        """Optionally validate a built transition against *target_space*."""
        if not self._validate_transition_flag:
            return
        if not target_space.contains(transition):
            raise ValueError(
                f"Built transition does not match target space. "
                f"Space: {target_space}, Transition keys: "
                f"{list(transition.keys()) if isinstance(transition, dict) else type(transition)}"
            )

    # ===================================================================
    # Internal – auto-dump helpers
    # ===================================================================

    def _dump_buffer(self, rb: ReplayBuffer) -> None:
        """Dump a replay buffer to the configured ``dump_path``.

        This method assumes that any open segment has already been closed
        (via ``mark_segment_end``) before calling this method.
        """
        if self.dump_path is None:
            raise RuntimeError(
                "Cannot dump replay buffer: dump_path is not set. "
                "Provide dump_path when enabling dump_on_episode_end or "
                "dump_on_reset_boundary."
            )
        rb.dumps(self.dump_path)

    def _dump_shared_on_reset_boundary(self) -> None:
        """Dump the shared replay buffer on a reset boundary.

        This is called for shared mode (unbatched or B=1) when
        ``dump_on_reset_boundary`` is enabled.  For shared B>1 mode, this
        represents flat-buffer persistence, not per-episode dumping.
        """
        rb = self._replay_buffers[0]
        # Check if there's an open segment to close (for segment-aware buffers).
        # For shared B>1 mode, buffers are segment-unaware, so this check is safe.
        has_open = (
            rb.storage is not None
            and rb.storage.has_open_segment
        )
        if has_open:
            rb.mark_segment_end()

        # Only dump if there's data to dump
        if len(rb) > 0:
            self._dump_buffer(rb)

    def _done_any_batched(
        self,
        terminated: Union[bool, BArrayType],
        truncated: Union[bool, BArrayType],
    ) -> bool:
        """Check if any slot in a batched done signal is True."""
        done = self._done(terminated, truncated)
        if self.backend.is_backendarray(done):
            return bool(self.backend.any(done))
        return bool(done)

    # ===================================================================
    # Public – replay buffer swapping
    # ===================================================================

    def set_replay_buffers(
        self,
        replay_buffers: Union[ReplayBuffer, Sequence[ReplayBuffer]],
        *,
        replay_mode: Optional[Literal["auto", "shared", "per_env"]] = None,
        require_same_space: bool = True,
        allow_mid_episode: bool = False,
        finalize_old_segments: bool = False,
        dump_old: bool = False,
    ) -> List[ReplayBuffer]:
        """Swap the replay buffer(s) without rebuilding the wrapper.

        Parameters
        ----------
        replay_buffers : ReplayBuffer | Sequence[ReplayBuffer]
            The new replay buffer(s) to use.
        replay_mode : ``"auto"``, ``"shared"``, ``"per_env"``, or None
            If provided, change the replay mode.  If ``None``, keep the
            current mode.
        require_same_space : bool
            If ``True``, the new buffers must have the same ``single_space``
            as the old buffers.  Default is ``True``.
        allow_mid_episode : bool
            If ``True``, allow swapping even if this splits an episode
            across old/new buffers.  Default is ``False``.
        finalize_old_segments : bool
            If ``True``, close any open segments in the old buffers before
            swapping.  Default is ``False``.
        dump_old : bool
            If ``True``, dump the old buffers before swapping.  Requires
            ``dump_path`` to be set.  Only does this safely after
            segment finalization where needed.  Default is ``False``.

        Returns
        -------
        List[ReplayBuffer]
            The old replay buffer(s) that were replaced.

        Raises
        ------
        ValueError
            If the new buffers don't match the expected layout or space.
        RuntimeError
            If there are open segments and ``allow_mid_episode`` and
            ``finalize_old_segments`` are both ``False``.
        """
        new_buffers = self._normalize_replay_buffers(replay_buffers)
        new_mode = self._resolve_replay_mode(replay_mode) if replay_mode is not None else self._replay_mode
        old_buffers = self._replay_buffers

        # Validate the new layout
        self._validate_swap_layout(new_buffers, new_mode)

        # Validate spaces if required
        if require_same_space:
            self._validate_swap_spaces(old_buffers, new_buffers, new_mode)

        # Handle open segments in old buffers
        has_open_segments = self._has_open_segments(old_buffers, new_mode)

        if has_open_segments:
            if not allow_mid_episode and not finalize_old_segments and not dump_old:
                raise RuntimeError(
                    "Cannot swap replay buffers: old buffers have open segments. "
                    "Set finalize_old_segments=True to close them, "
                    "allow_mid_episode=True to allow mid-episode swap, or "
                    "dump_old=True to dump before swapping."
                )

            # Dump old buffers if requested (before finalization)
            if dump_old:
                # Need to finalize first if there are open segments
                if has_open_segments and not finalize_old_segments:
                    # Temporarily finalize for dumping
                    self._finalize_all_segments(old_buffers, new_mode)
                    self._dump_all_buffers(old_buffers)
                    # Note: we've already finalized, so don't finalize again
                    finalize_old_segments = False
                else:
                    if finalize_old_segments:
                        self._finalize_all_segments(old_buffers, new_mode)
                        finalize_old_segments = False  # Already done
                    self._dump_all_buffers(old_buffers)
            elif finalize_old_segments:
                self._finalize_all_segments(old_buffers, new_mode)

        # Perform the swap
        self._replay_buffers = new_buffers
        if replay_mode is not None:
            self._replay_mode = new_mode

        return old_buffers

    # Alias for convenience
    buffer_swap = set_replay_buffers

    def _validate_swap_layout(
        self,
        new_buffers: List[ReplayBuffer],
        new_mode: Literal["shared", "per_env"],
    ) -> None:
        """Validate that the new buffers match the expected layout."""
        B = self.batch_size
        K = len(new_buffers)

        if new_mode == "shared":
            if K != 1:
                raise ValueError(
                    f"Shared mode requires exactly 1 replay buffer, got {K}."
                )
            if B is not None and B > 1:
                if self._has_segment_metadata(new_buffers[0]):
                    raise ValueError(
                        f"Shared mode with batch_size={B} > 1 requires a "
                        f"segment-unaware replay buffer."
                    )

        elif new_mode == "per_env":
            if B is None:
                raise ValueError(
                    "Per-env mode requires a batched environment (batch_size > 1)."
                )
            if B == 1:
                raise ValueError(
                    "Per-env mode requires batch_size > 1, got batch_size=1."
                )
            if K != B:
                raise ValueError(
                    f"Per-env mode requires exactly one replay buffer per "
                    f"environment slot: got {K} buffers for batch_size={B}."
                )

    def _validate_swap_spaces(
        self,
        old_buffers: List[ReplayBuffer],
        new_buffers: List[ReplayBuffer],
        new_mode: Literal["shared", "per_env"],
    ) -> None:
        """Validate that new buffers have the same single_space as old buffers."""
        if new_mode == "shared":
            old_space = old_buffers[0].single_space
            new_space = new_buffers[0].single_space
            if old_space != new_space:
                raise ValueError(
                    f"New shared replay buffer has different single_space. "
                    f"Old: {old_space}, New: {new_space}"
                )
        elif new_mode == "per_env":
            for i, (old_rb, new_rb) in enumerate(zip(old_buffers, new_buffers)):
                old_space = old_rb.single_space
                new_space = new_rb.single_space
                if old_space != new_space:
                    raise ValueError(
                        f"New per_env replay buffer {i} has different single_space. "
                        f"Old: {old_space}, New: {new_space}"
                    )

    def _has_open_segments(
        self,
        buffers: List[ReplayBuffer],
        mode: Literal["shared", "per_env"],
    ) -> bool:
        """Check if any buffer has an open segment."""
        for rb in buffers:
            if rb.storage is not None and rb.storage.has_open_segment:
                return True
        return False

    def _finalize_all_segments(
        self,
        buffers: List[ReplayBuffer],
        mode: Literal["shared", "per_env"],
    ) -> None:
        """Close all open segments in the buffers."""
        for rb in buffers:
            if rb.storage is not None and rb.storage.has_open_segment:
                rb.mark_segment_end()

    def _dump_all_buffers(self, buffers: List[ReplayBuffer]) -> None:
        """Dump all buffers (used for dump_old during swap)."""
        for rb in buffers:
            # Only dump if there's data
            if len(rb) > 0:
                self._dump_buffer(rb)

# ===================================================================
# Module compatibility
# ===================================================================

__all__ = ["ReplayBufferCollectionWrapper"]
