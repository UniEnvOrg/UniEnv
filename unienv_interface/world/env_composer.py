from typing import Optional, Dict, Any, Tuple, Union, SupportsFloat, Sequence, Iterable

from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType

from .world import World
from .node import WorldNode
from .combined_node import CombinedWorldNode


class WorldEnv(Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]):
    """Composes an ``Env`` from a ``World`` and one or more ``WorldNode`` instances.

    If multiple nodes are passed, they are automatically wrapped in a
    ``CombinedWorldNode``.
    """

    def __init__(
        self,
        world: World[BArrayType, BDeviceType, BDtypeType, BRNGType],
        node_or_nodes: Union[
            WorldNode[ContextType, ObsType, ActType, BArrayType, BDeviceType, BDtypeType, BRNGType],
            Iterable[WorldNode[Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        ],
        *,
        render_mode: Optional[str] = None,
    ):
        if isinstance(node_or_nodes, WorldNode):
            self.node = node_or_nodes
        else:
            self.node = CombinedWorldNode("combined", node_or_nodes, render_mode=render_mode)

        self.world = world

        assert world.is_control_timestep_compatible(self.node.control_timestep), \
            f"Control timestep {self.node.control_timestep} is not compatible " \
            f"with world timestep {world.world_timestep}."

        # Sub-stepping
        if self.node.control_timestep is not None and world.world_timestep is not None:
            self._n_substeps = round(self.node.control_timestep / world.world_timestep)
        else:
            self._n_substeps = 1
        self._control_dt = self.node.control_timestep or world.world_timestep

        self.rng = self.backend.random.random_number_generator(device=self.device)

        self._first_reset = True

    @property
    def observation_space(self):
        return self.node.observation_space

    @property
    def action_space(self):
        return self.node.action_space

    @property
    def context_space(self):
        return self.node.context_space

    @property
    def backend(self):
        return self.world.backend

    @property
    def device(self):
        return self.world.device

    @property
    def batch_size(self):
        return self.world.batch_size

    @property
    def render_mode(self) -> Optional[str]:
        return self.node.render_mode

    @property
    def render_fps(self) -> Optional[int]:
        if self.node.control_timestep is not None:
            return int(round(1 / self.node.control_timestep))
        return None

    # ========== Env interface ==========

    def reload(
        self,
        *,
        mask: Optional[BArrayType] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        self.world.reload(seed=seed, mask=mask, **kwargs)
        for p in sorted(self.node.reload_priorities, reverse=True):
            self.node.reload(priority=p, seed=seed, mask=mask, **kwargs)

        for p in sorted(self.node.after_reset_priorities, reverse=True):
            self.node.after_reset(priority=p, mask=mask)

        context = self.node.get_context() if self.node.context_space is not None else None
        obs = self.node.get_observation() if self.node.observation_space is not None else None
        info = self.node.get_info() or {}
        self._first_reset = False
        return context, obs, info

    def reset(
        self,
        *,
        mask: Optional[BArrayType] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        if self._first_reset:
            return self.reload(seed=seed, mask=mask, **kwargs)

        self.world.reset(seed=seed, mask=mask, **kwargs)
        for p in sorted(self.node.reset_priorities, reverse=True):
            self.node.reset(priority=p, seed=seed, mask=mask, **kwargs)

        # after_reset — call at each priority for side effects
        for p in sorted(self.node.after_reset_priorities, reverse=True):
            self.node.after_reset(priority=p, mask=mask)

        # Read final state
        context = self.node.get_context() if self.node.context_space is not None else None
        obs = self.node.get_observation() if self.node.observation_space is not None else None
        info = self.node.get_info() or {}
        return context, obs, info

    def step(
        self, action: ActType
    ) -> Tuple[
        ObsType,
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
        Dict[str, Any],
    ]:
        # 1. Set action
        if self.node.action_space is not None:
            self.node.set_next_action(action)

        # 2. Pre-environment step (once, before all sub-steps)
        for p in sorted(self.node.pre_environment_step_priorities, reverse=True):
            self.node.pre_environment_step(self._control_dt, priority=p)

        # 3. World sub-steps
        actual_dt = None
        for _ in range(self._n_substeps):
            actual_dt = self.world.step()

        # 4. Post-environment step (once, after all sub-steps)
        post_dt = self._control_dt if self._control_dt is not None else actual_dt
        for p in sorted(self.node.post_environment_step_priorities, reverse=True):
            self.node.post_environment_step(post_dt, priority=p)

        # 5. Read results
        obs = self.node.get_observation() if self.node.observation_space is not None else None

        if self.node.has_reward:
            reward = self.node.get_reward()
        elif self.batch_size is not None:
            reward = self.backend.zeros(
                (self.batch_size,), dtype=self.backend.default_floating_dtype, device=self.device
            )
        else:
            reward = 0.0

        if self.node.has_termination_signal:
            terminated = self.node.get_termination()
        elif self.batch_size is not None:
            terminated = self.backend.zeros(
                (self.batch_size,), dtype=self.backend.default_bool_dtype, device=self.device
            )
        else:
            terminated = False

        if self.node.has_truncation_signal:
            truncated = self.node.get_truncation()
        elif self.batch_size is not None:
            truncated = self.backend.zeros(
                (self.batch_size,), dtype=self.backend.default_bool_dtype, device=self.device
            )
        else:
            truncated = False

        info = self.node.get_info() or {}
        return obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | Sequence[RenderFrame] | None:
        if self.node.can_render:
            return self.node.render()
        return None

    def close(self):
        self.node.close()
        self.world.close()
