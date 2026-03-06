from typing import Optional, Dict, Any, Tuple, Union, SupportsFloat, Sequence, Iterable, Callable, Type

from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType

from .world import World
from .node import WorldNode
from .nodes.combined_node import CombinedWorldNode


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
        render_mode: Optional[str] = 'auto',
    ):
        if isinstance(node_or_nodes, WorldNode):
            self.node = node_or_nodes
        else:
            self.node = CombinedWorldNode("combined", node_or_nodes, render_mode=render_mode)

        self.world = world

        assert world.is_control_timestep_compatible(self.node.control_timestep), \
            f"Control timestep {self.node.control_timestep} is not compatible " \
            f"with world timestep {world.world_timestep}."

        # Two-level sub-stepping: update_timestep sits between world and control
        update_ts = self.node.effective_update_timestep
        control_ts = self.node.control_timestep
        world_ts = world.world_timestep

        assert world.is_control_timestep_compatible(update_ts), \
            f"Update timestep {update_ts} is not compatible with world timestep {world_ts}."

        if control_ts is not None and update_ts is not None:
            self._n_update_substeps = round(control_ts / update_ts)
        else:
            self._n_update_substeps = 1

        if update_ts is not None and world_ts is not None:
            self._n_world_substeps = round(update_ts / world_ts)
        else:
            self._n_world_substeps = 1

        self._update_dt = update_ts or world_ts
        self._control_dt = control_ts or update_ts or world_ts

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

    # ========== Node query methods ==========
    def get_node(
        self,
        nested_keys: Union[str, Sequence[str]],
    ) -> Optional[WorldNode[Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]]:
        return self.node.get_node(nested_keys)

    def get_nodes_by_fn(
        self,
        fn: Callable[[WorldNode[Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]], bool],
    ) -> list[WorldNode[Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]]:
        return self.node.get_nodes_by_fn(fn)

    def get_nodes_by_type(
        self,
        node_type: Type[WorldNode[Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]],
    ) -> list[WorldNode[Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]]:
        return self.node.get_nodes_by_type(node_type)

    # ========== Env interface ==========

    def reload(
        self,
        *,
        mask: Optional[BArrayType] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        """Reload the environment - reload scene with entities from nodes.

        Flow:
            1. world.reload()       - World prepares (clear, load assets)
            2. node.reload()        - Nodes add entities to scene
            3. world.after_reload() - World compiles scene with all entities
            4. node.after_reload()  - Nodes cache entity references
        """
        # 1. World prepares (clear old state, load assets)
        self.world.reload(seed=seed, mask=mask, **kwargs)

        # 2. Nodes add entities to the scene
        for p in sorted(self.node.reload_priorities, reverse=True):
            self.node.reload(priority=p, seed=seed, mask=mask, **kwargs)

        # 3. World compiles scene with all entities added by nodes
        self.world.after_reload(seed=seed, mask=mask, **kwargs)

        # 4. Nodes cache references to entities (now that scene is compiled)
        for p in sorted(self.node.after_reload_priorities, reverse=True):
            self.node.after_reload(priority=p, mask=mask)

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
        reload: bool = False,
        **kwargs,
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        if self._first_reset or reload:
            return self.reload(seed=seed, mask=mask, **kwargs)

        # 1. World reset
        self.world.reset(seed=seed, mask=mask, **kwargs)
        
        # 2. Nodes reset
        for p in sorted(self.node.reset_priorities, reverse=True):
            self.node.reset(priority=p, seed=seed, mask=mask, **kwargs)

        # 3. World post-reset hook
        self.world.after_reset(seed=seed, mask=mask, **kwargs)
        
        # 4. Nodes post-reset hook
        for p in sorted(self.node.after_reset_priorities, reverse=True):
            self.node.after_reset(priority=p, mask=mask)

        # 5. Read final state
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

        # 2. Update-level substep loop
        actual_dt = None
        for _ in range(self._n_update_substeps):
            for p in sorted(self.node.pre_environment_step_priorities, reverse=True):
                self.node.pre_environment_step(self._update_dt, priority=p)
            for _ in range(self._n_world_substeps):
                actual_dt = self.world.step()
            post_dt = self._update_dt if self._update_dt is not None else actual_dt
            for p in sorted(self.node.post_environment_step_priorities, reverse=True):
                self.node.post_environment_step(post_dt, priority=p)

        # 3. Read results
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
                (self.batch_size,), dtype=self.backend.default_boolean_dtype, device=self.device
            )
        else:
            terminated = False

        if self.node.has_truncation_signal:
            truncated = self.node.get_truncation()
        elif self.batch_size is not None:
            truncated = self.backend.zeros(
                (self.batch_size,), dtype=self.backend.default_boolean_dtype, device=self.device
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
