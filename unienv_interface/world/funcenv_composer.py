from typing import Optional, Dict, Any, Tuple, Union, SupportsFloat, Sequence, Iterable
from dataclasses import dataclass

from unienv_interface.env_base.funcenv import FuncEnv, FuncEnvCommonRenderInfo
from unienv_interface.env_base.env import ContextType, ObsType, ActType, RenderFrame
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType

from .funcworld import FuncWorld, WorldStateT
from .funcnode import FuncWorldNode, NodeStateT
from .funcnodes.combined_funcnode import CombinedFuncWorldNode


@dataclass
class WorldFuncEnvState:
	"""State object used by ``FuncWorldEnv``, bundling world and node states."""
	world_state: Any
	node_state: Any


class FuncWorldEnv(FuncEnv[
	WorldFuncEnvState,
	None,  # RenderStateType — rendering not supported
	BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
]):
	"""Composes a ``FuncEnv`` from a ``FuncWorld`` and one or more ``FuncWorldNode`` instances.

	If multiple nodes are passed, they are automatically wrapped in a
	``CombinedFuncWorldNode``.
	"""

	def __init__(
		self,
		world: FuncWorld[WorldStateT, BArrayType, BDeviceType, BDtypeType, BRNGType],
		node_or_nodes: Union[
			FuncWorldNode[WorldStateT, NodeStateT, ContextType, ObsType, ActType, BArrayType, BDeviceType, BDtypeType, BRNGType],
			Iterable[FuncWorldNode[WorldStateT, Any, Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]],
		],
		*,
		render_mode: Optional[str] = 'auto',
	):
		if isinstance(node_or_nodes, FuncWorldNode):
			self.node = node_or_nodes
		else:
			self.node = CombinedFuncWorldNode("combined", node_or_nodes, render_mode=render_mode)

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

	# ========== FuncEnv interface ==========

	def reload(
		self,
		state: WorldFuncEnvState,
		*,
		seed: Optional[int] = None,
		**kwargs,
	) -> Tuple[WorldFuncEnvState, ContextType, ObsType, Dict[str, Any]]:
		# 1. World prepares (clear old state, load assets)
		world_state = self.world.reload(
			state.world_state,
			seed=seed, **kwargs
		)

		# 2. Nodes add entities to the scene
		node_state = None
		for p in sorted(self.node.reload_priorities, reverse=True):
			world_state, ns_p = self.node.reload(world_state, priority=p, seed=seed, **kwargs)
			if isinstance(ns_p, dict) and isinstance(node_state, dict):
				node_state.update(ns_p)
			else:
				node_state = ns_p

		# 3. World compiles scene with all entities
		world_state = self.world.after_reload(
			world_state,
			seed=seed, **kwargs
		)

		# 4. Nodes cache references to entities (now that scene is compiled)
		for p in sorted(self.node.after_reload_priorities, reverse=True):
			world_state, node_state = self.node.after_reload(
				world_state, node_state, priority=p
			)

		context = self.node.get_context(world_state, node_state) if self.node.context_space is not None else None
		obs = self.node.get_observation(world_state, node_state) if self.node.observation_space is not None else None
		info = self.node.get_info(world_state, node_state) or {}
		return WorldFuncEnvState(world_state, node_state), context, obs, info

	def initial(
		self,
		*,
		seed: Optional[int] = None,
		**kwargs,
	) -> Tuple[WorldFuncEnvState, ContextType, ObsType, Dict[str, Any]]:
		world_state = self.world.initial(seed=seed, **kwargs)

		# Create node state across priorities
		node_state = None
		for p in sorted(self.node.initial_priorities, reverse=True):
			world_state, ns_p = self.node.initial(world_state, priority=p, seed=seed, **kwargs)
			if isinstance(ns_p, dict) and isinstance(node_state, dict):
				node_state.update(ns_p)
			else:
				node_state = ns_p

		# after_reset — call at each priority for side effects
		for p in sorted(self.node.after_reset_priorities, reverse=True):
			world_state, node_state = self.node.after_reset(
				world_state, node_state, priority=p
			)

		# Read final state
		context = self.node.get_context(world_state, node_state) if self.node.context_space is not None else None
		obs = self.node.get_observation(world_state, node_state) if self.node.observation_space is not None else None
		info = self.node.get_info(world_state, node_state) or {}
		return WorldFuncEnvState(world_state, node_state), context, obs, info

	def reset(
		self,
		state: WorldFuncEnvState,
		*,
		seed: Optional[int] = None,
		mask: Optional[BArrayType] = None,
		reload: bool = False,
		**kwargs,
	) -> Tuple[WorldFuncEnvState, ContextType, ObsType, Dict[str, Any]]:
		if reload:
			return self.reload(state, seed=seed, **kwargs)
		# 1. World reset
		world_state = self.world.reset(state.world_state, seed=seed, mask=mask, **kwargs)
		node_state = state.node_state

		# 2. Nodes reset
		for p in sorted(self.node.reset_priorities, reverse=True):
			world_state, node_state = self.node.reset(
				world_state, node_state, priority=p, seed=seed, mask=mask, **kwargs
			)

		# 3. World post-reset hook
		world_state = self.world.after_reset(
			world_state, seed=seed, mask=mask, **kwargs
		)

		# 4. Nodes post-reset hook
		for p in sorted(self.node.after_reset_priorities, reverse=True):
			world_state, node_state = self.node.after_reset(
				world_state, node_state, priority=p, mask=mask
			)

		# 5. Read final state
		context = self.node.get_context(world_state, node_state) if self.node.context_space is not None else None
		obs = self.node.get_observation(world_state, node_state) if self.node.observation_space is not None else None
		info = self.node.get_info(world_state, node_state) or {}
		return WorldFuncEnvState(world_state, node_state), context, obs, info

	def step(
		self, state: WorldFuncEnvState, action: ActType
	) -> Tuple[
		WorldFuncEnvState,
		ObsType,
		Union[SupportsFloat, BArrayType],
		Union[bool, BArrayType],
		Union[bool, BArrayType],
		Dict[str, Any],
	]:
		world_state = state.world_state
		node_state = state.node_state

		# 1. Set action
		if self.node.action_space is not None:
			world_state, node_state = self.node.set_next_action(world_state, node_state, action)

		# 2. Update-level substep loop
		actual_dt = None
		for _ in range(self._n_update_substeps):
			for p in sorted(self.node.pre_environment_step_priorities, reverse=True):
				world_state, node_state = self.node.pre_environment_step(
					world_state, node_state, self._update_dt, priority=p
				)
			for _ in range(self._n_world_substeps):
				world_state, actual_dt = self.world.step(world_state)
			post_dt = self._update_dt if self._update_dt is not None else actual_dt
			for p in sorted(self.node.post_environment_step_priorities, reverse=True):
				world_state, node_state = self.node.post_environment_step(
					world_state, node_state, post_dt, priority=p
				)

		# 3. Read results
		obs = self.node.get_observation(world_state, node_state) if self.node.observation_space is not None else None

		if self.node.has_reward:
			reward = self.node.get_reward(world_state, node_state)
		elif self.batch_size is not None:
			reward = self.backend.zeros(
				(self.batch_size,), dtype=self.backend.default_floating_dtype, device=self.device
			)
		else:
			reward = 0.0

		if self.node.has_termination_signal:
			terminated = self.node.get_termination(world_state, node_state)
		elif self.batch_size is not None:
			terminated = self.backend.zeros(
				(self.batch_size,), dtype=self.backend.default_boolean_dtype, device=self.device
			)
		else:
			terminated = False

		if self.node.has_truncation_signal:
			truncated = self.node.get_truncation(world_state, node_state)
		elif self.batch_size is not None:
			truncated = self.backend.zeros(
				(self.batch_size,), dtype=self.backend.default_boolean_dtype, device=self.device
			)
		else:
			truncated = False

		info = self.node.get_info(world_state, node_state) or {}
		return WorldFuncEnvState(world_state, node_state), obs, reward, terminated, truncated, info

	def close(self, state: WorldFuncEnvState) -> None:
		world_state = self.node.close(state.world_state, state.node_state)
		self.world.close(world_state)

	# ========== Render interface ==========

	def render_init(self, state, *, seed=None, render_mode=None, **kwargs):
		return state, None, FuncEnvCommonRenderInfo(render_mode=render_mode)

	def render_image(self, state, render_state):
		if self.node.can_render:
			image = self.node.render(state.world_state, state.node_state)
		else:
			image = None
		return image, state, render_state

	def render_close(self, state, render_state):
		return state
