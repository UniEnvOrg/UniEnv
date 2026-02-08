from typing import Optional, Dict, Any, Tuple, Union, SupportsFloat, Sequence, Iterable
from dataclasses import dataclass

from unienv_interface.env_base.funcenv import FuncEnv
from unienv_interface.env_base.env import ContextType, ObsType, ActType, RenderFrame
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType

from .funcworld import FuncWorld, WorldStateT
from .funcnode import FuncWorldNode, NodeStateT
from .combined_funcnode import CombinedFuncWorldNode


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
	):
		if isinstance(node_or_nodes, FuncWorldNode):
			self.node = node_or_nodes
		else:
			self.node = CombinedFuncWorldNode("combined", node_or_nodes)

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

		# Copy properties from node / world
		self.observation_space = self.node.observation_space
		self.action_space = self.node.action_space
		self.context_space = self.node.context_space
		self.backend = world.backend
		self.device = world.device
		self.batch_size = world.batch_size

	# ========== FuncEnv interface ==========

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
		**kwargs,
	) -> Tuple[WorldFuncEnvState, ContextType, ObsType, Dict[str, Any]]:
		world_state = self.world.reset(state.world_state, seed=seed, mask=mask, **kwargs)
		node_state = state.node_state

		for p in sorted(self.node.reset_priorities, reverse=True):
			world_state, node_state = self.node.reset(
				world_state, node_state, priority=p, seed=seed, mask=mask, **kwargs
			)

		# after_reset
		for p in sorted(self.node.after_reset_priorities, reverse=True):
			world_state, node_state = self.node.after_reset(
				world_state, node_state, priority=p, mask=mask
			)

		# Read final state
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

		# 2. Pre-environment step (once, before all sub-steps)
		for p in sorted(self.node.pre_environment_step_priorities, reverse=True):
			world_state, node_state = self.node.pre_environment_step(
				world_state, node_state, self._control_dt, priority=p
			)

		# 3. World sub-steps
		actual_dt = None
		for _ in range(self._n_substeps):
			world_state, actual_dt = self.world.step(world_state)

		# 4. Post-environment step (once, after all sub-steps)
		post_dt = self._control_dt if self._control_dt is not None else actual_dt
		for p in sorted(self.node.post_environment_step_priorities, reverse=True):
			world_state, node_state = self.node.post_environment_step(
				world_state, node_state, post_dt, priority=p
			)

		# 5. Read results
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
