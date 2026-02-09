from typing import Optional, Dict, Set, Any, Tuple, Union, Iterable, Mapping
from math import lcm

from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace

from .funcnode import FuncWorldNode
from .funcworld import WorldStateT

CombinedDataT = Union[Dict[str, Any], Any]
CombinedNodeStateT = Dict[str, Any]

class CombinedFuncWorldNode(FuncWorldNode[
	WorldStateT, CombinedNodeStateT,
	Optional[CombinedDataT],  # Context type (can be None)
	CombinedDataT,             # Observation type
	CombinedDataT,             # Action type
	BArrayType, BDeviceType, BDtypeType, BRNGType
]):
	"""A functional counterpart to `CombinedWorldNode` that composes multiple `FuncWorldNode`s.

	It aggregates spaces (context, observation, action) and runtime data (context, observation, info, reward, termination, truncation)
	across child nodes. If only one child exposes a given interface and `direct_return=True`, the value is passed through directly.
	"""

	supported_render_modes = ('dict', 'auto')

	_COUNTER_PRE = "__pre_substeps"
	_COUNTER_POST = "__post_substeps"
	_COUNTER_ACTION = "__action_substeps"
	_CACHED_ACTIONS = "__cached_actions"

	def __init__(
		self,
		name: str,
		nodes: Iterable[FuncWorldNode[WorldStateT, Any, Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]],
		direct_return: bool = True,
		render_mode: Optional[str] = 'dict',
	):
		nodes = list(nodes)
		if len(nodes) == 0:
			raise ValueError("At least one node is required to create a CombinedFuncWorldNode.")

		first_node = nodes[0]
		# Ensure all nodes share the same world
		for node in nodes[1:]:
			assert node.world is first_node.world, "All nodes must belong to the same world." \
				f" Mismatch between {first_node.name} and {node.name}."

		names = [node.name for node in nodes]
		if len(names) != len(set(names)):
			raise ValueError("All nodes must have unique names.")

		self.nodes = nodes

		# Aggregate spaces similar to `CombinedWorldNode`
		_, self.context_space = self.aggregate_spaces(
			{node.name: node.context_space for node in nodes if node.context_space is not None},
			direct_return=direct_return,
		)
		_, self.observation_space = self.aggregate_spaces(
			{node.name: node.observation_space for node in nodes if node.observation_space is not None},
			direct_return=direct_return,
		)
		self._action_node_name_direct, self.action_space = self.aggregate_spaces(
			{node.name: node.action_space for node in nodes if node.action_space is not None},
			direct_return=direct_return,
		)

		self.has_reward = any(node.has_reward for node in nodes)
		self.has_termination_signal = any(node.has_termination_signal for node in nodes)
		self.has_truncation_signal = any(node.has_truncation_signal for node in nodes)

		self.name = name
		self.direct_return = direct_return

		# Rendering
		renderable_nodes = [node for node in nodes if node.can_render]
		self._renderable_nodes = renderable_nodes
		self._true_render_mode = render_mode
		if render_mode == 'auto':
			if len(renderable_nodes) == 1:
				self.render_mode = renderable_nodes[0].render_mode
			elif len(renderable_nodes) > 1:
				self.render_mode = 'dict'
			else:
				self.render_mode = None
		elif render_mode == 'dict':
			self.render_mode = 'dict' if renderable_nodes else None
		else:
			self.render_mode = render_mode

		# Multi-frequency validation + precomputation
		eff_steps = [n.effective_update_timestep for n in nodes if n.effective_update_timestep is not None]
		ctrls = [n.control_timestep for n in nodes if n.control_timestep is not None]

		if eff_steps:
			self._smallest_update_ts = min(eff_steps)
			for node in nodes:
				es = node.effective_update_timestep
				if es is not None:
					r = es / self._smallest_update_ts
					assert abs(r - round(r)) < 1e-9, \
						f"Node {node.name} update_timestep ({es}) not integer multiple of smallest ({self._smallest_update_ts})"
		else:
			self._smallest_update_ts = None

		if ctrls:
			self._smallest_control_ts = min(ctrls)
			largest_ctrl = max(ctrls)
			for node in nodes:
				if node.control_timestep is not None:
					r = largest_ctrl / node.control_timestep
					assert abs(r - round(r)) < 1e-9, \
						f"Largest control ({largest_ctrl}) not integer multiple of {node.name}'s ({node.control_timestep})"
		else:
			self._smallest_control_ts = None

		# Precompute routing ratios
		self._update_ratios = {}
		self._action_ratios = {}
		for node in nodes:
			es = node.effective_update_timestep
			self._update_ratios[node.name] = round(es / self._smallest_update_ts) if (es and self._smallest_update_ts) else 1
			ct = node.control_timestep
			self._action_ratios[node.name] = round(ct / self._smallest_control_ts) if (ct and self._smallest_control_ts) else 1

		# Wrapping periods (LCM of ratios) to keep counters bounded
		self._update_period = lcm(*self._update_ratios.values()) if self._update_ratios else 1
		self._action_period = lcm(*self._action_ratios.values()) if self._action_ratios else 1

	# ========== Helper aggregation methods ==========
	@staticmethod
	def aggregate_spaces(
		spaces: Dict[str, Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]]],
		direct_return: bool = True,
	) -> Tuple[Optional[str], Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]]:
		if len(spaces) == 0:
			return None, None
		elif len(spaces) == 1 and direct_return:
			return next(iter(spaces.items()))
		else:
			backend = next(iter(spaces.values())).backend
			return None, DictSpace(
				backend,
				{name: space for name, space in spaces.items() if space is not None},
			)

	@staticmethod
	def aggregate_data(
		data: Dict[str, Any],
		direct_return: bool = True,
	) -> Optional[Union[Dict[str, Any], Any]]:
		if len(data) == 0:
			return None
		elif len(data) == 1 and direct_return:
			return next(iter(data.values()))
		else:
			return data

	# ========== properties ==========
	@property
	def world(self):  # type: ignore[override]
		return self.nodes[0].world

	@property
	def control_timestep(self):  # type: ignore[override]
		return self._smallest_control_ts

	@property
	def update_timestep(self):  # type: ignore[override]
		return self._smallest_update_ts

	@property
	def effective_update_timestep(self):  # type: ignore[override]
		return self._smallest_update_ts

	# ========== Aggregated priority properties ==========
	@staticmethod
	def _collect_priorities(nodes, attr_name) -> Set[int]:
		return set().union(*(getattr(node, attr_name) for node in nodes))

	@property
	def initial_priorities(self) -> Set[int]:
		return self._collect_priorities(self.nodes, 'initial_priorities')

	@property
	def reset_priorities(self) -> Set[int]:
		return self._collect_priorities(self.nodes, 'reset_priorities')

	@property
	def reload_priorities(self) -> Set[int]:
		return self._collect_priorities(self.nodes, 'reload_priorities')

	@property
	def after_reset_priorities(self) -> Set[int]:
		return self._collect_priorities(self.nodes, 'after_reset_priorities')

	@property
	def pre_environment_step_priorities(self) -> Set[int]:
		return self._collect_priorities(self.nodes, 'pre_environment_step_priorities')

	@property
	def post_environment_step_priorities(self) -> Set[int]:
		return self._collect_priorities(self.nodes, 'post_environment_step_priorities')

	# ========== Lifecycle methods ==========
	def initial(
		self,
		world_state: WorldStateT,
		*,
		priority: int = 0,
		seed: Optional[int] = None,
		pernode_kwargs: Dict[str, Dict[str, Any]] = {},
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		node_states: CombinedNodeStateT = {}
		for node in self.nodes:
			if priority in node.initial_priorities:
				world_state, node_state = node.initial(world_state, priority=priority, seed=seed, **pernode_kwargs.get(node.name, {}))
				node_states[node.name] = node_state
		node_states[self._COUNTER_PRE] = 0
		node_states[self._COUNTER_POST] = 0
		node_states[self._COUNTER_ACTION] = 0
		node_states[self._CACHED_ACTIONS] = {}
		return world_state, node_states

	def reload(
		self,
		world_state: WorldStateT,
		*,
		priority: int = 0,
		seed: Optional[int] = None,
		pernode_kwargs: Dict[str, Dict[str, Any]] = {},
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		node_states: CombinedNodeStateT = {}
		for node in self.nodes:
			if priority in node.reload_priorities:
				world_state, node_state = node.reload(world_state, priority=priority, seed=seed, **pernode_kwargs.get(node.name, {}))
				node_states[node.name] = node_state
		node_states[self._COUNTER_PRE] = 0
		node_states[self._COUNTER_POST] = 0
		node_states[self._COUNTER_ACTION] = 0
		node_states[self._CACHED_ACTIONS] = {}
		return world_state, node_states

	def reset(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
		*,
		priority: int = 0,
		seed: Optional[int] = None,
		mask: Optional[BArrayType] = None,
		pernode_kwargs: Dict[str, Dict[str, Any]] = {},
		**kwargs,
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		node_state = node_state.copy()
		for node in self.nodes:
			if priority in node.reset_priorities:
				ns = node_state[node.name]
				world_state, ns = node.reset(
					world_state,
					ns,
					priority=priority,
					seed=seed,
					mask=mask,
					**pernode_kwargs.get(node.name, {}),
				)
				node_state[node.name] = ns
		return world_state, node_state

	def after_reset(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
		*,
		priority: int = 0,
		mask: Optional[BArrayType] = None,
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		node_state = node_state.copy()
		all_prios = self.after_reset_priorities
		if all_prios and priority == max(all_prios):
			node_state[self._COUNTER_PRE] = 0
			node_state[self._COUNTER_POST] = 0
			node_state[self._COUNTER_ACTION] = 0
			node_state[self._CACHED_ACTIONS] = {}

		for node in self.nodes:
			if priority in node.after_reset_priorities:
				ns = node_state[node.name]
				world_state, ns = node.after_reset(world_state, ns, priority=priority, mask=mask)
				node_state[node.name] = ns
		return world_state, node_state

	def pre_environment_step(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
		dt: Union[float, BArrayType],
		*,
		priority: int = 0,
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		node_state = node_state.copy()
		all_prios = self.pre_environment_step_priorities
		if all_prios and priority == max(all_prios):
			node_state[self._COUNTER_PRE] = (node_state[self._COUNTER_PRE] % self._update_period) + 1

		for node in self.nodes:
			if priority in node.pre_environment_step_priorities:
				ratio = self._update_ratios[node.name]
				if (node_state[self._COUNTER_PRE] - 1) % ratio == 0:
					ns = node_state[node.name]
					world_state, ns = node.pre_environment_step(world_state, ns, node.effective_update_timestep, priority=priority)
					node_state[node.name] = ns
		return world_state, node_state

	def set_next_action(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
		action: CombinedDataT,
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		assert self.action_space is not None, "Action space is None, cannot set action."

		node_state = node_state.copy()
		node_state[self._COUNTER_ACTION] = (node_state[self._COUNTER_ACTION] % self._action_period) + 1

		if self._action_node_name_direct is not None:
			child_actions = {self._action_node_name_direct: action}
		else:
			assert isinstance(action, Mapping), "Action must be a mapping when there are multiple action spaces."
			child_actions = action

		cached = node_state[self._CACHED_ACTIONS].copy()
		for node in self.nodes:
			if node.action_space is not None:
				if node.name in child_actions:
					cached[node.name] = child_actions[node.name]
				ratio = self._action_ratios[node.name]
				if (node_state[self._COUNTER_ACTION] - 1) % ratio == 0:
					ns = node_state[node.name]
					world_state, ns = node.set_next_action(world_state, ns, cached[node.name])
					node_state[node.name] = ns
		node_state[self._CACHED_ACTIONS] = cached
		return world_state, node_state

	def post_environment_step(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
		dt: Union[float, BArrayType],
		*,
		priority: int = 0,
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		node_state = node_state.copy()
		all_prios = self.post_environment_step_priorities
		if all_prios and priority == max(all_prios):
			node_state[self._COUNTER_POST] = (node_state[self._COUNTER_POST] % self._update_period) + 1

		for node in self.nodes:
			if priority in node.post_environment_step_priorities:
				ratio = self._update_ratios[node.name]
				if (node_state[self._COUNTER_POST] - 1) % ratio == 0:
					ns = node_state[node.name]
					world_state, ns = node.post_environment_step(world_state, ns, node.effective_update_timestep, priority=priority)
					node_state[node.name] = ns

		if all_prios and priority == max(all_prios):
			assert node_state[self._COUNTER_PRE] == node_state[self._COUNTER_POST]
		return world_state, node_state

	def close(self, world_state: WorldStateT, node_state: CombinedNodeStateT) -> WorldStateT:  # type: ignore[override]
		for node in self.nodes:
			world_state = node.close(world_state, node_state[node.name])
		return world_state

	# ========== Data accessors ==========
	def get_context(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> CombinedDataT:
		assert self.context_space is not None, "Context space is None, cannot get context."
		return self.aggregate_data(
			{
				node.name: node.get_context(world_state, node_state[node.name])
				for node in self.nodes
				if node.context_space is not None
			},
			direct_return=self.direct_return,
		)

	def get_observation(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> CombinedDataT:
		assert self.observation_space is not None, "Observation space is None, cannot get observation."
		return self.aggregate_data(
			{
				node.name: node.get_observation(world_state, node_state[node.name])
				for node in self.nodes
				if node.observation_space is not None
			},
			direct_return=self.direct_return,
		)

	def get_reward(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> Union[float, BArrayType]:
		assert self.has_reward, "This node does not provide a reward."
		if self.world.batch_size is None:
			return sum(
				node.get_reward(world_state, node_state[node.name])
				for node in self.nodes
				if node.has_reward
			)
		rewards = self.backend.zeros(
			(self.world.batch_size,),
			dtype=self.backend.default_floating_dtype,
			device=self.device,
		)
		for node in self.nodes:
			if node.has_reward:
				rewards = rewards + node.get_reward(world_state, node_state[node.name])
		return rewards

	def get_termination(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> Union[bool, BArrayType]:
		assert self.has_termination_signal, "This node does not provide a termination signal."
		if self.world.batch_size is None:
			return any(
				node.get_termination(world_state, node_state[node.name])
				for node in self.nodes
				if node.has_termination_signal
			)
		terminations = self.backend.zeros(
			(self.world.batch_size,),
			dtype=self.backend.default_boolean_dtype,
			device=self.device,
		)
		for node in self.nodes:
			if node.has_termination_signal:
				terminations = self.backend.logical_or(
					terminations, node.get_termination(world_state, node_state[node.name])
				)
		return terminations

	def get_truncation(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> Union[bool, BArrayType]:
		assert self.has_truncation_signal, "This node does not provide a truncation signal."
		if self.world.batch_size is None:
			return any(
				node.get_truncation(world_state, node_state[node.name])
				for node in self.nodes
				if node.has_truncation_signal
			)
		truncations = self.backend.zeros(
			(self.world.batch_size,),
			dtype=self.backend.default_boolean_dtype,
			device=self.device,
		)
		for node in self.nodes:
			if node.has_truncation_signal:
				truncations = self.backend.logical_or(
					truncations, node.get_truncation(world_state, node_state[node.name])
				)
		return truncations

	def get_info(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> Optional[Dict[str, Any]]:
		infos: Dict[str, Any] = {}
		for node in self.nodes:
			info = node.get_info(world_state, node_state[node.name])
			if info is not None:
				infos[node.name] = info
		return self.aggregate_data(infos, direct_return=False)  # Always dict if not empty

	def render(self, world_state, node_state):
		if not self.can_render:
			return None
		if len(self._renderable_nodes) == 1 and self._true_render_mode != 'dict':
			return self._renderable_nodes[0].render(world_state, node_state[self._renderable_nodes[0].name])
		result = {}
		for node in self._renderable_nodes:
			r = node.render(world_state, node_state[node.name])
			if r is None:
				continue
			if isinstance(r, dict):
				for k, v in r.items():
					result[f"{node.name}.{k}"] = v
			else:
				result[node.name] = r
		return result if result else None
