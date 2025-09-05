from typing import Optional, Dict, Any, Tuple, Union, Iterable, Mapping

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

	def __init__(
		self,
		name: str,
		nodes: Iterable[FuncWorldNode[WorldStateT, Any, Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]],
		direct_return: bool = True,
	):
		nodes = list(nodes)
		if len(nodes) == 0:
			raise ValueError("At least one node is required to create a CombinedFuncWorldNode.")

		first_node = nodes[0]
		# Ensure all nodes share the same world & control timestep
		for node in nodes[1:]:
			assert node.world is first_node.world, "All nodes must belong to the same world." \
				f" Mismatch between {first_node.name} and {node.name}."
			assert node.control_timestep == first_node.control_timestep, "All nodes must have the same control timestep." \
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
		return self.nodes[0].control_timestep

	# ========== Lifecycle methods ==========
	def initial(
		self,
		world_state: WorldStateT,
		*,
		seed: Optional[int] = None,
		pernode_kwargs: Dict[str, Dict[str, Any]] = {},
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		node_states: CombinedNodeStateT = {}
		for node in self.nodes:
			world_state, node_state = node.initial(world_state, seed=seed, **pernode_kwargs.get(node.name, {}))
			node_states[node.name] = node_state
		return world_state, node_states

	def reset(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
		*,
		seed: Optional[int] = None,
		mask: Optional[BArrayType] = None,
		pernode_kwargs: Dict[str, Dict[str, Any]] = {},
		**kwargs,
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		node_state = node_state.copy()
		for node in self.nodes:
			ns = node_state[node.name]
			world_state, ns = node.reset(
				world_state,
				ns,
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
		mask: Optional[BArrayType] = None,
	) -> Tuple[
		WorldStateT,
		CombinedNodeStateT,
		Optional[CombinedDataT],
		Optional[CombinedDataT],
		Optional[Dict[str, Any]],
	]:
		node_state = node_state.copy()
		contexts: Dict[str, Any] = {}
		observations: Dict[str, Any] = {}
		infos: Dict[str, Any] = {}

		for node in self.nodes:
			ns = node_state[node.name]
			world_state, ns, ctx, obs, info = node.after_reset(world_state, ns, mask=mask)
			node_state[node.name] = ns
			if ctx is not None:
				contexts[node.name] = ctx
			if obs is not None:
				observations[node.name] = obs
			if info is not None:
				infos[node.name] = info

		return (
			world_state,
			node_state,
			self.aggregate_data(contexts, direct_return=self.direct_return),
			self.aggregate_data(observations, direct_return=self.direct_return),
			self.aggregate_data(infos, direct_return=False),
		)

	def pre_environment_step(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
		dt: Union[float, BArrayType],
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		node_state = node_state.copy()
		for node in self.nodes:
			ns = node_state[node.name]
			world_state, ns = node.pre_environment_step(world_state, ns, dt)
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
		if self._action_node_name_direct is not None:
			# Only one actionable node
			for node in self.nodes:
				if node.name == self._action_node_name_direct:
					ns = node_state[node.name]
					world_state, ns = node.set_next_action(world_state, ns, action)  # type: ignore[arg-type]
					node_state[node.name] = ns
					break
		else:
			assert isinstance(action, Mapping), "Action must be a mapping when there are multiple action spaces."
			for node in self.nodes:
				if node.action_space is not None:
					assert node.name in action, f"Action for node {node.name} is missing."
					ns = node_state[node.name]
					world_state, ns = node.set_next_action(world_state, ns, action[node.name])
					node_state[node.name] = ns
		return world_state, node_state

	def post_environment_step(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
		dt: Union[float, BArrayType],
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		node_state = node_state.copy()
		for node in self.nodes:
			ns = node_state[node.name]
			world_state, ns = node.post_environment_step(world_state, ns, dt)
			node_state[node.name] = ns
		return world_state, node_state

	def close(self, world_state: WorldStateT, node_state: CombinedNodeStateT) -> WorldStateT:  # type: ignore[override]
		for node in self.nodes:
			world_state = node.close(world_state, node_state[node.name])
		return world_state

	# ========== Data accessors ==========
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
