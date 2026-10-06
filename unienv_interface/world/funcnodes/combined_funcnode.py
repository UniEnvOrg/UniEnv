from typing import Optional, Dict, Set, Any, Tuple, Union, Iterable, Mapping, Generic, Sequence, Callable
from math import lcm

from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace

from ..funcnode import FuncWorldNode
from ..funcworld import WorldStateT

CombinedDataT = Union[Dict[str, Any], Any]
CombinedNodeStateT = Dict[str, Any]

# Reserved internal key under which an optional ``root_node`` participates in
# every dispatch structure of a combined func node, and under which its node
# state is persisted inside ``CombinedNodeStateT``. The empty string can never
# collide with a child node name (child names are validated to be non-empty) and
# is a valid ``str``, so persisted state dictionaries stay serialization
# compatible (``CombinedNodeStateT = Dict[str, Any]``).  Kept in sync with
# :data:`unienv_interface.world.nodes.combined_node.ROOT_NODE_KEY`.
ROOT_NODE_KEY = ""

class CombinedFuncWorldNode(FuncWorldNode[
	WorldStateT, CombinedNodeStateT,
	Optional[CombinedDataT],  # Context type (can be None)
	CombinedDataT,             # Observation type
	CombinedDataT,             # Action type
	BArrayType, BDeviceType, BDtypeType, BRNGType
], Generic[
	WorldStateT, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
	"""A functional counterpart to `CombinedWorldNode` that composes multiple `FuncWorldNode`s.

	It aggregates spaces (context, observation, action) and runtime data (context, observation, info, reward, termination, truncation)
	across child nodes. If only one child exposes a given interface and `direct_return=True`, the value is passed through directly.

	Root node
	---------
	An optional ``root_node`` may be passed next to ``nodes``. It is a regular
	``FuncWorldNode`` (sharing the same ``FuncWorld``) that is NOT part of the
	public ``nodes`` list and whose data merges at the TOP LEVEL of the aggregated
	dictionaries instead of nesting under its own name. It rides the same
	machinery as a child node -- priorities, control/update rate ratios, lifecycle
	dispatch, rewards/signals, info, render and close -- and its node state is
	stored in the combined ``node_state`` under the reserved key
	:data:`ROOT_NODE_KEY` (``""``), a string key that cannot collide with a child
	node name (child names are validated non-empty) and therefore keeps persisted
	``CombinedNodeStateT`` dictionaries loadable without migration.

	- **Spaces / data (context, observation, action -- independently).** Spaces are
	  aggregated once at construction (functional nodes declare static spaces).
	  When the root node is the *sole* contributor of a channel, the existing
	  ``direct_return`` rule applies verbatim: ``direct_return=True`` passes the
	  root's own space/data through unwrapped (any ``Space`` type, e.g. a plain
	  ``BoxSpace``), while ``direct_return=False`` wraps it under the reserved key
	  (``DictSpace({ROOT_NODE_KEY: space})`` / ``{ROOT_NODE_KEY: data}``). When the
	  root and at least one named child contribute, the root MUST expose a
	  ``DictSpace`` for that channel and its keys are merged next to the child
	  entries: ``{child_name: child_space, ..., **root_space.spaces}`` (resp.
	  ``{child_name: child_data, ..., **root_data}``). A root key equal to a child
	  node name raises ``ValueError``.
	- **Actions.** For a sole-root action channel the aggregate action space is the
	  root's action space and the whole action value is forwarded to the root at
	  its own control-rate tick. In the mixed case the incoming mapping is routed
	  by key: keys matching action-child names update those children's cache
	  entries (exactly as before) and every remaining key accumulates into the
	  root's entry (stored under the reserved key) as a dict. Each consumer is
	  dispatched at its own ratio tick with hold-last semantics identical to
	  children.
	- **Info / render.** These channels nest by node name for children, so the
	  root's entries are keyed by the root's own ``name`` (the reserved key is only
	  used for internal state/routing and aggregation).
	- ``get_node`` never resolves the reserved key (it is not a user-facing path);
	  ``get_nodes_by_fn`` does traverse the root node.
	"""

	supported_render_modes = ('dict', 'auto')

	# Reserved internal priority always present in the aggregated
	# ``after_reset_priorities`` / ``after_reload_priorities`` so the combined
	# node's after-hooks are invoked even when no child registers one, which
	# guarantees the routing state (the shared counter etc.) is reset.
	_INTERNAL_RESET_PRIORITY = -10**6

	# Shared update-substep counter. This key is intentionally the legacy
	# ``"__pre_substeps"`` name so that previously persisted node states (which
	# store ``"__pre_substeps"`` and a now-ignored ``"__post_substeps"``) load
	# without migration: the stale ``"__post_substeps"`` entry is simply ignored.
	_COUNTER_PRE = "__pre_substeps"
	_COUNTER_ACTION = "__action_substeps"
	_CACHED_ACTIONS = "__cached_actions"

	def __init__(
		self,
		name: str,
		nodes: Iterable[FuncWorldNode[WorldStateT, Any, Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]],
		direct_return: bool = True,
		render_mode: Optional[str] = 'auto',
		root_node: Optional[FuncWorldNode[WorldStateT, Any, Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]] = None,
	):
		nodes = list(nodes)
		if len(nodes) == 0:
			raise ValueError("At least one node is required to create a CombinedFuncWorldNode.")
		# Internal machinery list: named children first, the root node (added
		# below) last.
		self._all_nodes = list(nodes)

		first_node = nodes[0]
		# Ensure all nodes share the same world
		for node in nodes[1:]:
			assert node.world is first_node.world, "All nodes must belong to the same world." \
				f" Mismatch between {first_node.name} and {node.name}."

		names = [node.name for node in nodes]
		if len(names) != len(set(names)):
			raise ValueError("All nodes must have unique names.")
		# The reserved root key must not be reachable as a child name.
		assert all(node.name for node in nodes), \
			"Child node names must be non-empty strings (the empty string is reserved for the root_node)."

		self.nodes = nodes

		# Optional root node: a node merged at the top level of the aggregated
		# data (and whose state lives under the reserved key) instead of nesting
		# under its own name. It is deliberately NOT part of `self.nodes`.
		if root_node is not None:
			for node in nodes:
				assert node is not root_node, \
					"root_node must not also appear in nodes; it is passed separately."
			assert root_node.world is first_node.world, \
				"root_node must belong to the same world as the other nodes."
			assert root_node.name, \
				"root_node must have a non-empty name (used for info / render keys)."
		self.root_node = root_node
		if root_node is not None:
			self._all_nodes.append(root_node)

		self.name = name
		self.direct_return = direct_return

		# Aggregate spaces. For FuncWorldNodes, spaces are static interface metadata
		# declared at construction; they are not refreshed after after_reload.
		_, self.context_space = self._aggregate_channel_spaces('context_space', 'context')
		_, self.observation_space = self._aggregate_channel_spaces('observation_space', 'observation')
		self._action_node_name_direct, self.action_space = self._aggregate_channel_spaces('action_space', 'action')

		self.has_reward = any(node.has_reward for node in self._all_nodes)
		self.has_termination_signal = any(node.has_termination_signal for node in self._all_nodes)
		self.has_truncation_signal = any(node.has_truncation_signal for node in self._all_nodes)

		# Rendering
		renderable_nodes = [node for node in self._all_nodes if node.can_render]
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
		eff_steps = [n.effective_update_timestep for n in self._all_nodes if n.effective_update_timestep is not None]
		ctrls = [n.control_timestep for n in self._all_nodes if n.control_timestep is not None]

		if eff_steps:
			self._smallest_update_ts = min(eff_steps)
			for node in self._all_nodes:
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
			for node in self._all_nodes:
				if node.control_timestep is not None:
					r = largest_ctrl / node.control_timestep
					assert abs(r - round(r)) < 1e-9, \
						f"Largest control ({largest_ctrl}) not integer multiple of {node.name}'s ({node.control_timestep})"
		else:
			self._smallest_control_ts = None

		# Precompute routing ratios (keyed by the internal node key, so the root
		# node is addressed by the reserved key).
		self._update_ratios = {}
		self._action_ratios = {}
		for node in self._all_nodes:
			key = self._node_key(node)
			es = node.effective_update_timestep
			self._update_ratios[key] = round(es / self._smallest_update_ts) if (es and self._smallest_update_ts) else 1
			ct = node.control_timestep
			self._action_ratios[key] = round(ct / self._smallest_control_ts) if (ct and self._smallest_control_ts) else 1

		# Wrapping periods (LCM of ratios) to keep counters bounded
		self._update_period = lcm(*self._update_ratios.values()) if self._update_ratios else 1
		self._action_period = lcm(*self._action_ratios.values()) if self._action_ratios else 1

	# ========== Helper aggregation methods ==========
	def _node_key(self, node: FuncWorldNode) -> str:
		"""Internal dispatch / state key for ``node``.

		The root node is addressed by the reserved :data:`ROOT_NODE_KEY` (its own
		``name`` is only used for display keys such as info / render output), every
		other node keeps its own name. The reserved key is a ``str`` so persisted
		``CombinedNodeStateT`` dictionaries stay plain ``Dict[str, Any]``.
		"""
		return ROOT_NODE_KEY if (self.root_node is not None and node is self.root_node) else node.name

	def _aggregate_channel_spaces(
		self,
		space_attr: str,
		channel: str,
	) -> Tuple[Optional[str], Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]]:
		"""Aggregate one space channel (context / observation / action) over the
		named children and the optional root node.

		- No root contribution -> the plain :meth:`aggregate_spaces` rule.
		- Root as the sole contributor -> the plain ``direct_return`` rule applied
		  to the single reserved-key entry (so ``direct_return=True`` passes the
		  root's own space through unwrapped, whatever its type, and
		  ``direct_return=False`` wraps it under the reserved key).
		- Root + at least one named child -> a merged ``DictSpace`` holding the
		  children under their names plus the root's own keys at the top level. The
		  root must be a ``DictSpace`` in this case, and its keys must not collide
		  with child node names.
		"""
		children_space = {
			node.name: getattr(node, space_attr)
			for node in self.nodes
			if getattr(node, space_attr) is not None
		}
		root = self.root_node
		root_space = getattr(root, space_attr) if root is not None else None
		if root_space is None:
			return self.aggregate_spaces(children_space, direct_return=self.direct_return)
		if not children_space:
			return self.aggregate_spaces({ROOT_NODE_KEY: root_space}, direct_return=self.direct_return)
		assert isinstance(root_space, DictSpace), (
			f"root_node '{root.name}' must expose a DictSpace {channel} space when combined "
			f"with named child nodes, got {type(root_space).__name__}."
		)
		for key in root_space.spaces:
			if key in children_space:
				raise ValueError(
					f"Root {channel} key '{key}' of root_node '{root.name}' collides with "
					f"the child node named '{key}'; root {channel} keys must not collide with "
					f"child node names."
				)
		backend = next(iter(children_space.values())).backend
		return None, DictSpace(backend, {**children_space, **root_space.spaces})

	def _aggregate_channel_data(
		self,
		nodes: Iterable[FuncWorldNode],
		data_method: str,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> Optional[Union[Dict[str, Any], Any]]:
		"""Collect ``data_method`` from ``nodes`` and aggregate it.

		Mirrors :meth:`_aggregate_channel_spaces`: a sole root contributor follows
		the ``direct_return`` convention, a mixed composition merges the root
		mapping at the top level next to the child entries.
		"""
		child_data: Dict[str, Any] = {}
		root_data: Any = None
		root_contributes = False
		for node in nodes:
			key = self._node_key(node)
			data = getattr(node, data_method)(world_state, node_state[key])
			if self.root_node is not None and node is self.root_node:
				root_data = data
				root_contributes = True
			else:
				child_data[key] = data

		if not root_contributes:
			return self.aggregate_data(child_data, direct_return=self.direct_return)
		if not child_data:
			return self.aggregate_data({ROOT_NODE_KEY: root_data}, direct_return=self.direct_return)
		assert isinstance(root_data, Mapping), (
			f"root_node '{self.root_node.name}' must return a mapping for the aggregated "
			f"channel when combined with named child nodes, got {type(root_data).__name__}."
		)
		return {**child_data, **root_data}

	@staticmethod
	def aggregate_spaces(
		spaces: Dict[str, Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]]],
		direct_return: bool = True,
	) -> Tuple[Optional[str], Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]]:
		"""Aggregate child spaces into either a passthrough or a ``DictSpace``."""
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
		"""Aggregate child outputs using the same direct-return convention as spaces."""
		if len(data) == 0:
			return None
		elif len(data) == 1 and direct_return:
			return next(iter(data.values()))
		else:
			return data

	# ========== Node query methods ==========
	def get_node(self, nested_keys: Union[str, Sequence[str]]) -> Optional[FuncWorldNode]:
		"""Resolve a child node by dotted-path-like key traversal."""
		if isinstance(nested_keys, str):
			keys = [nested_keys]
		else:
			keys = list(nested_keys)

		if len(keys) == 0:
			return self

		key = keys[0]
		child = next((node for node in self.nodes if node.name == key), None)
		if child is None:
			return None

		if len(keys) == 1:
			return child
		return child.get_node(keys[1:])

	def get_nodes_by_fn(self, fn: Callable[[FuncWorldNode], bool]) -> list[FuncWorldNode]:
		"""Return every node in the subtree that satisfies ``fn`` (root included)."""
		result: list[FuncWorldNode] = []
		if fn(self):
			result.append(self)
		for node in self._all_nodes:
			result.extend(node.get_nodes_by_fn(fn))
		return result

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
		return self._collect_priorities(self._all_nodes, 'initial_priorities')

	@property
	def reset_priorities(self) -> Set[int]:
		return self._collect_priorities(self._all_nodes, 'reset_priorities')

	@property
	def reload_priorities(self) -> Set[int]:
		return self._collect_priorities(self._all_nodes, 'reload_priorities')

	@property
	def after_reset_priorities(self) -> Set[int]:
		# Always include the reserved internal priority so the combined node's
		# after_reset hook is invoked (and routing state reset) even when no
		# child registers an after_reset priority.
		return self._collect_priorities(self._all_nodes, 'after_reset_priorities') | {self._INTERNAL_RESET_PRIORITY}

	@property
	def after_reload_priorities(self) -> Set[int]:
		# Always include the reserved internal priority so the combined node's
		# after_reload hook is invoked (and routing state reset) even when no
		# child registers an after_reload priority.
		return self._collect_priorities(self._all_nodes, 'after_reload_priorities') | {self._INTERNAL_RESET_PRIORITY}

	@property
	def pre_environment_step_priorities(self) -> Set[int]:
		return self._collect_priorities(self._all_nodes, 'pre_environment_step_priorities')

	@property
	def post_environment_step_priorities(self) -> Set[int]:
		return self._collect_priorities(self._all_nodes, 'post_environment_step_priorities')

	# ========== Lifecycle methods ==========
	def initial(
		self,
		world_state: WorldStateT,
		*,
		priority: int = 0,
		seed: Optional[int] = None,
		pernode_kwargs: Dict[str, Dict[str, Any]] = {},
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		"""Create node states (children and the optional root node, whose state is
		stored under the reserved key) for the current initialization priority."""
		node_states: CombinedNodeStateT = {}
		for node in self._all_nodes:
			if priority in node.initial_priorities:
				world_state, node_state = node.initial(world_state, priority=priority, seed=seed, **pernode_kwargs.get(node.name, {}))
				node_states[self._node_key(node)] = node_state
		node_states[self._COUNTER_PRE] = 0
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
		"""Recreate node states (children and the optional root node) for the current reload priority."""
		node_states: CombinedNodeStateT = {}
		for node in self._all_nodes:
			if priority in node.reload_priorities:
				world_state, node_state = node.reload(world_state, priority=priority, seed=seed, **pernode_kwargs.get(node.name, {}))
				node_states[self._node_key(node)] = node_state
		node_states[self._COUNTER_PRE] = 0
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
		"""Forward reset calls to the nodes (children and the optional root node)
		that participate at ``priority``."""
		node_state = node_state.copy()
		for node in self._all_nodes:
			key = self._node_key(node)
			if priority in node.reset_priorities:
				ns = node_state[key]
				world_state, ns = node.reset(
					world_state,
					ns,
					priority=priority,
					seed=seed,
					mask=mask,
					**pernode_kwargs.get(node.name, {}),
				)
				node_state[key] = ns
		return world_state, node_state

	def after_reset(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
		*,
		priority: int = 0,
		mask: Optional[BArrayType] = None,
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		"""Forward post-reset hooks and reset internal routing counters."""
		node_state = node_state.copy()
		all_prios = self.after_reset_priorities
		# Reset routing state at the top after_reset priority (existing
		# behaviour) and at the reserved internal priority (guarantees a reset
		# even when no child registers an after_reset priority). Both are
		# idempotent.
		if (all_prios and priority == max(all_prios)) or priority == self._INTERNAL_RESET_PRIORITY:
			node_state[self._COUNTER_PRE] = 0
			node_state[self._COUNTER_ACTION] = 0
			node_state[self._CACHED_ACTIONS] = {}

		for node in self._all_nodes:
			key = self._node_key(node)
			if priority in node.after_reset_priorities:
				ns = node_state[key]
				world_state, ns = node.after_reset(world_state, ns, priority=priority, mask=mask)
				node_state[key] = ns
		return world_state, node_state

	def after_reload(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
		*,
		priority: int = 0,
		mask: Optional[BArrayType] = None,
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		"""Call after_reload on child nodes. Similar to after_reset but for reload flow."""
		node_state = node_state.copy()
		all_prios = self.after_reload_priorities
		# Reset routing state at the top after_reload priority (existing
		# behaviour) and at the reserved internal priority (guarantees a reset
		# even when no child registers an after_reload priority). Both are
		# idempotent.
		if (all_prios and priority == max(all_prios)) or priority == self._INTERNAL_RESET_PRIORITY:
			node_state[self._COUNTER_PRE] = 0
			node_state[self._COUNTER_ACTION] = 0
			node_state[self._CACHED_ACTIONS] = {}

		for node in self._all_nodes:
			key = self._node_key(node)
			if priority in node.after_reload_priorities:
				ns = node_state[key]
				world_state, ns = node.after_reload(world_state, ns, priority=priority, mask=mask)
				node_state[key] = ns
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
		pre_prios = self.pre_environment_step_priorities
		# Advance the shared update counter exactly once per update substep:
		# at the top pre_step priority when any child registers one. When no
		# child registers a pre_step priority (composition with no pre-step
		# priorities), the counter is advanced in post_environment_step
		# instead, so it is NOT advanced here.
		if pre_prios and priority == max(pre_prios):
			node_state[self._COUNTER_PRE] = (node_state[self._COUNTER_PRE] % self._update_period) + 1

		for node in self._all_nodes:
			key = self._node_key(node)
			if priority in node.pre_environment_step_priorities:
				ratio = self._update_ratios[key]
				if (node_state[self._COUNTER_PRE] - 1) % ratio == 0:
					ns = node_state[key]
					world_state, ns = node.pre_environment_step(world_state, ns, node.effective_update_timestep, priority=priority)
					node_state[key] = ns
		return world_state, node_state

	def _split_child_actions(self, action: CombinedDataT) -> Dict[str, Any]:
		"""Map an incoming combined action to per-node actions for the nodes
		that receive a *new* action this call, keyed by node name.

		The default implementation routes by node name (nested actions) or
		passes the action through to the single direct-return action node.
		Subclasses may override this to implement a different routing scheme
		(e.g. :class:`FlatCombinedFuncWorldNode` slices flat action dicts by
		data keys); :meth:`set_next_action` owns the counter / cache / ratio
		dispatch orchestration and should not need to be overridden.

		This contract stays children-only: an optional ``root_node`` is handled
		inline by :meth:`set_next_action`, which extracts the action keys no child
		claims and stores them under the reserved root key.
		"""
		if self._action_node_name_direct is not None:
			return {self._action_node_name_direct: action}
		assert isinstance(action, Mapping), "Action must be a mapping when there are multiple action spaces."
		return action

	def set_next_action(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
		action: CombinedDataT,
	) -> Tuple[WorldStateT, CombinedNodeStateT]:
		"""Route combined actions to child nodes (and the optional root node),
		respecting per-node control rates.

		The incoming action is split by :meth:`_split_child_actions` (children
		only). When a ``root_node`` owns an action space and no single node is the
		direct-action passthrough provider, the keys that do not belong to any
		named child are the root's action keys: they accumulate into the root's
		cache entry (stored under the reserved key) as a dict. Every consumer
		(children and root) is then dispatched at its own ratio tick, and only once
		it has received at least one action (hold-last semantics, so a child-only
		step re-delivers the root's previous action at its tick and vice versa).
		"""
		if self.action_space is None:
			assert action is None, "Cannot provide an action when action_space is None."
			return world_state, node_state

		node_state = node_state.copy()
		node_state[self._COUNTER_ACTION] = (node_state[self._COUNTER_ACTION] % self._action_period) + 1

		child_actions = self._split_child_actions(action)

		# Root remainder routing, inline so that `_split_child_actions` keeps its
		# children-only contract. Flat subclasses already emit node-keyed slices
		# (including the reserved root key), in which case this is a no-op because
		# the reserved key is excluded from the remainder.
		root = self.root_node
		if root is not None and root.action_space is not None \
				and self._action_node_name_direct is None and isinstance(child_actions, Mapping):
			child_names = {node.name for node in self.nodes}
			root_actions = {
				key: value for key, value in child_actions.items()
				if key not in child_names and key != ROOT_NODE_KEY
			}
			if root_actions:
				node_state[self._CACHED_ACTIONS] = {
					**node_state[self._CACHED_ACTIONS],
					ROOT_NODE_KEY: root_actions,
				}

		cached = node_state[self._CACHED_ACTIONS].copy()
		for node in self._all_nodes:
			if node.action_space is None:
				continue
			key = self._node_key(node)
			if key in child_actions:
				cached[key] = child_actions[key]
			ratio = self._action_ratios[key]
			# Dispatch at the node's control-rate tick, but only once it has
			# received at least one action (sparse/partial actions are valid).
			if (node_state[self._COUNTER_ACTION] - 1) % ratio == 0 and key in cached:
				world_state, ns = node.set_next_action(world_state, node_state[key], cached[key])
				node_state[key] = ns
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
		pre_prios = self.pre_environment_step_priorities
		post_prios = self.post_environment_step_priorities
		# Advance the shared update counter here only when no child registers a
		# pre_step priority (composition with no pre-step priorities);
		# otherwise it was already advanced in pre_environment_step.
		if not pre_prios and post_prios and priority == max(post_prios):
			node_state[self._COUNTER_PRE] = (node_state[self._COUNTER_PRE] % self._update_period) + 1

		for node in self._all_nodes:
			key = self._node_key(node)
			if priority in node.post_environment_step_priorities:
				ratio = self._update_ratios[key]
				if (node_state[self._COUNTER_PRE] - 1) % ratio == 0:
					ns = node_state[key]
					world_state, ns = node.post_environment_step(world_state, ns, node.effective_update_timestep, priority=priority)
					node_state[key] = ns

		# With a single shared counter the pre/post balance is trivially
		# consistent (there is only one counter), so no equality assert is
		# needed here.
		return world_state, node_state

	def close(self, world_state: WorldStateT, node_state: CombinedNodeStateT) -> WorldStateT:  # type: ignore[override]
		"""Close every child node and the optional root node."""
		for node in self._all_nodes:
			world_state = node.close(world_state, node_state[self._node_key(node)])
		return world_state

	# ========== Data accessors ==========
	def get_context(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> CombinedDataT:
		assert self.context_space is not None, "Context space is None, cannot get context."
		return self._aggregate_channel_data(
			[node for node in self._all_nodes if node.context_space is not None],
			'get_context',
			world_state,
			node_state,
		)

	def get_observation(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> CombinedDataT:
		assert self.observation_space is not None, "Observation space is None, cannot get observation."
		return self._aggregate_channel_data(
			[node for node in self._all_nodes if node.observation_space is not None],
			'get_observation',
			world_state,
			node_state,
		)

	def get_reward(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> Union[float, BArrayType]:
		assert self.has_reward, "This node does not provide a reward."
		if self.world.batch_size is None:
			return sum(
				node.get_reward(world_state, node_state[self._node_key(node)])
				for node in self._all_nodes
				if node.has_reward
			)
		rewards = self.backend.zeros(
			(self.world.batch_size,),
			dtype=self.backend.default_floating_dtype,
			device=self.device,
		)
		for node in self._all_nodes:
			if node.has_reward:
				rewards = rewards + node.get_reward(world_state, node_state[self._node_key(node)])
		return rewards

	def get_termination(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> Union[bool, BArrayType]:
		assert self.has_termination_signal, "This node does not provide a termination signal."
		if self.world.batch_size is None:
			return any(
				node.get_termination(world_state, node_state[self._node_key(node)])
				for node in self._all_nodes
				if node.has_termination_signal
			)
		terminations = self.backend.zeros(
			(self.world.batch_size,),
			dtype=self.backend.default_boolean_dtype,
			device=self.device,
		)
		for node in self._all_nodes:
			if node.has_termination_signal:
				terminations = self.backend.logical_or(
					terminations, node.get_termination(world_state, node_state[self._node_key(node)])
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
				node.get_truncation(world_state, node_state[self._node_key(node)])
				for node in self._all_nodes
				if node.has_truncation_signal
			)
		truncations = self.backend.zeros(
			(self.world.batch_size,),
			dtype=self.backend.default_boolean_dtype,
			device=self.device,
		)
		for node in self._all_nodes:
			if node.has_truncation_signal:
				truncations = self.backend.logical_or(
					truncations, node.get_truncation(world_state, node_state[self._node_key(node)])
				)
		return truncations

	def get_info(
		self,
		world_state: WorldStateT,
		node_state: CombinedNodeStateT,
	) -> Optional[Dict[str, Any]]:
		infos: Dict[str, Any] = {}
		for node in self._all_nodes:
			info = node.get_info(world_state, node_state[self._node_key(node)])
			if info is not None:
				# Info nests by node name (display key), so the root node uses
				# its own name here -- never the reserved internal key.
				infos[node.name] = info
		return self.aggregate_data(infos, direct_return=False)  # Always dict if not empty

	def render(self, world_state, node_state):
		if not self.can_render:
			return None
		if len(self._renderable_nodes) == 1 and self._true_render_mode != 'dict':
			node = self._renderable_nodes[0]
			return node.render(world_state, node_state[self._node_key(node)])
		result = {}
		for node in self._renderable_nodes:
			r = node.render(world_state, node_state[self._node_key(node)])
			if r is None:
				continue
			if isinstance(r, dict):
				for k, v in r.items():
					result[f"{node.name}.{k}"] = v
			else:
				result[node.name] = r
		return result if result else None
