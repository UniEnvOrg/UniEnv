from typing import Optional, Dict, Set, Mapping, Any, Tuple, Union, Iterable, Generic, Sequence, Callable, List
from math import lcm
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace
from unienv_interface.utils.control_util import find_best_timestep

from ..world import World
from ..node import WorldNode, ContextType, ObsType, ActType

CombinedDataT = Union[Dict[str, Any], BArrayType]

# Reserved internal key under which an optional ``root_node`` participates in
# every dispatch / cache / ratio structure of a combined node. The empty string
# can never collide with a child node name (child names are validated to be
# non-empty) and is a valid ``str`` so it can also be used as a state-dict key
# by the functional variant (``CombinedNodeStateT = Dict[str, Any]`` is
# persisted as-is).
ROOT_NODE_KEY = ""

# Mapping from short phase names (used in the public cache API) to the
# corresponding priority-set attribute on WorldNode.
_PHASE_ATTR_MAP: Dict[str, str] = {
    'reload': 'reload_priorities',
    'after_reload': 'after_reload_priorities',
    'reset': 'reset_priorities',
    'after_reset': 'after_reset_priorities',
    'pre_step': 'pre_environment_step_priorities',
    'post_step': 'post_environment_step_priorities',
}

class CombinedWorldNode(WorldNode[
    Optional[CombinedDataT], CombinedDataT, CombinedDataT,
    BArrayType, BDeviceType, BDtypeType, BRNGType
], Generic[
    BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    """
    A WorldNode that combines multiple WorldNodes into one node, using a dictionary to store the data from each node.
    The observation, reward, termination, truncation, and info are combined from all child nodes.
    The keys in the dictionary are the names of the child nodes.
    If there is only one child node that supports value and `direct_return` is set to True, the value is returned directly instead of a dictionary.

    Root node
    ---------
    An optional ``root_node`` may be passed next to ``nodes``. It is a regular
    node (sharing the same :class:`~unienv_interface.world.world.World`) that is
    NOT part of the public ``nodes`` list and whose data merges at the TOP LEVEL
    of the aggregated dictionaries instead of nesting under its own name. It
    rides the same machinery as a child node -- priorities, control/update rate
    ratios, lifecycle dispatch, rewards/signals, info, render and close -- under
    the reserved internal key :data:`ROOT_NODE_KEY` (``""``), which cannot
    collide with a child node name (child names are validated non-empty).

    - **Spaces / data (context, observation, action -- independently).** When the
      root node is the *sole* contributor of a channel, the existing
      ``direct_return`` rule applies verbatim: ``direct_return=True`` passes the
      root's own space/data through unwrapped (any ``Space`` type, e.g. a plain
      ``BoxSpace``), while ``direct_return=False`` wraps it under the reserved
      key (``DictSpace({ROOT_NODE_KEY: space})`` / ``{ROOT_NODE_KEY: data}``).
      When the root and at least one named child contribute, the root MUST expose
      a ``DictSpace`` for that channel and its keys are merged next to the child
      entries: ``{child_name: child_space, ..., **root_space.spaces}`` (resp.
      ``{child_name: child_data, ..., **root_data}``). A root key equal to a
      child node name raises ``ValueError``.
    - **Actions.** For a sole-root action channel the aggregate action space is
      the root's action space and the whole action value is forwarded to the
      root at its own control-rate tick. In the mixed case the incoming mapping
      is routed by key: keys matching action-child names update those children's
      caches (exactly as before), and every remaining key accumulates into the
      root's action entry as a dict. Each consumer is dispatched at its own ratio
      tick with hold-last semantics identical to children.
    - **Info / render.** These channels nest by node name for children, so the
      root's entries are keyed by the root's own ``name`` (the reserved key is
      only used for internal state/routing and aggregation).
    - ``get_node`` never resolves the reserved key (it is not a user-facing
      path); ``get_nodes_by_fn`` does traverse the root node.
    """

    supported_render_modes = ('dict', 'auto')

    # Reserved internal priority always present in the aggregated
    # ``after_reset_priorities`` / ``after_reload_priorities`` so the combined
    # node's after-hooks are invoked even when no child registers one, which
    # guarantees the routing state (``_update_substeps`` etc.) is reset.
    _INTERNAL_RESET_PRIORITY = -10**6

    def __init__(
        self,
        name : str,
        nodes : Iterable[WorldNode[Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        direct_return : bool = True,
        render_mode : Optional[str] = 'auto',
        root_node : Optional[WorldNode[Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]] = None,
    ):
        nodes = list(nodes)
        if len(nodes) == 0:
            raise ValueError("At least one node is required to create a CombinedWorldNode.")
        # Internal machinery list: named children first, the root node (added
        # below) last. Assigned early so that `close()` (called from `__del__`)
        # is safe even if the remaining validation raises.
        self._all_nodes = list(nodes)

        # Check that all nodes have the same world
        first_node = nodes[0]
        for node in nodes[1:]:
            assert node.world is first_node.world, "All nodes must belong to the same world."
        # Check that all nodes have unique names
        names = [node.name for node in nodes]
        if len(names) != len(set(names)):
            raise ValueError("All nodes must have unique names.")
        # The reserved root key must not be reachable as a child name.
        assert all(node.name for node in nodes), \
            "Child node names must be non-empty strings (the empty string is reserved for the root_node)."
        self.nodes = nodes

        # Optional root node: a node merged at the top level of the aggregated
        # data instead of nesting under its own name. It is deliberately NOT
        # part of `self.nodes` (the public children list) but rides the same
        # dispatch machinery through `self._all_nodes` / `self._node_key`.
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

        self.has_reward = any(node.has_reward for node in self._all_nodes)
        self.has_termination_signal = any(node.has_termination_signal for node in self._all_nodes)
        self.has_truncation_signal = any(node.has_truncation_signal for node in self._all_nodes)

        # Save attributes
        self.name = name
        self.direct_return = direct_return

        # Aggregate spaces (preliminary snapshot; may include None/unbounded placeholders
        # for nodes like robots whose final spaces are only known after after_reload.
        # _refresh_spaces() is called again at the end of after_reload to capture finals.)
        self._refresh_spaces()

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

        # Substep counters.
        # ``_update_substeps`` is the single shared counter for the per-node
        # update-frequency ratio dispatch used by both pre_environment_step and
        # post_environment_step. It is incremented exactly once per update
        # substep: at the top pre_step priority when any child has pre_step
        # priorities, otherwise at the top post_step priority (a composition
        # with no pre-step priorities). This keeps the counter consistent
        # regardless of whether a step has pre-step participants.
        self._update_substeps = 0
        self._action_substeps = 0
        self._cached_actions = {}

        # Stage-1: build dispatch caches (priority orders, participants) and
        # space-dependent node-list caches.
        self._build_dispatch_caches()
        self._refresh_space_caches()

    @staticmethod
    def aggregate_spaces(
        spaces : Dict[str, Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]]],
        direct_return : bool = True,
    ) -> Tuple[
        Optional[str],
        Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]
    ]:
        """Aggregate child spaces into either a passthrough or a ``DictSpace``."""
        if len(spaces) == 0:
            return None, None
        elif len(spaces) == 1 and direct_return:
            return next(iter(spaces.items()))
        else:
            backend = next(iter(spaces.values())).backend
            return None, DictSpace(
                backend,
                {
                    name: space for name, space in spaces.items() if space is not None
                }
            )

    @staticmethod
    def aggregate_data(
        data : Dict[str, Any],
        direct_return : bool = True,
    ) -> Optional[Union[Dict[str, Any], Any]]:
        """Aggregate child outputs using the same direct-return convention as spaces."""
        if len(data) == 0:
            return None
        elif len(data) == 1 and direct_return:
            return next(iter(data.values()))
        else:
            return data

    def _node_key(self, node : WorldNode) -> str:
        """Internal dispatch / state key for ``node``.

        The root node is addressed by the reserved :data:`ROOT_NODE_KEY` so that
        its aggregated data never nests under its own name and can never collide
        with a child node name (child names are validated non-empty); every
        other node keeps its own name.
        """
        return ROOT_NODE_KEY if (self.root_node is not None and node is self.root_node) else node.name

    def _aggregate_channel_spaces(
        self,
        space_attr : str,
        channel : str,
    ) -> Tuple[Optional[str], Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]]:
        """Aggregate one space channel (context / observation / action) over the
        named children and the optional root node.

        - No root contribution -> the plain :meth:`aggregate_spaces` rule.
        - Root as the sole contributor -> the plain ``direct_return`` rule
          applied to the single reserved-key entry (so ``direct_return=True``
          passes the root's own space through unwrapped, whatever its type, and
          ``direct_return=False`` wraps it under the reserved key).
        - Root + at least one named child -> a merged ``DictSpace`` holding the
          children under their names plus the root's own keys at the top level.
          The root must be a ``DictSpace`` in this case, and its keys must not
          collide with child node names.
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
        nodes : List[WorldNode],
        data_method : str,
    ) -> Optional[Union[Dict[str, Any], Any]]:
        """Collect ``data_method`` from ``nodes`` and aggregate it.

        Mirrors :meth:`_aggregate_channel_spaces`: a sole root contributor
        follows the ``direct_return`` convention, a mixed composition merges the
        root mapping at the top level next to the child entries.
        """
        child_data: Dict[str, Any] = {}
        root_data: Any = None
        root_contributes = False
        for node in nodes:
            if self.root_node is not None and node is self.root_node:
                root_data = getattr(node, data_method)()
                root_contributes = True
            else:
                child_data[node.name] = getattr(node, data_method)()

        if not root_contributes:
            return self.aggregate_data(child_data, direct_return=self.direct_return)
        if not child_data:
            return self.aggregate_data({ROOT_NODE_KEY: root_data}, direct_return=self.direct_return)
        assert isinstance(root_data, Mapping), (
            f"root_node '{self.root_node.name}' must return a mapping for the aggregated "
            f"channel when combined with named child nodes, got {type(root_data).__name__}."
        )
        return {**child_data, **root_data}

    def _refresh_spaces(self) -> None:
        """Re-aggregate spaces from child nodes (and the optional root node) and
        cache them as instance attributes.

        Called once at construction (for a preliminary snapshot that may include
        ``None`` placeholders for nodes whose spaces aren't yet known, e.g. robots
        before their scene is built) and again at the end of ``after_reload`` once
        every child has finished its post-build initialisation and set its final spaces.

        The root node's space contributes at the top level of the aggregated
        spaces (see the class docstring), i.e. its own name is never used as a
        nesting key for the ``context_space`` / ``observation_space`` /
        ``action_space`` it provides.

        Subclasses may override this to implement a different aggregation strategy
        (e.g. :class:`FlatCombinedWorldNode` merges keys instead of nesting them).
        """
        _, self.context_space = self._aggregate_channel_spaces('context_space', 'context')
        _, self.observation_space = self._aggregate_channel_spaces('observation_space', 'observation')
        self._action_node_name_direct, self.action_space = self._aggregate_channel_spaces('action_space', 'action')

    # ========== Cache management (stage-1) ==========
    def _build_dispatch_caches(self) -> None:
        """Build priority-order and per-priority participant caches.

        These depend only on child-node priority sets and routing ratios, which
        are fixed at construction time, so this method is called once from
        ``__init__``.
        """
        # Sorted priority orders (descending) per phase
        self._cached_priority_orders: Dict[str, List[int]] = {}
        # (phase, priority) -> list of participating child nodes
        self._cached_participants: Dict[Tuple[str, int], List[WorldNode]] = {}
        # phase -> {priority -> [(node, ratio, effective_dt), ...]}
        # Only built for pre_step / post_step where the runtime needs ratio + dt.
        self._cached_step_entries: Dict[str, Dict[int, List[Tuple[WorldNode, int, Optional[float]]]]] = {}
        # Aggregated priority sets (union of child priority sets per attr), cached
        # so the aggregated priority properties don't recompute unions on each access.
        self._cached_aggregated_priorities: Dict[str, Set[int]] = {}
        # Info-node list: every node (children and the optional root) participates
        # in get_info() (no flag gates it), so this is static and fixed at
        # construction.
        self._cached_info_nodes: List[WorldNode] = list(self._all_nodes)

        for phase, attr in _PHASE_ATTR_MAP.items():
            aggregated_prios = self._collect_priorities(self._all_nodes, attr)
            # Always reserve the internal reset priority for after_reset /
            # after_reload so the combined node's routing state is reset even
            # when no child registers a hook at these phases.
            if attr in ('after_reset_priorities', 'after_reload_priorities'):
                aggregated_prios = aggregated_prios | {self._INTERNAL_RESET_PRIORITY}
            order = sorted(aggregated_prios, reverse=True)
            self._cached_priority_orders[phase] = order
            # Cache the aggregated (unioned) priority set for the property accessor.
            self._cached_aggregated_priorities[attr] = aggregated_prios

            for p in order:
                participants_at_p = [n for n in self._all_nodes if p in getattr(n, attr)]
                self._cached_participants[(phase, p)] = participants_at_p

            # For pre_step / post_step, build enriched entries
            if phase in ('pre_step', 'post_step'):
                step_entries: Dict[int, List[Tuple[WorldNode, int, Optional[float]]]] = {}
                for p in order:
                    step_entries[p] = [
                        (n, self._update_ratios[self._node_key(n)], n.effective_update_timestep)
                        for n in self._cached_participants[(phase, p)]
                    ]
                self._cached_step_entries[phase] = step_entries

    def _refresh_space_caches(self) -> None:
        """Rebuild node-list caches that depend on node spaces / signal flags.

        Called once at construction and again after the final ``after_reload``
        priority, because child nodes may only set their final spaces during
        ``after_reload`` (e.g. robots whose DOF count is only known after the
        scene is compiled). The optional root node is included exactly like a
        child.
        """
        self._cached_context_nodes: List[WorldNode] = [
            n for n in self._all_nodes if n.context_space is not None
        ]
        self._cached_observation_nodes: List[WorldNode] = [
            n for n in self._all_nodes if n.observation_space is not None
        ]
        self._cached_action_nodes: List[WorldNode] = [
            n for n in self._all_nodes if n.action_space is not None
        ]
        self._cached_reward_nodes: List[WorldNode] = [
            n for n in self._all_nodes if n.has_reward
        ]
        self._cached_termination_nodes: List[WorldNode] = [
            n for n in self._all_nodes if n.has_termination_signal
        ]
        self._cached_truncation_nodes: List[WorldNode] = [
            n for n in self._all_nodes if n.has_truncation_signal
        ]

    def get_priority_order(self, phase: str) -> List[int]:
        """Return the cached descending priority order for a named lifecycle phase.

        Supported phase names: ``'reload'``, ``'after_reload'``, ``'reset'``,
        ``'after_reset'``, ``'pre_step'``, ``'post_step'``.
        """
        return self._cached_priority_orders[phase]

    # ========== Node query methods ==========
    def get_node(self, nested_keys: Union[str, Sequence[str]]) -> Optional[WorldNode]:
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

    def get_nodes_by_fn(self, fn: Callable[[WorldNode], bool]) -> list[WorldNode]:
        """Return every node in the subtree that satisfies ``fn`` (root included)."""
        result: list[WorldNode] = []
        if fn(self):
            result.append(self)
        for node in self._all_nodes:
            result.extend(node.get_nodes_by_fn(fn))
        return result

    @property
    def world(self) -> World[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.nodes[0].world

    @property
    def control_timestep(self) -> Optional[float]:
        return self._smallest_control_ts

    @property
    def update_timestep(self) -> Optional[float]:
        return self._smallest_update_ts

    @property
    def effective_update_timestep(self) -> Optional[float]:
        return self._smallest_update_ts

    # ========== Aggregated priority properties ==========
    @staticmethod
    def _collect_priorities(nodes, attr_name) -> Set[int]:
        return set().union(*(getattr(node, attr_name) for node in nodes))

    @property
    def reset_priorities(self) -> Set[int]:
        return self._cached_aggregated_priorities['reset_priorities']

    @property
    def reload_priorities(self) -> Set[int]:
        return self._cached_aggregated_priorities['reload_priorities']

    @property
    def after_reset_priorities(self) -> Set[int]:
        return self._cached_aggregated_priorities['after_reset_priorities']

    @property
    def after_reload_priorities(self) -> Set[int]:
        return self._cached_aggregated_priorities['after_reload_priorities']

    @property
    def pre_environment_step_priorities(self) -> Set[int]:
        return self._cached_aggregated_priorities['pre_environment_step_priorities']

    @property
    def post_environment_step_priorities(self) -> Set[int]:
        return self._cached_aggregated_priorities['post_environment_step_priorities']

    # ========== Lifecycle methods ==========
    def pre_environment_step(self, dt, *, priority : int = 0):
        """Dispatch pre-step callbacks to the nodes (children and the optional
        root node) at the matching frequency."""
        pre_order = self._cached_priority_orders['pre_step']
        # Advance the shared update counter exactly once per update substep:
        # at the top pre_step priority when any child registers one. When no
        # child registers a pre_step priority (composition with no pre-step
        # priorities), the counter is advanced in post_environment_step
        # instead, so it is NOT advanced here.
        if pre_order and priority == pre_order[0]:
            self._update_substeps = (self._update_substeps % self._update_period) + 1

        for node, ratio, eff_dt in self._cached_step_entries['pre_step'].get(priority, ()):
            if (self._update_substeps - 1) % ratio == 0:
                node.pre_environment_step(eff_dt, priority=priority)
    
    def get_context(self):
        assert self.context_space is not None, "Context space is None, cannot get context."
        return self._aggregate_channel_data(self._cached_context_nodes, 'get_context')

    def get_observation(self):
        assert self.observation_space is not None, "Observation space is None, cannot get observation."
        return self._aggregate_channel_data(self._cached_observation_nodes, 'get_observation')
    
    def get_reward(self):
        assert self.has_reward, "This node does not provide a reward."
        if self.world.batch_size is None:
            return sum(node.get_reward() for node in self._cached_reward_nodes)
        else:
            rewards = self.backend.zeros((self.world.batch_size,), dtype=self.backend.default_floating_dtype, device=self.device)
            for node in self._cached_reward_nodes:
                rewards = rewards + node.get_reward()
            return rewards
    
    def get_termination(self):
        assert self.has_termination_signal, "This node does not provide a termination signal."
        if self.world.batch_size is None:
            return any(node.get_termination() for node in self._cached_termination_nodes)
        else:
            terminations = self.backend.zeros((self.world.batch_size,), dtype=self.backend.default_boolean_dtype, device=self.device)
            for node in self._cached_termination_nodes:
                terminations = self.backend.logical_or(terminations, node.get_termination())
            return terminations
        
    def get_truncation(self):
        assert self.has_truncation_signal, "This node does not provide a truncation signal."
        if self.world.batch_size is None:
            return any(node.get_truncation() for node in self._cached_truncation_nodes)
        else:
            truncations = self.backend.zeros((self.world.batch_size,), dtype=self.backend.default_boolean_dtype, device=self.device)
            for node in self._cached_truncation_nodes:
                truncations = self.backend.logical_or(truncations, node.get_truncation())
            return truncations
    
    def get_info(self) -> Optional[Dict[str, Any]]:
        infos = {}
        for node in self._cached_info_nodes:
            info = node.get_info()
            if info is not None:
                # Info nests by node name (display key), so the root node uses
                # its own name here -- never the reserved internal key.
                infos[node.name] = info
            
        return self.aggregate_data(
            infos,
            direct_return=False
        )

    def render(self):
        if not self.can_render:
            return None
        if len(self._renderable_nodes) == 1 and self._true_render_mode != 'dict':
            return self._renderable_nodes[0].render()
        result = {}
        for node in self._renderable_nodes:
            r = node.render()
            if r is None:
                continue
            if isinstance(r, dict):
                for k, v in r.items():
                    result[f"{node.name}.{k}"] = v
            else:
                result[node.name] = r
        return result if result else None

    def _split_child_actions(self, action: CombinedDataT) -> Dict[str, Any]:
        """Map an incoming combined action to per-node actions for the nodes
        that receive a *new* action this call, keyed by node name.

        The default implementation routes by node name (nested actions) or
        passes the action through to the single direct-return action node.
        Subclasses may override this to implement a different routing scheme
        (e.g. :class:`FlatCombinedWorldNode` slices flat action dicts by data
        keys); :meth:`set_next_action` owns the counter / cache / ratio
        dispatch orchestration and should not need to be overridden.

        This contract stays children-only: an optional ``root_node`` is handled
        inline by :meth:`set_next_action`, which extracts the action keys no
        child claims and stores them under the reserved root key.
        """
        if self._action_node_name_direct is not None:
            return {self._action_node_name_direct: action}
        assert isinstance(action, Mapping), "Action must be a mapping when there are multiple action spaces."
        return action

    def set_next_action(self, action):
        """Route combined actions to child nodes (and the optional root node),
        respecting per-node control rates.

        The incoming action is split by :meth:`_split_child_actions` (children
        only). When a ``root_node`` owns an action space and no single node is
        the direct-action passthrough provider, the keys that do not belong to
        any named child are the root's action keys: they accumulate into the
        root's cache entry under the reserved key as a dict. Every consumer
        (children and root) is then dispatched at its own ratio tick, and only
        once it has received at least one action (hold-last semantics, so a
        child-only step re-delivers the root's previous action at its tick and
        vice versa).
        """
        if self.action_space is None:
            assert action is None, "Cannot provide an action when action_space is None."
            return
        self._action_substeps = (self._action_substeps % self._action_period) + 1

        child_actions = self._split_child_actions(action)

        # Root remainder routing, inline so that `_split_child_actions` keeps its
        # children-only contract. Flat subclasses already emit node-keyed slices
        # (including the reserved root key), in which case this is a no-op
        # because the reserved key is excluded from the remainder.
        root = self.root_node
        if root is not None and root.action_space is not None \
                and self._action_node_name_direct is None and isinstance(child_actions, Mapping):
            child_names = {node.name for node in self.nodes}
            root_actions = {
                key: value for key, value in child_actions.items()
                if key not in child_names and key != ROOT_NODE_KEY
            }
            if root_actions:
                self._cached_actions[ROOT_NODE_KEY] = root_actions

        for node in self._cached_action_nodes:
            key = self._node_key(node)
            if key in child_actions:
                self._cached_actions[key] = child_actions[key]
            ratio = self._action_ratios[key]
            # Dispatch at the node's control-rate tick, but only once it has
            # received at least one action (sparse/partial actions are valid).
            if (self._action_substeps - 1) % ratio == 0 and key in self._cached_actions:
                node.set_next_action(self._cached_actions[key])
    
    def post_environment_step(self, dt, *, priority : int = 0):
        """Dispatch post-step callbacks to the nodes (children and the optional
        root node) at the matching frequency."""
        pre_order = self._cached_priority_orders['pre_step']
        post_order = self._cached_priority_orders['post_step']
        # Advance the shared update counter here only when no child registers a
        # pre_step priority (composition with no pre-step priorities);
        # otherwise it was already advanced in pre_environment_step.
        if not pre_order and post_order and priority == post_order[0]:
            self._update_substeps = (self._update_substeps % self._update_period) + 1

        for node, ratio, eff_dt in self._cached_step_entries['post_step'].get(priority, ()):
            if (self._update_substeps - 1) % ratio == 0:
                node.post_environment_step(eff_dt, priority=priority)

        # With a single shared counter the pre/post balance is trivially
        # consistent (there is only one counter), so no equality assert is
        # needed here.
    
    def reset(self, *, priority : int = 0, seed = None, mask = None, pernode_kwargs : Dict[str, Any] = {}):
        """Forward reset calls to the nodes (children and the optional root node)
        that participate at ``priority``."""
        for node in self._cached_participants.get(('reset', priority), ()):
            node.reset(
                priority=priority,
                seed=seed,
                mask=mask,
                **pernode_kwargs.get(node.name, {})
            )

    def reload(self, *, priority : int = 0, seed = None, mask = None, pernode_kwargs : Dict[str, Any] = {}):
        """Forward reload calls to the nodes (children and the optional root node)
        that participate at ``priority``."""
        for node in self._cached_participants.get(('reload', priority), ()):
            node.reload(
                priority=priority,
                seed=seed,
                mask=mask,
                **pernode_kwargs.get(node.name, {})
            )

    def after_reset(self, *, priority : int = 0, mask = None):
        """Forward post-reset hooks and reset internal routing counters."""
        order = self._cached_priority_orders['after_reset']
        # Reset routing state at the top after_reset priority (existing
        # behaviour) and at the reserved internal priority (guarantees a reset
        # even when no child registers an after_reset priority). Both are
        # idempotent.
        if (order and priority == order[0]) or priority == self._INTERNAL_RESET_PRIORITY:
            self._update_substeps = 0
            self._action_substeps = 0
            self._cached_actions.clear()

        for node in self._cached_participants.get(('after_reset', priority), ()):
            node.after_reset(priority=priority, mask=mask)

    def after_reload(self, *, priority : int = 0, mask = None):
        """Call after_reload on the nodes (children and the optional root node).
        Similar to after_reset but for the reload flow."""
        order = self._cached_priority_orders['after_reload']
        if (order and priority == order[0]) or priority == self._INTERNAL_RESET_PRIORITY:
            self._update_substeps = 0
            self._action_substeps = 0
            self._cached_actions.clear()

        for node in self._cached_participants.get(('after_reload', priority), ()):
            node.after_reload(priority=priority, mask=mask)

        # After child nodes finish their lowest-priority after_reload (the last call
        # in the priority sequence), re-aggregate spaces so that the cached
        # action_space / observation_space / context_space reflect the final,
        # post-build spaces that nodes like robots set during after_reload.
        # Also refresh space-dependent dispatch caches so that subsequent
        # get_observation / get_context / set_next_action calls see the updated
        # child spaces.
        if order and priority == order[-1]:
            self._refresh_spaces()
            self._refresh_space_caches()

    def close(self):
        """Close every child node and the optional root node."""
        for node in self._all_nodes:
            node.close()
