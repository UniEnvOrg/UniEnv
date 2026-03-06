from typing import Optional, Dict, Set, Mapping, Any, Tuple, Union, Iterable, Generic, Sequence, Callable
from math import lcm
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace
from unienv_interface.utils.control_util import find_best_timestep

from ..world import World
from ..node import WorldNode, ContextType, ObsType, ActType

CombinedDataT = Union[Dict[str, Any], BArrayType]

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
    """

    supported_render_modes = ('dict', 'auto')

    def __init__(
        self,
        name : str,
        nodes : Iterable[WorldNode[Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        direct_return : bool = True,
        render_mode : Optional[str] = 'auto',
    ):
        nodes = list(nodes)
        if len(nodes) == 0:
            raise ValueError("At least one node is required to create a CombinedWorldNode.")
        
        # Check that all nodes have the same world
        first_node = nodes[0]
        for node in nodes[1:]:
            assert node.world is first_node.world, "All nodes must belong to the same world."
        # Check that all nodes have unique names
        names = [node.name for node in nodes]
        if len(names) != len(set(names)):
            raise ValueError("All nodes must have unique names.")
        self.nodes = nodes

        self.has_reward = any(node.has_reward for node in nodes)
        self.has_termination_signal = any(node.has_termination_signal for node in nodes)
        self.has_truncation_signal = any(node.has_truncation_signal for node in nodes)

        # Save attributes
        self.name = name
        self.direct_return = direct_return

        # Aggregate spaces (preliminary snapshot; may include None/unbounded placeholders
        # for nodes like robots whose final spaces are only known after after_reload.
        # _refresh_spaces() is called again at the end of after_reload to capture finals.)
        self._refresh_spaces()

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

        # Substep counters
        self._pre_substeps = 0
        self._post_substeps = 0
        self._action_substeps = 0
        self._cached_actions = {}

    @staticmethod
    def aggregate_spaces(
        spaces : Dict[str, Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]]],
        direct_return : bool = True,
    ) -> Tuple[
        Optional[str],
        Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]
    ]:
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
        if len(data) == 0:
            return None
        elif len(data) == 1 and direct_return:
            return next(iter(data.values()))
        else:
            return data

    def _refresh_spaces(self) -> None:
        """Re-aggregate spaces from child nodes and cache them as instance attributes.

        Called once at construction (for a preliminary snapshot that may include
        ``None`` placeholders for nodes whose spaces aren't yet known, e.g. robots
        before their scene is built) and again at the end of ``after_reload`` once
        every child has finished its post-build initialisation and set its final spaces.

        Subclasses may override this to implement a different aggregation strategy
        (e.g. :class:`FlatCombinedWorldNode` merges keys instead of nesting them).
        """
        _, self.context_space = self.aggregate_spaces(
            {node.name: node.context_space for node in self.nodes if node.context_space is not None},
            direct_return=self.direct_return,
        )
        _, self.observation_space = self.aggregate_spaces(
            {node.name: node.observation_space for node in self.nodes if node.observation_space is not None},
            direct_return=self.direct_return,
        )
        self._action_node_name_direct, self.action_space = self.aggregate_spaces(
            {node.name: node.action_space for node in self.nodes if node.action_space is not None},
            direct_return=self.direct_return,
        )

    # ========== Node query methods ==========
    def get_node(self, nested_keys: Union[str, Sequence[str]]) -> Optional[WorldNode]:
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
        result: list[WorldNode] = []
        if fn(self):
            result.append(self)
        for node in self.nodes:
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
        return self._collect_priorities(self.nodes, 'reset_priorities')

    @property
    def reload_priorities(self) -> Set[int]:
        return self._collect_priorities(self.nodes, 'reload_priorities')

    @property
    def after_reset_priorities(self) -> Set[int]:
        return self._collect_priorities(self.nodes, 'after_reset_priorities')

    @property
    def after_reload_priorities(self) -> Set[int]:
        return self._collect_priorities(self.nodes, 'after_reload_priorities')

    @property
    def pre_environment_step_priorities(self) -> Set[int]:
        return self._collect_priorities(self.nodes, 'pre_environment_step_priorities')

    @property
    def post_environment_step_priorities(self) -> Set[int]:
        return self._collect_priorities(self.nodes, 'post_environment_step_priorities')

    # ========== Lifecycle methods ==========
    def pre_environment_step(self, dt, *, priority : int = 0):
        all_prios = self.pre_environment_step_priorities
        if all_prios and priority == max(all_prios):
            self._pre_substeps = (self._pre_substeps % self._update_period) + 1

        for node in self.nodes:
            if priority in node.pre_environment_step_priorities:
                ratio = self._update_ratios[node.name]
                if (self._pre_substeps - 1) % ratio == 0:
                    node.pre_environment_step(node.effective_update_timestep, priority=priority)
    
    def get_context(self):
        assert self.context_space is not None, "Context space is None, cannot get context."
        return self.aggregate_data(
            {
                node.name: node.get_context()
                for node in self.nodes
                if node.context_space is not None
            },
            direct_return=self.direct_return,
        )

    def get_observation(self):
        assert self.observation_space is not None, "Observation space is None, cannot get observation."
        return self.aggregate_data(
            {
                node.name: node.get_observation() 
                for node in self.nodes 
                if node.observation_space is not None
            },
            direct_return=self.direct_return,
        )
    
    def get_reward(self):
        assert self.has_reward, "This node does not provide a reward."
        if self.world.batch_size is None:
            return sum(
                node.get_reward() 
                for node in self.nodes 
                if node.has_reward
            )
        else:
            rewards = self.backend.zeros((self.world.batch_size,), dtype=self.backend.default_floating_dtype, device=self.device)
            for node in self.nodes:
                if node.has_reward:
                    rewards = rewards + node.get_reward()
            return rewards
    
    def get_termination(self):
        assert self.has_termination_signal, "This node does not provide a termination signal."
        if self.world.batch_size is None:
            return any(
                node.get_termination() 
                for node in self.nodes 
                if node.has_termination_signal
            )
        else:
            terminations = self.backend.zeros((self.world.batch_size,), dtype=self.backend.default_boolean_dtype, device=self.device)
            for node in self.nodes:
                if node.has_termination_signal:
                    terminations = self.backend.logical_or(terminations, node.get_termination())
            return terminations
        
    def get_truncation(self):
        assert self.has_truncation_signal, "This node does not provide a truncation signal."
        if self.world.batch_size is None:
            return any(
                node.get_truncation() 
                for node in self.nodes 
                if node.has_truncation_signal
            )
        else:
            truncations = self.backend.zeros((self.world.batch_size,), dtype=self.backend.default_boolean_dtype, device=self.device)
            for node in self.nodes:
                if node.has_truncation_signal:
                    truncations = self.backend.logical_or(truncations, node.get_truncation())
            return truncations
    
    def get_info(self) -> Optional[Dict[str, Any]]:
        infos = {}
        for node in self.nodes:
            info = node.get_info()
            if info is not None:
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

    def set_next_action(self, action):
        assert self.action_space is not None, "Action space is None, cannot set action."
        self._action_substeps = (self._action_substeps % self._action_period) + 1

        if self._action_node_name_direct is not None:
            child_actions = {self._action_node_name_direct: action}
        else:
            assert isinstance(action, Mapping), "Action must be a mapping when there are multiple action spaces."
            child_actions = action

        for node in self.nodes:
            if node.action_space is not None:
                if node.name in child_actions:
                    self._cached_actions[node.name] = child_actions[node.name]
                ratio = self._action_ratios[node.name]
                if (self._action_substeps - 1) % ratio == 0:
                    node.set_next_action(self._cached_actions[node.name])
    
    def post_environment_step(self, dt, *, priority : int = 0):
        all_prios = self.post_environment_step_priorities
        if all_prios and priority == max(all_prios):
            self._post_substeps = (self._post_substeps % self._update_period) + 1

        for node in self.nodes:
            if priority in node.post_environment_step_priorities:
                ratio = self._update_ratios[node.name]
                if (self._post_substeps - 1) % ratio == 0:
                    node.post_environment_step(node.effective_update_timestep, priority=priority)

        if all_prios and priority == max(all_prios):
            assert self._pre_substeps == self._post_substeps
    
    def reset(self, *, priority : int = 0, seed = None, mask = None, pernode_kwargs : Dict[str, Any] = {}):
        for node in self.nodes:
            if priority in node.reset_priorities:
                node.reset(
                    priority=priority,
                    seed=seed,
                    mask=mask,
                    **pernode_kwargs.get(node.name, {})
                )

    def reload(self, *, priority : int = 0, seed = None, mask = None, pernode_kwargs : Dict[str, Any] = {}):
        for node in self.nodes:
            if priority in node.reload_priorities:
                node.reload(
                    priority=priority,
                    seed=seed,
                    mask=mask,
                    **pernode_kwargs.get(node.name, {})
                )

    def after_reset(self, *, priority : int = 0, mask = None):
        all_prios = self.after_reset_priorities
        if all_prios and priority == max(all_prios):
            self._pre_substeps = 0
            self._post_substeps = 0
            self._action_substeps = 0
            self._cached_actions.clear()

        for node in self.nodes:
            if priority in node.after_reset_priorities:
                node.after_reset(priority=priority, mask=mask)

    def after_reload(self, *, priority : int = 0, mask = None):
        """Call after_reload on child nodes. Similar to after_reset but for reload flow."""
        all_prios = self.after_reload_priorities
        if all_prios and priority == max(all_prios):
            self._pre_substeps = 0
            self._post_substeps = 0
            self._action_substeps = 0
            self._cached_actions.clear()

        for node in self.nodes:
            if priority in node.after_reload_priorities:
                node.after_reload(priority=priority, mask=mask)

        # After child nodes finish their lowest-priority after_reload (the last call
        # in the priority sequence), re-aggregate spaces so that the cached
        # action_space / observation_space / context_space reflect the final,
        # post-build spaces that nodes like robots set during after_reload.
        if all_prios and priority == min(all_prios):
            self._refresh_spaces()

    def close(self):
        for node in self.nodes:
            node.close()
