"""FlatCombinedFuncWorldNode - A functional node that flattens combined node data structures.

Unlike CombinedFuncWorldNode which nests data under node names as keys, 
FlatCombinedFuncWorldNode merges dictionary values directly, requiring all 
nodes to have DictSpace observations/actions/contexts with unique keys.
"""
from typing import Optional, Dict, Any, Union, Iterable, Mapping, Sequence, Set
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace

from ..funcnode import FuncWorldNode
from .combined_funcnode import CombinedFuncWorldNode, CombinedDataT, CombinedNodeStateT, ROOT_NODE_KEY
from ..funcworld import WorldStateT


class FlatCombinedFuncWorldNode(CombinedFuncWorldNode[WorldStateT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """
    A FuncWorldNode that combines multiple FuncWorldNodes and flattens their data.
    
    Unlike CombinedFuncWorldNode which stores data as {node_name: {key: value}},
    FlatCombinedFuncWorldNode merges dictionaries directly as {key1: value1, key2: value2}.
    
    This requires:
    - All nodes with observation/action/context spaces must use DictSpace
    - Keys across all nodes must be unique (no overlaps)
    
    The node names are only used for identification, not for nesting data.

    Root node
    ---------
    An optional ``root_node`` (passed separately, sharing the same world and not
    listed in ``nodes``) is handled as one more flat contributor:

    - Its context / observation spaces and data are appended to the flatten list,
      so its keys appear directly at the top level next to the children's keys (an
      overlapping key raises ``ValueError``). When the root is the only
      contributor its space/data is returned unwrapped, mirroring the existing
      single-child flat behavior.
    - For actions, every key not claimed by any child's action space is routed to
      the root under the reserved internal key (its state lives under that same
      key), while claimed keys are sliced to their children exactly as before.
      Without a ``root_node`` such unclaimed keys are rejected with a
      ``ValueError``.
    - Rewards / signals / info / render and the lifecycle hooks include the root
      like any other node (info and render use the root's own ``name``).
    """

    def __init__(
        self,
        name: str,
        nodes: Iterable[FuncWorldNode[WorldStateT, Any, Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        render_mode: Optional[str] = 'auto',
        root_node: Optional[FuncWorldNode[WorldStateT, Any, Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]] = None,
    ):
        """
        Initialize a FlatCombinedFuncWorldNode.
        
        Args:
            name: Name of this combined node
            nodes: Iterable of nodes to combine
            render_mode: Render mode ('dict', 'auto', or specific mode)
            root_node: Optional node merged at the top level of the flattened
                data instead of nesting under its own name

        Raises:
            ValueError: If node spaces have overlapping keys or non-DictSpace types
        """
        # Always set direct_return=False for flat combination
        # We'll handle the flattening ourselves
        super().__init__(
            name=name,
            nodes=nodes,
            direct_return=False,
            render_mode=render_mode,
            root_node=root_node,
        )
        
        # Validate and flatten spaces (the optional root node takes part in the
        # flattening exactly like a child node)
        self.context_space = self._flatten_channel_spaces('context_space')
        self.observation_space = self._flatten_channel_spaces('observation_space')
        self.action_space = self._flatten_channel_spaces('action_space')

    def _flatten_channel_spaces(
        self,
        space_attr: str,
    ) -> Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]]:
        """Flatten one space channel over the named children and the optional root node."""
        spaces = [
            getattr(node, space_attr) for node in self.nodes if getattr(node, space_attr) is not None
        ]
        if self.root_node is not None and getattr(self.root_node, space_attr) is not None:
            spaces.append(getattr(self.root_node, space_attr))
        return self._flatten_spaces(spaces)

    @staticmethod
    def _flatten_spaces(
        spaces: list[Space[Any, BDeviceType, BDtypeType, BRNGType]],
    ) -> Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]]:
        """
        Flatten a list of spaces by merging their keys.
        
        Args:
            spaces: List of spaces to flatten
            
        Returns:
            Merged DictSpace or None if no spaces
            
        Raises:
            ValueError: If spaces are not DictSpaces or have overlapping keys
        """
        if not spaces or len(spaces) == 0:
            return None
            
        assert len(spaces) == 1 or all(isinstance(space, DictSpace) for space in spaces), (
            f"All spaces must be DictSpace for FlatCombinedFuncWorldNode or there must be only one space. "
            f"Found non-DictSpace in spaces."
        )
        
        if len(spaces) == 1:
            return spaces[0]

        merged_spaces: Dict[str, Space[Any, BDeviceType, BDtypeType, BRNGType]] = {}
        for space in spaces:
            assert isinstance(space, DictSpace), (
                f"All spaces must be DictSpace for FlatCombinedFuncWorldNode. "
                f"Found non-DictSpace: {type(space).__name__}"
            )
            for key in space.spaces.keys():
                if key in merged_spaces:
                    raise ValueError(
                        f"Overlapping key '{key}' found in spaces of FlatCombinedFuncWorldNode. "
                        f"Keys must be unique across all nodes. "
                        f"Conflict found in space with keys: {list(space.spaces.keys())}"
                    )
            merged_spaces.update(space.spaces)
            
        # Get backend from first space
        backend = spaces[0].backend
        return DictSpace(backend, merged_spaces)

    @staticmethod
    def _flatten_data(
        all_data : Sequence[Any]
    ) -> Optional[Union[Dict[str, Any], Any]]:
        """
        Flatten a list of data items by merging dictionaries.
        
        Args:
            all_data: List of data items (dicts) to flatten
            
        Returns:
            Merged dictionary or single item if only one
            
        Raises:
            RuntimeError: If data items are not dicts or have overlapping keys
        """
        if not all_data:
            return None
        if len(all_data) == 1:
            return all_data[0]
        
        merged_data: Dict[str, Any] = {}
        for data in all_data:
            if not isinstance(data, dict):
                raise RuntimeError(
                    f"Expected dict data for flattening in FlatCombinedFuncWorldNode, got {type(data).__name__}. "
                    f"All data items must be dictionaries."
                )
            for key in data.keys():
                if key in merged_data:
                    raise RuntimeError(
                        f"Overlapping key '{key}' found in data during flattening in FlatCombinedFuncWorldNode. "
                        f"Keys must be unique across all nodes. "
                        f"Conflict found in data with keys: {list(data.keys())}"
                    )
            merged_data.update(data)
        return merged_data

    def get_context(
        self,
        world_state: WorldStateT,
        node_state: CombinedNodeStateT
    ) -> Optional[CombinedDataT]:
        """Get context by flattening all node contexts into one dictionary."""
        if self.context_space is None:
            return None
            
        all_contexts = [
            node.get_context(world_state, node_state[self._node_key(node)])
            for node in self._all_nodes
            if node.context_space is not None
        ]
        return self._flatten_data(all_contexts)

    def get_observation(
        self,
        world_state: WorldStateT,
        node_state: CombinedNodeStateT
    ) -> CombinedDataT:
        """Get observation by flattening all node observations into one dictionary."""
        assert self.observation_space is not None, "Observation space is None, cannot get observation."
        
        all_observations = [
            node.get_observation(world_state, node_state[self._node_key(node)])
            for node in self._all_nodes
            if node.observation_space is not None
        ]
        return self._flatten_data(all_observations)

    def get_info(
        self,
        world_state: WorldStateT,
        node_state: CombinedNodeStateT
    ) -> Optional[Dict[str, Any]]:
        """Get info by merging all node info dictionaries."""
        all_info = []
        for node in self._all_nodes:
            info = node.get_info(world_state, node_state[self._node_key(node)])
            if info is not None:
                all_info.append(info)
        return self._flatten_data(all_info)

    def _split_child_actions(self, action: CombinedDataT) -> Dict[str, Any]:
        """Split a flat action dict into per-node slices keyed by internal node key.

        Each DictSpace child receives the sub-dict of its own keys (only when
        at least one of its keys is present, so partial actions are valid); a
        single non-DictSpace action node receives the entire action. Action keys
        that no child claims are routed to the ``root_node`` (when one provides
        an action space) under the reserved key; if no root node is provided they
        are rejected with a ``ValueError``. The inherited
        :meth:`CombinedFuncWorldNode.set_next_action` handles caching and
        per-node control-rate dispatch.
        """
        child_actions: Dict[str, Any] = {}
        claimed_keys: Set[str] = set()
        has_mapping_action = isinstance(action, Mapping)
        for node in self._all_nodes:
            if node is self.root_node or node.action_space is None:
                continue
            if isinstance(node.action_space, DictSpace):
                assert has_mapping_action, (
                    f"Action must be a mapping to route keys to DictSpace child node "
                    f"'{node.name}' of FlatCombinedFuncWorldNode, got {type(action).__name__}."
                )
                node_action = {key: action[key] for key in node.action_space.spaces.keys() if key in action}
                claimed_keys.update(node_action.keys())
                if node_action:
                    child_actions[node.name] = node_action
            else:
                # Single non-DictSpace action node receives the entire action.
                child_actions[node.name] = action

        root = self.root_node
        if root is not None and root.action_space is not None:
            if isinstance(root.action_space, DictSpace):
                assert has_mapping_action, (
                    f"Action must be a mapping to route keys to the DictSpace action space of "
                    f"root_node '{root.name}' of FlatCombinedFuncWorldNode, got {type(action).__name__}."
                )
                # Every key no child claims belongs to the root node.
                root_action = {
                    key: value for key, value in action.items() if key not in claimed_keys
                }
            else:
                # A non-DictSpace root action space is only reachable as the sole
                # action provider, in which case it receives the entire action.
                root_action = action
            if root_action:
                child_actions[ROOT_NODE_KEY] = root_action
        elif has_mapping_action:
            unclaimed_keys = sorted(
                key for key in action.keys() if key not in claimed_keys
            )
            if unclaimed_keys:
                raise ValueError(
                    f"Action key(s) {unclaimed_keys} of FlatCombinedFuncWorldNode are not claimed by "
                    f"any child node's action space and no root_node is provided to receive them."
                )
        return child_actions