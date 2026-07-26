"""FlatCombinedFuncWorldNode - A functional node that flattens combined node data structures.

Unlike CombinedFuncWorldNode which nests data under node names as keys, 
FlatCombinedFuncWorldNode merges dictionary values directly, requiring all 
nodes to have DictSpace observations/actions/contexts with unique keys.
"""
from typing import Optional, Dict, Any, Union, Iterable, Mapping, Sequence
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace

from ..funcnode import FuncWorldNode
from .combined_funcnode import CombinedFuncWorldNode, CombinedDataT, CombinedNodeStateT
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
    """

    def __init__(
        self,
        name: str,
        nodes: Iterable[FuncWorldNode[WorldStateT, Any, Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        render_mode: Optional[str] = 'auto',
    ):
        """
        Initialize a FlatCombinedFuncWorldNode.
        
        Args:
            name: Name of this combined node
            nodes: Iterable of nodes to combine
            render_mode: Render mode ('dict', 'auto', or specific mode)
            
        Raises:
            ValueError: If node spaces have overlapping keys or non-DictSpace types
        """
        # Always set direct_return=False for flat combination
        # We'll handle the flattening ourselves
        super().__init__(name=name, nodes=nodes, direct_return=False, render_mode=render_mode)
        
        # Validate and flatten spaces
        self.context_space = self._flatten_spaces(
            [node.context_space for node in self.nodes if node.context_space is not None],
            ignore_duplicate_keys=False
        )
        self.observation_space = self._flatten_spaces(
            [node.observation_space for node in self.nodes if node.observation_space is not None],
            ignore_duplicate_keys=False
        )
        self.action_space = self._flatten_spaces(
            [node.action_space for node in self.nodes if node.action_space is not None],
            ignore_duplicate_keys=True  # For actions, we allow overlapping keys since they will be routed to nodes
        )

    @staticmethod
    def _flatten_spaces(
        spaces: list[Space[Any, BDeviceType, BDtypeType, BRNGType]],
        ignore_duplicate_keys: bool = False,
    ) -> Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]]:
        """
        Flatten a list of spaces by merging their keys.
        
        Args:
            spaces: List of spaces to flatten
            ignore_duplicate_keys: Whether to ignore duplicate keys when merging spaces
            
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
                if key in merged_spaces and not ignore_duplicate_keys:
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
            
        all_contexts = []
        for node in self.nodes:
            if node.context_space is not None:
                context = node.get_context(world_state, node_state[node.name])
                all_contexts.append(context)
        return self._flatten_data(all_contexts)

    def get_observation(
        self,
        world_state: WorldStateT,
        node_state: CombinedNodeStateT
    ) -> CombinedDataT:
        """Get observation by flattening all node observations into one dictionary."""
        assert self.observation_space is not None, "Observation space is None, cannot get observation."
        
        all_observations = []
        for node in self.nodes:
            if node.observation_space is not None:
                obs = node.get_observation(world_state, node_state[node.name])
                all_observations.append(obs)
        return self._flatten_data(all_observations)

    def get_info(
        self,
        world_state: WorldStateT,
        node_state: CombinedNodeStateT
    ) -> Optional[Dict[str, Any]]:
        """Get info by merging all node info dictionaries."""
        all_info = []
        for node in self.nodes:
            info = node.get_info(world_state, node_state[node.name])
            if info is not None:
                all_info.append(info)
        return self._flatten_data(all_info)

    def _split_child_actions(self, action: CombinedDataT) -> Dict[str, Any]:
        """Split a flat action dict into per-node slices keyed by node name.

        Each DictSpace child receives the sub-dict of its own keys (only when
        at least one of its keys is present, so partial actions are valid); a
        single non-DictSpace action node receives the entire action. The
        inherited :meth:`CombinedFuncWorldNode.set_next_action` handles
        caching and per-node control-rate dispatch.
        """
        child_actions: Dict[str, Any] = {}
        for node in self.nodes:
            if node.action_space is None:
                continue
            if isinstance(node.action_space, DictSpace):
                assert isinstance(action, Mapping), (
                    f"Action must be a mapping to route keys to DictSpace child node "
                    f"'{node.name}' of FlatCombinedFuncWorldNode, got {type(action).__name__}."
                )
                node_action = {key: action[key] for key in node.action_space.spaces.keys() if key in action}
                if node_action:
                    child_actions[node.name] = node_action
            else:
                # Single non-DictSpace action node receives the entire action.
                child_actions[node.name] = action
        return child_actions