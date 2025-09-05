from typing import Optional, Dict, Mapping, Any, Tuple, Union, Iterable
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace
from unienv_interface.utils.control_util import find_best_timestep

from .world import World
from .node import WorldNode, ContextType, ObsType, ActType

CombinedDataT = Union[Dict[str, Any], BArrayType]

class CombinedWorldNode(WorldNode[
    Optional[CombinedDataT], CombinedDataT, CombinedDataT,
    BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    """
    A WorldNode that combines multiple WorldNodes into one node, using a dictionary to store the data from each node.
    The observation, reward, termination, truncation, and info are combined from all child nodes.
    The keys in the dictionary are the names of the child nodes.
    If there is only one child node that supports value and `direct_return` is set to True, the value is returned directly instead of a dictionary.
    """

    def __init__(
        self,
        name : str,
        nodes : Iterable[WorldNode[WorldNode[Any, Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType], Any, Any, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        direct_return : bool = True,
    ):
        nodes = list(nodes)
        if len(nodes) == 0:
            raise ValueError("At least one node is required to create a CombinedWorldNode.")
        
        # Check that all nodes have the same world
        first_node = nodes[0]
        for node in nodes[1:]:
            assert node.world is first_node.world, "All nodes must belong to the same world."
            assert node.control_timestep == first_node.control_timestep, "All nodes must have the same control timestep."
        # Check that all nodes have unique names
        names = [node.name for node in nodes]
        if len(names) != len(set(names)):
            raise ValueError("All nodes must have unique names.")
        self.nodes = nodes

        # Aggregate Spaces
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

        # Save attributes
        self.name = name
        self.direct_return = direct_return

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

    @property
    def world(self) -> World[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.nodes[0].world
    
    @property
    def control_timestep(self) -> Optional[float]:
        return self.nodes[0].control_timestep
    
    def pre_environment_step(self, dt):
        for node in self.nodes:
            node.pre_environment_step(dt)
    
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
            terminations = self.backend.zeros((self.world.batch_size,), dtype=self.backend.default_bool_dtype, device=self.device)
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
            truncations = self.backend.zeros((self.world.batch_size,), dtype=self.backend.default_bool_dtype, device=self.device)
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
    
    def set_next_action(self, action):
        assert self.action_space is not None, "Action space is None, cannot set action."
        if self._action_node_name_direct is not None:
            for node in self.nodes:
                if node.name == self._action_node_name_direct:
                    node.set_next_action(action)
                    break
        else:
            assert isinstance(action, Mapping), "Action must be a mapping when there are multiple action spaces."
            for node in self.nodes:
                if node.action_space is not None:
                    assert node.name in action, f"Action for node {node.name} is missing."
                    node.set_next_action(action[node.name])
    
    def post_environment_step(self, dt):
        for node in self.nodes:
            node.post_environment_step(dt)
    
    def reset(self, *, seed = None, mask = None, pernode_kwargs : Dict[str, Any] = {}):
        for node in self.nodes:
            node.reset(
                seed=seed,
                mask=mask,
                **pernode_kwargs.get(node.name, {})
            )
    
    def after_reset(self, *, mask = None):
        contexts = {}
        observations = {}
        infos = {}


        for node in self.nodes:
            context, observation, info = node.after_reset(mask=mask)
            if context is not None:
                contexts[node.name] = context
            if observation is not None:
                observations[node.name] = observation
            if info is not None:
                infos[node.name] = info

        return self.aggregate_data(
            contexts,
            direct_return=self.direct_return,
        ), self.aggregate_data(
            observations,
            direct_return=self.direct_return,
        ), self.aggregate_data(
            infos,
            direct_return=False
        )
    

    def close(self):
        for node in self.nodes:
            node.close()