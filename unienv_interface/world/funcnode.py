from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union
from abc import ABC, abstractmethod
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from unienv_interface.env_base.env import ContextType, ObsType, ActType

from .funcworld import FuncWorld, WorldStateT

NodeStateT = TypeVar("NodeStateT")

class FuncWorldNode(ABC, Generic[
    WorldStateT, NodeStateT, 
    ContextType, ObsType, ActType, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    """
    Each `FuncWorldNode` in the simulated / real world will manage some aspect of the environment.
    How the methods in this class will be called once environment resets:
    `FuncWorld.reset(...)` -> `FuncWorldNode.reset(...)` -> `FuncWorldNode.after_reset(...)` -> `FuncWorldNode.get_observation(...)` -> World can start stepping normally
    How the methods in this class will be called during a environment step:
    `FuncWorldNode.set_next_action(...)` -> `FuncWorldNode.pre_environment_step(...)` -> `FuncWorld.step(...)` -> `FuncWorldNode.post_environment_step(...)` -> `FuncWorldNode.get_observation(...)` -> `FuncWorldNode.get_reward(...)` -> `FuncWorldNode.get_termination(...)` -> `FuncWorldNode.get_truncation(...)` -> `FuncWorldNode.get_info(...)`
    """

    name : str
    world : FuncWorld[WorldStateT, BArrayType, BDeviceType, BDtypeType, BRNGType]
    control_timestep : Optional[float] = None
    context_space : Optional[Space[ContextType, BDeviceType, BDtypeType, BRNGType]] = None
    observation_space : Optional[Space[ObsType, BDeviceType, BDtypeType, BRNGType]] = None
    action_space : Optional[Space[ActType, BDeviceType, BDtypeType, BRNGType]] = None
    has_reward : bool = False
    has_termination_signal : bool = False
    has_truncation_signal : bool = False

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.world.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.world.device

    @abstractmethod
    def initial(
        self,
        world_state : WorldStateT,
        *,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[WorldStateT, NodeStateT]:
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> Tuple[WorldStateT, NodeStateT]:
        """
        This method is called after `FuncWorld.reset(...)` has been called.
        """
        return world_state, node_state

    def after_reset(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
        *,
        mask : Optional[BArrayType] = None
    ) -> Tuple[
        WorldStateT,
        NodeStateT,
        Optional[ContextType],
        Optional[ObsType],
        Optional[Dict[str, Any]]
    ]:
        """
        This method is called after `WorldNode`'s has been called with `reset` (e.g. the environment reset is effectively done)
        Returns:
            world_state: The updated world state after reset.
            node_state: The updated node state after reset.
            context: The optional context of the node after reset.
            observation: The optional observation of the node after reset.
            info: The auxiliary information of the node after reset.
        """
        return world_state, node_state, None, self.get_observation(world_state, node_state), self.get_info(world_state, node_state)

    def pre_environment_step(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
        dt : Union[float, BArrayType]
    ) -> Tuple[WorldStateT, NodeStateT]:
        """
        This method is called before each environment step.
        Args:
            world_state (WorldStateT): The current state of the world.
            node_state (NodeStateT): The current state of the node.
            dt (Union[float, BArrayType]): The time delta since the last step.
        """
        return world_state, node_state

    def get_observation(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT
    ) -> ObsType:
        """
        Get the current observation from the sensor.
        If the observation space is None, this method should not be called.
        """
        raise NotImplementedError

    def get_reward(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT
    ) -> Union[float, BArrayType]:
        """
        Get the current reward from the environment.
        If the reward space is None, this method should not be called.
        """
        raise NotImplementedError
    
    def get_termination(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT
    ) -> Union[bool, BArrayType]:
        """
        Get the current termination status from the environment.
        If the termination space is None, this method should not be called.
        """
        raise NotImplementedError
    
    def get_truncation(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT
    ) -> Union[bool, BArrayType]:
        """
        Get the current truncation status from the environment.
        If the truncation space is None, this method should not be called.
        """
        raise NotImplementedError
    
    def get_info(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current info from the environment.
        """
        return None
    
    def set_next_action(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
        action : ActType
    ) -> Tuple[WorldStateT, NodeStateT]:
        """
        Update the next action to be taken by the node.
        This method should be called before `pre_environment_step` call.
        If this method is not called after a world step or an action of None is given in the call, the node will compute a dummy action to try retain the same state of the robot.
        Note that if the action space is None, this method should not be called.
        """
        raise NotImplementedError
    
    def post_environment_step(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
        dt : Union[float, BArrayType]
    ) -> Tuple[WorldStateT, NodeStateT]:
        """
        This method is called after the environment step to update the sensor's internal state.
        Args:
            world_state (WorldStateT): The current state of the world.
            node_state (NodeStateT): The current state of the node.
            dt (Union[float, BArrayType]): The time delta since the last step.
        """
        return world_state, node_state

    def close(self, world_state : WorldStateT, node_state : NodeStateT) -> WorldStateT:
        return world_state
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncWorldNode":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["FuncWorldNode"]:
        return None
    
    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the attribute `name` exists in the environment."""
        return hasattr(self, name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        """Gets the attribute `name` from the environment."""
        return getattr(self, name)
    
    def set_wrapper_attr(self, name: str, value: Any):
        """Sets the attribute `name` on the environment with `value`."""
        setattr(self, name, value)