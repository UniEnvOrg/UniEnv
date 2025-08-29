from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union
from abc import ABC, abstractmethod
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from unienv_interface.env_base.env import ContextType, ObsType, ActType

from .world import World

class WorldNode(ABC, Generic[ContextType, ObsType, ActType, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """
    Each `WorldNode` in the simulated / real world will manage some aspect of the environment.
    How the methods in this class will be called once environment resets:
    `World.reset(...)` -> `WorldNode.reset(...)` -> `WorldNode.after_reset(...)` -> `WorldNode.get_observation(...)` -> World can start stepping normally
    How the methods in this class will be called during a environment step:
    `WorldNode.set_next_action(...)` -> `WorldNode.pre_environment_step(...)` -> `World.step(...)` -> `WorldNode.post_environment_step(...)` -> `WorldNode.get_observation(...)` -> `WorldNode.get_reward(...)` -> `WorldNode.get_termination(...)` -> `WorldNode.get_truncation(...)` -> `WorldNode.get_info(...)`
    """

    name : str
    world : World[BArrayType, BDeviceType, BDtypeType, BRNGType]
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

    def pre_environment_step(self, dt : Union[float, BArrayType]) -> None:
        """
        This method is called before the environment step
        Args:
            dt (float/BArrayType): The time elapsed between the last world step and the current step (NOT the current step and next step).
        """
        pass

    def get_observation(self) -> ObsType:
        """
        Get the current observation from the sensor.
        If the observation space is None, this method should not be called.
        """
        raise NotImplementedError

    def get_reward(self) -> Union[float, BArrayType]:
        """
        Get the current reward from the environment.
        If `has_reward` is `False`, this method should not be called.
        """
        return 0

    def get_termination(self) -> Union[bool, BArrayType]:
        """
        Get the current termination signal from the environment.
        If `has_termination_signal` is `False`, this method should not be called.
        """
        return False
    
    def get_truncation(self) -> Union[bool, BArrayType]:
        """
        Get the current truncation signal from the environment.
        If `has_truncation_signal` is `False`, this method should not be called.
        """
        return False

    def get_info(self) -> Optional[Dict[str, Any]]:
        """
        Get optional auxiliary information with this node.
        """
        return None

    def set_next_action(self, action: ActType) -> None:
        """
        Update the next action to be taken by the node.
        This method should be called before `pre_environment_step` call.
        If this method is not called after a world step or an action of None is given in the call, the node will compute a dummy action to try retain the same state of the robot.
        Note that if the action space is None, this method should not be called.
        """
        raise NotImplementedError

    def post_environment_step(self, dt : Union[float, BArrayType]) -> None:
        """
        This method is called after the environment step to update the sensor's internal state.
        Args:
            dt (float/BArrayType): The time elapsed between the last world step and the current step.
        """
        pass

    def reset(
        self,
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> None:
        """
        This method is called after `World.reset(...)` has been called.
        Reset the node and update its internal state.
        """
        pass

    def after_reset(
        self,
        *,
        mask : Optional[BArrayType] = None,
    ) -> Tuple[Optional[ContextType], Optional[ObsType], Optional[Dict[str, Any]]]:
        """
        This method is called after all `WorldNode`s has been called with `reset` (e.g. the environment reset is effectively done)
        Returns:
            context: The optional context of the node after reset.
            observation: The optional observation of the node after reset.
            info: The auxiliary information of the node after reset.
        """
        self.post_environment_step(self.control_timestep)
        return None, self.get_observation(), self.get_info()

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "WorldNode":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["WorldNode"]:
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