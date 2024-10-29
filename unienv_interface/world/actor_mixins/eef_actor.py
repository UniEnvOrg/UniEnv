from abc import abstractmethod
from typing import Tuple, Union, Any, Generic, TypeVar, Optional
from ..actor import StateType, ActorStateT, FuncEnvCommonState
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BRNGType

class EndEffectorActorMixin:
    is_eef_relative : bool
    @abstractmethod
    def get_current_eef_position(
        self,
    ) -> BArrayType:
        """
        Get the current position of the end effector, with shape (3,)
        If the actor is a batched actor, the shape would be (B, 3)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_current_eef_quaternion(
        self,
    ) -> BArrayType:
        """
        Get the current quaternion of the end effector, with shape (4,)
        If the actor is a batched actor, the shape would be (B, 4)
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_eef(
        self,
        position: Optional[BArrayType],
        quaternion: Optional[BArrayType]
    ) -> None:
        """
        Set the position and quaternion of the end effector, with shape (3,) and (4,)
        If the actor is a batched actor, the shape would be (B, 3) and (B, 4)
        """
        raise NotImplementedError

class EndEffectorFuncActorMixin(
    Generic[StateType, ActorStateT, BDeviceType, BRNGType]
):
    is_eef_relative : bool
    @abstractmethod
    def get_current_eef_position(
        self,
        state : StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT
    ) -> BArrayType:
        """
        Get the current position of the end effector, with shape (3,)
        If the actor is a batched actor, the shape would be (B, 3)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_current_eef_quaternion(
        self,
        state : StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT
    ) -> BArrayType:
        """
        Get the current quaternion of the end effector, with shape (4,)
        If the actor is a batched actor, the shape would be (B, 4)
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_eef(
        self,
        state : StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT,
        position: Optional[BArrayType],
        quaternion: Optional[BArrayType]
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        """
        Set the position and quaternion of the end effector, with shape (3,) and (4,)
        If the actor is a batched actor, the shape would be (B, 3) and (B, 4)
        """
        raise NotImplementedError
    