from abc import abstractmethod
from typing import Tuple, Union, Any, Generic, TypeVar, Optional
from ..actor import StateType, ActorStateT, FuncEnvCommonState, ActorActT
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BRNGType

class EndEffectorActorMixin(Generic[ActorActT]):
    """
    An end effector actor is an actor that can control the end effector of the robot.
    """
    num_eefs : int
    is_eef_relative : bool
    @abstractmethod
    def get_current_eef_position(
        self,
    ) -> BArrayType:
        """
        Get the current position of the end effector, with shape (num_eefs, 3) if num_eefs > 1 else (3,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 3) if num_eefs > 1 else (B, 3)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_current_eef_quaternion(
        self,
    ) -> BArrayType:
        """
        Get the current quaternion of the end effector, with shape (num_eefs, 4) if num_eefs > 1 else (4,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 4) if num_eefs > 1 else (B, 4)
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_eef(
        self,
        position: Optional[BArrayType],
        quaternion: Optional[BArrayType]
    ) -> None:
        """
        Set the position and quaternion of the end effector, with shape (num_eefs, 3) and (num_eefs, 4) if num_eefs > 1 else (3,) and (4,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 3) and (B, num_eefs, 4) if num_eefs > 1 else (B, 3) and (B, 4)
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_target_eef_in_action(
        self,
        action: ActorActT,
        position: Optional[BArrayType],
        quaternion: Optional[BArrayType]
    ) -> ActorActT:
        """
        Set the position and quaternion of the end effector in the action, with shape (num_eefs, 3) and (num_eefs, 4) if num_eefs > 1 else (3,) and (4,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 3) and (B, num_eefs, 4) if num_eefs > 1 else (B, 3) and (B, 4)
        """
        raise NotImplementedError

class GripperActorMixin(Generic[ActorActT]):
    """
    An gripper actor is an actor that can control the gripper of the robot.
    """
    num_grippers : int

    @abstractmethod
    def get_current_gripper_position(
        self,
    ) -> BArrayType:
        """
        Get the current position of the gripper, with shape (num_grippers, 1) if num_grippers > 1 else (1,)
        If the actor is a batched actor, the shape would be (B, num_grippers, 1) if num_grippers > 1 else (B, 1)
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_gripper(
        self,
        position: BArrayType
    ) -> None:
        """
        Set the position of the gripper, with shape (num_grippers, 1) if num_grippers > 1 else (1,)
        If the actor is a batched actor, the shape would be (B, num_grippers, 1) if num_grippers > 1 else (B, 1)
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_target_gripper_in_action(
        self,
        action: ActorActT,
        position: BArrayType
    ) -> ActorActT:
        """
        Set the position of the gripper in the action, with shape (num_grippers, 1) if num_grippers > 1 else (1,)
        If the actor is a batched actor, the shape would be (B, num_grippers, 1) if num_grippers > 1 else (B, 1)
        """
        raise NotImplementedError

class EndEffectorFuncActorMixin(
    Generic[StateType, ActorStateT, ActorActT, BDeviceType, BRNGType]
):
    num_eefs : int
    is_eef_relative : bool
    @abstractmethod
    def get_current_eef_position(
        self,
        state : StateType,
        actor_state: ActorStateT
    ) -> BArrayType:
        """
        Get the current position of the end effector, with shape (num_eefs, 3) if num_eefs > 1 else (3,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 3) if num_eefs > 1 else (B, 3)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_current_eef_quaternion(
        self,
        state : StateType,
        actor_state: ActorStateT
    ) -> BArrayType:
        """
        Get the current quaternion of the end effector, with shape (num_eefs, 4) if num_eefs > 1 else (4,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 4) if num_eefs > 1 else (B, 4)
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
        Set the position and quaternion of the end effector, with shape (num_eefs, 3) and (num_eefs, 4) if num_eefs > 1 else (3,) and (4,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 3) and (B, num_eefs, 4) if num_eefs > 1 else (B, 3) and (B, 4)
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_target_eef_in_action(
        self,
        state : StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT,
        action: ActorActT,
        position: Optional[BArrayType],
        quaternion: Optional[BArrayType]
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT,
        ActorActT,
    ]:
        """
        Set the position and quaternion of the end effector in the action, with shape (num_eefs, 3) and (num_eefs, 4) if num_eefs > 1 else (3,) and (4,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 3) and (B, num_eefs, 4) if num_eefs > 1 else (B, 3) and (B, 4)
        """
        raise NotImplementedError
    
class GripperFuncActorMixin(
    Generic[StateType, ActorStateT, ActorActT, BDeviceType, BRNGType]
):
    num_grippers : int
    @abstractmethod
    def get_current_gripper_position(
        self,
        state : StateType,
        actor_state: ActorStateT
    ) -> BArrayType:
        """
        Get the current position of the gripper, with shape (num_grippers, 1) if num_grippers > 1 else (1,)
        If the actor is a batched actor, the shape would be (B, num_grippers, 1) if num_grippers > 1 else (B, 1)
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_gripper(
        self,
        state : StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT,
        position: BArrayType
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        """
        Set the position of the gripper, with shape (num_grippers, 1) if num_grippers > 1 else (1,)
        If the actor is a batched actor, the shape would be (B, num_grippers, 1) if num_grippers > 1 else (B, 1)
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_target_gripper_in_action(
        self,
        state : StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT,
        action: ActorActT,
        position: BArrayType
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT,
        ActorActT,
    ]:
        """
        Set the position of the gripper in the action, with shape (num_grippers, 1) if num_grippers > 1 else (1,)
        If the actor is a batched actor, the shape would be (B, num_grippers, 1) if num_grippers > 1 else (B, 1)
        """
        raise NotImplementedError