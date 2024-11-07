from abc import abstractmethod
from typing import Tuple, Type, Union, Any, Generic, TypeVar, Optional, Dict

from unienv_interface.space import Dict as DictSpace, Box, Space
from ..actor import StateType, ActorStateT, FuncEnvCommonState, ActorMixin
from unienv_interface.space import batch_utils as space_batch_utils
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BRNGType
import numpy as np

def euler_to_quaternion(backend : ComputeBackend, rotation: BArrayType) -> BArrayType:
    roll = rotation[..., 0]
    pitch = rotation[..., 1]
    yaw = rotation[..., 2]

    array_workspace = backend.array_api_namespace

    qx = array_workspace.sin(roll/2) * array_workspace.cos(pitch/2) * array_workspace.cos(yaw/2) - array_workspace.cos(roll/2) * array_workspace.sin(pitch/2) * array_workspace.sin(yaw/2)
    qy = array_workspace.cos(roll/2) * array_workspace.sin(pitch/2) * array_workspace.cos(yaw/2) + array_workspace.sin(roll/2) * array_workspace.cos(pitch/2) * array_workspace.sin(yaw/2)
    qz = array_workspace.cos(roll/2) * array_workspace.cos(pitch/2) * array_workspace.sin(yaw/2) - array_workspace.sin(roll/2) * array_workspace.sin(pitch/2) * array_workspace.cos(yaw/2)
    qw = array_workspace.cos(roll/2) * array_workspace.cos(pitch/2) * array_workspace.cos(yaw/2) + array_workspace.sin(roll/2) * array_workspace.sin(pitch/2) * array_workspace.sin(yaw/2)

    return array_workspace.stack([qx, qy, qz, qw], axis=-1)

def normalize_euler(backend : ComputeBackend, euler: BArrayType) -> BArrayType:
    return (euler + 2*np.pi + np.pi) % (2 * np.pi) - np.pi

class EndEffectorActorInterface():
    """
    An end effector actor is an actor that can control the end effector of the robot.
    """
    num_eefs : int
    is_eef_relative : bool # Whether the end-effector SE(3) is relative to a moving part of the robot or fixed in the world frame
    eef_workspace_translation : Optional[Box] = None # Shape (3,)
    eef_workspace_rotation : Optional[Box] = None # Shape (3,)

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
    def get_current_eef_euler(
        self,
    ) -> BArrayType:
        """
        Get the current euler angles of the end effector, with shape (num_eefs, 3) if num_eefs > 1 else (3,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 3) if num_eefs > 1 else (B, 3)
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_target_eef(
        self,
        position: BArrayType,
        euler: BArrayType
    ) -> None:
        """
        Set the position and quaternion of the end effector in the action, with shape (num_eefs, 3) and (num_eefs, 3) if num_eefs > 1 else (3,) and (3,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 3) and (B, num_eefs, 3) if num_eefs > 1 else (B, 3) and (B, 3)
        """
        raise NotImplementedError

class EndEffectorFuncActorInterface(
    Generic[StateType, ActorStateT, BDeviceType, BRNGType]
):
    num_eefs : int
    is_eef_relative : bool # Whether the end-effector SE(3) is relative to a moving part of the robot or fixed in the world frame
    eef_workspace_translation : Optional[Box] = None # Shape (3,) or (num_eefs, 3)
    eef_workspace_rotation : Optional[Box] = None # Shape (3,) or (num_eefs, 3)
    eef_max_translation_velocity : Optional[BArrayType] = None # Shape (3,) or (num_eefs, 3)
    eef_max_rotation_velocity : Optional[BArrayType] = None # Shape (3,) or (num_eefs, 3)

    @abstractmethod
    def get_current_eef_position(
        self,
        state : StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT
    ) -> BArrayType:
        """
        Get the current position of the end effector, with shape (num_eefs, 3) if num_eefs > 1 else (3,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 3) if num_eefs > 1 else (B, 3)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_current_eef_euler(
        self,
        state : StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT
    ) -> BArrayType:
        """
        Get the current euler angles of the end effector, with shape (num_eefs, 3) if num_eefs > 1 else (3,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 3) if num_eefs > 1 else (B, 3)
        """
        raise NotImplementedError

    @abstractmethod
    def set_target_eef(
        self,
        state : StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT,
        position: BArrayType,
        euler: BArrayType
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        """
        Set the position and quaternion of the end effector in the action, with shape (num_eefs, 3) and (num_eefs, 3) if num_eefs > 1 else (3,) and (3,)
        If the actor is a batched actor, the shape would be (B, num_eefs, 3) and (B, num_eefs, 3) if num_eefs > 1 else (B, 3) and (B, 3)
        """
        raise NotImplementedError

class EndEffectorActorMixin(ActorMixin[EndEffectorActorInterface, EndEffectorFuncActorInterface]):
    mixin_name : str = "eef"
    mixin_actor_type = EndEffectorActorInterface
    mixin_func_actor_type = EndEffectorFuncActorInterface

    @classmethod
    def get_mixin_observation_space(
        cls,
        instance: Union[EndEffectorActorInterface, EndEffectorFuncActorInterface],
        backend: ComputeBackend,
        device: Optional[BDeviceType]
    ) -> DictSpace:
        batch_space = (instance.num_eefs, ) if instance.num_eefs > 1 else ()

        position_batched_space = Box(
            backend=backend,
            low=-backend.array_api_namespace.inf,
            high=backend.array_api_namespace.inf,
            dtype=backend.default_floating_dtype,
            shape=batch_space + (3, )
        )
        euler_batched_space = Box(
            backend=backend,
            low=-np.pi,
            high=np.pi,
            dtype=backend.default_floating_dtype,
            shape=batch_space + (3, )
        )
        return DictSpace(
            backend,
            spaces={
                "position": position_batched_space,
                "euler": euler_batched_space
            },
            device=device
        )
    
    @classmethod
    def get_mixin_action_space(
        cls, 
        instance: Union[EndEffectorActorInterface, EndEffectorFuncActorInterface], 
        backend: ComputeBackend, 
        device: BDeviceType
    ) -> Space:
        position_batched_space = Box(
            backend,
            low=-backend.array_api_namespace.inf,
            high=backend.array_api_namespace.inf,
            dtype=backend.default_floating_dtype,
            shape=(instance.num_eefs, 3) if instance.num_eefs > 1 else (3, ),
            device=device
        ) if instance.eef_workspace_translation is None else instance.eef_workspace_translation
        
        euler_batched_space = Box(
            backend,
            low=-np.pi,
            high=np.pi,
            dtype=backend.default_floating_dtype,
            shape=(instance.num_eefs, 3) if instance.num_eefs > 1 else (3, ),
            device=device
        ) if instance.eef_workspace_rotation is None else instance.eef_workspace_rotation
        
        return DictSpace(
            backend,
            spaces={
                "position": position_batched_space,
                "euler": euler_batched_space
            },
            device=device
        )
    
    @classmethod
    def read_mixin_data(
        cls, 
        instance: EndEffectorActorInterface
    ) -> Dict[str, Any]:
        return {
            "position": instance.get_current_eef_position(),
            "euler": instance.get_current_eef_euler()
        }
    
    @classmethod
    def read_mixin_data_func(
        cls, 
        instance: EndEffectorFuncActorInterface,
        state: StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT,
        Dict[str, Any]
    ]:
        return state, common_state, actor_state, {
            "position": instance.get_current_eef_position(state, common_state, actor_state),
            "euler": instance.get_current_eef_euler(state, common_state, actor_state)
        }
    
    @classmethod
    def clip_target_eef(
        cls,
        instance: Union[EndEffectorActorInterface, EndEffectorFuncActorInterface],
        target_position: BArrayType,
        target_rotation: BArrayType
    ) -> Tuple[BArrayType, BArrayType]:
        if instance.eef_workspace_translation is not None:
            target_position = instance.eef_workspace_translation.clip(target_position)
        target_rotation = normalize_euler(instance.backend, target_rotation)
        if instance.eef_workspace_rotation is not None:
            target_rotation = instance.eef_workspace_rotation.clip(target_rotation)
        return target_position, target_rotation

    @classmethod
    def apply_mixin_action(
        cls,
        instance: EndEffectorActorInterface,
        action: Dict[str, BArrayType]
    ) -> None:
        target_position, target_rotation = cls.clip_target_eef(instance, action['position'], action['euler'])
        instance.set_target_eef(target_position, target_rotation)

    @classmethod
    def apply_mixin_action_func(
        cls,
        instance: EndEffectorFuncActorInterface,
        state: StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT,
        action: Dict[str, BArrayType],
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        target_position, target_rotation = cls.clip_target_eef(instance, action['position'], action['euler'])
        return instance.set_target_eef(
            state, 
            common_state, 
            actor_state, 
            target_position, 
            target_rotation
        )

class RelativeEndEffectorActorMixin(EndEffectorActorMixin):
    mixin_name : str = "relative_eef"
    mixin_actor_type = EndEffectorActorInterface
    mixin_func_actor_type = EndEffectorFuncActorInterface

    DEFAULT_EEF_MAX_TRANSLATION_VELOCITY = 0.2 # m/s
    DEFAULT_EEF_MAX_ROTATION_VELOCITY = np.deg2rad(60) # rad/s

    @classmethod
    def get_mixin_action_space(
        cls, 
        instance: Union[EndEffectorActorInterface, EndEffectorFuncActorInterface], 
        backend: ComputeBackend, 
        device: BDeviceType
    ) -> Space:
        position_batched_space = Box(
            backend,
            low=-__class__.DEFAULT_EEF_MAX_TRANSLATION_VELOCITY,
            high=__class__.DEFAULT_EEF_MAX_TRANSLATION_VELOCITY,
            dtype=backend.default_floating_dtype,
            shape=(instance.num_eefs, 3) if instance.num_eefs > 1 else (3, ),
            device=device
        ) if instance.eef_max_translation_velocity is None else instance.eef_max_translation_velocity
        
        euler_batched_space = Box(
            backend,
            low=-__class__.DEFAULT_EEF_MAX_ROTATION_VELOCITY,
            high=__class__.DEFAULT_EEF_MAX_ROTATION_VELOCITY,
            dtype=backend.default_floating_dtype,
            shape=(instance.num_eefs, 3) if instance.num_eefs > 1 else (3, ),
            device=device
        ) if instance.eef_max_rotation_velocity is None else instance.eef_max_rotation_velocity
        
        return DictSpace(
            backend,
            spaces={
                "position": position_batched_space,
                "euler": euler_batched_space
            },
            device=device
        )
    
    @classmethod
    def map_target_abs_eef(
        cls,
        instance: Union[EndEffectorActorInterface, EndEffectorFuncActorInterface],
        current_translation: BArrayType,
        current_rotation: BArrayType,
        action: Dict[str, BArrayType],
    ) -> Dict[str, BArrayType]:
        action_space = cls.get_mixin_action_space(instance, instance.backend, instance.device)
        delta_translation = action_space.spaces['position'].clip(action['position'])
        delta_rotation = action_space.spaces['euler'].clip(action['euler'])
        target_translation = current_translation + delta_translation
        target_rotation = current_rotation + delta_rotation
        return {
            "position": target_translation,
            "euler": target_rotation
        }

    @classmethod
    def map_target_relative_eef(
        cls,
        instance: Union[EndEffectorActorInterface, EndEffectorFuncActorInterface],
        current_translation: BArrayType,
        current_rotation: BArrayType,
        action: Dict[str, BArrayType],
    ) -> Dict[str, BArrayType]:
        action_space = cls.get_mixin_action_space(instance, instance.backend, instance.device)
        delta_translation = action['position'] - current_translation
        delta_rotation = action['euler'] - current_rotation
        delta_rotation = normalize_euler(instance.backend, delta_rotation)
        delta_translation = action_space.spaces['position'].clip(delta_translation)
        delta_rotation = action_space.spaces['euler'].clip(delta_rotation)

        return {
            "position": delta_translation,
            "euler": delta_rotation
        }

    @classmethod
    def apply_mixin_action(
        self,
        instance: EndEffectorActorInterface,
        action: Dict[str, BArrayType]
    ) -> None:
        target_abs_action = self.map_target_abs_eef(
            instance, 
            instance.get_current_eef_position(), 
            instance.get_current_eef_euler(), 
            action
        )
        EndEffectorActorMixin.apply_mixin_action(
            instance,
            target_abs_action
        )

    @classmethod
    def apply_mixin_action_func(
        self,
        instance: EndEffectorFuncActorInterface,
        state: StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT,
        action: Dict[str, BArrayType],
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        target_abs_action = self.map_target_abs_eef(
            instance, 
            instance.get_current_eef_position(state, common_state, actor_state), 
            instance.get_current_eef_euler(state, common_state, actor_state), 
            action
        )
        return EndEffectorActorMixin.apply_mixin_action_func(
            instance,
            state,
            common_state,
            actor_state,
            target_abs_action,
            last_control_step_elapsed
        )


class GripperActorInterface:
    """
    An gripper actor is an actor that can control the gripper of the robot.
    """
    num_grippers : int
    is_gripper_relative : bool # Whether the gripper position is relative to a moving part of the robot or fixed in the world frame

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
    def set_target_gripper(
        self,
        position: BArrayType
    ) -> None:
        """
        Set the position of the gripper in the action, with shape (num_grippers, 1) if num_grippers > 1 else (1,)
        If the actor is a batched actor, the shape would be (B, num_grippers, 1) if num_grippers > 1 else (B, 1)
        """
        raise NotImplementedError

class GripperFuncActorInterface(
    Generic[StateType, ActorStateT, BDeviceType, BRNGType]
):
    num_grippers : int
    is_gripper_relative : bool # Whether the gripper position is relative to a moving part of the robot or fixed in the world frame

    @abstractmethod
    def get_current_gripper_position(
        self,
        state : StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT
    ) -> BArrayType:
        """
        Get the current position of the gripper, with shape (num_grippers, 1) if num_grippers > 1 else (1,)
        If the actor is a batched actor, the shape would be (B, num_grippers, 1) if num_grippers > 1 else (B, 1)
        """
        raise NotImplementedError

    @abstractmethod
    def set_target_gripper(
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
        Set the position of the gripper in the action, with shape (num_grippers, 1) if num_grippers > 1 else (1,)
        If the actor is a batched actor, the shape would be (B, num_grippers, 1) if num_grippers > 1 else (B, 1)
        """
        raise NotImplementedError
    
class GripperActorMixin(ActorMixin[GripperActorInterface, GripperFuncActorInterface]):
    mixin_name : str = "gripper"
    mixin_actor_type = GripperActorInterface
    mixin_func_actor_type = GripperFuncActorInterface

    @classmethod
    def get_mixin_observation_space(
        cls,
        instance: Union[GripperActorInterface, GripperFuncActorInterface],
        backend: ComputeBackend,
        device: Optional[BDeviceType]
    ) -> DictSpace:
        batch_space = (instance.num_grippers, ) if instance.num_grippers > 1 else ()
        return Box(
            backend=backend,
            low=0.0,
            high=1.0,
            shape=batch_space + (1, ),
            dtype=backend.default_floating_dtype,
            device=device
        )

    @classmethod
    def get_mixin_action_space(
        cls, 
        instance: Union[GripperActorInterface, GripperFuncActorInterface], 
        backend: ComputeBackend, 
        device: BDeviceType
    ) -> Space:
        batch_space = (instance.num_grippers, ) if instance.num_grippers > 1 else ()
        return Box(
            backend,
            low=0.0,
            high=1.0,
            shape=batch_space + (1, ),
            dtype=backend.default_floating_dtype,
            device=device
        )
    
    @classmethod
    def read_mixin_data(
        cls, 
        instance: GripperActorInterface
    ) -> BArrayType:
        return instance.get_current_gripper_position()
    
    @classmethod
    def read_mixin_data_func(
        cls, 
        instance: GripperFuncActorInterface,
        state: StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT,
        BArrayType
    ]:
        return state, common_state, actor_state, instance.get_current_gripper_position(state, common_state, actor_state)
    
    @classmethod
    def apply_mixin_action(
        cls,
        instance: GripperActorInterface,
        action: BArrayType
    ) -> None:
        instance.set_target_gripper(action)

    @classmethod
    def apply_mixin_action_func(
        cls,
        instance: GripperFuncActorInterface,
        state: StateType,
        common_state: FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state: ActorStateT,
        action: BArrayType,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        return instance.set_target_gripper(state, common_state, actor_state, action)