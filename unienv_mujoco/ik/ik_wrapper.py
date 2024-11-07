from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable
from unienv_interface.env_base.funcenv import FuncEnvCommonState
from unienv_interface.world.actor import FuncActorCombinedState, FuncActor, FuncActorWrapper, ActorStateT, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType, ActorWrapperStateT
from unienv_interface.world.actor_mixins import EndEffectorFuncActorInterface
from unienv_interface.backends.base import ComputeBackend, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, Box, Dict as DictSpace
from unienv_mujoco.base.world import MujocoFuncWorld, MujocoFuncWorldState
from dataclasses import dataclass, replace as dataclass_replace
import mink
import mujoco
import numpy as np
from . import ik_util
from .ik_control import MujocoIKClass, MujocoIKStateT, MujocoIKTargetT

@dataclass
class MujocoIKWrapperState(Generic[ActorStateT, MujocoIKTargetT, MujocoIKStateT]):
    inner_actor_state : ActorStateT
    target_transform : MujocoIKTargetT
    target_action : Optional[Any]
    inner_actor_remaining_elapsed : float
    ik_state : MujocoIKStateT

class MujocoIKWrapper(Generic[
    ActorStateT, MujocoIKTargetT, MujocoIKStateT
], FuncActorWrapper[
    MujocoFuncWorldState, 
    MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT], Any, np.dtype, np.random.Generator,
    ActorStateT, Any, np.dtype, np.random.Generator
], EndEffectorFuncActorInterface[
    MujocoFuncWorldState, MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT], Any, np.random.Generator
]):
    num_eefs = 1

    def __init__(
        self,
        actor: FuncActor[
            MujocoFuncWorldState, ActorStateT, Any, np.dtype, np.random.Generator
        ],
        ik: MujocoIKClass[MujocoIKStateT, MujocoIKTargetT],
        ik_mj_model : mujoco.MjModel,
        new_action_space : Optional[Space[Any, Any, Any, np.dtype, np.random.Generator]] = None,
        fn_action_transform : Callable[[Optional[Any], np.ndarray], Any] = lambda outer_action, target_qpos: target_qpos,
        eef_workspace_translation : Optional[Box] = None, # Shape (3,)
        eef_workspace_rotation : Optional[Box] = None, # Shape (3,)
        eef_max_translation_velocity : Optional[np.ndarray] = None, # Shape (3,)
        eef_max_rotation_velocity : Optional[np.ndarray] = None # Shape (3,)
    ):
        super().__init__(actor)
        self.ik = ik
        self.ik_mj_model = ik_mj_model
        self._extra_action_space = new_action_space
        self.fn_action_transform = fn_action_transform
        self.eef_workspace_translation = eef_workspace_translation
        self.eef_workspace_rotation = eef_workspace_rotation
        self.eef_max_translation_velocity = eef_max_translation_velocity
        self.eef_max_rotation_velocity = eef_max_rotation_velocity

    @property
    def is_eef_relative(self) -> bool:
        return self.ik.is_eef_relative
    
    def _translate_to_inner_actor_state(self, state: MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]) -> ActorStateT:
        return state.inner_actor_state
    
    def _translate_to_outer_actor_state(self, state: ActorStateT, old_wrapper_state: MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]) -> MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]:
        return dataclass_replace(old_wrapper_state, inner_actor_state=state)
    
    def onboard_initial(
        self, 
        world: MujocoFuncWorld,
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        *args, 
        seed: int, 
        **kwargs
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator], 
        MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]
    ]:
        state, common_state, inner_actor_state = self.actor.onboard_initial(
            world,
            state, 
            common_state, 
            *args, 
            seed=seed, 
            **kwargs
        )
        ik_state = self.ik.initial(self.ik_mj_model, state.data.qpos[:self.ik_mj_model.nq], seed=seed)
        current_transform = self.ik.get_target_from_data(
            ik_state,
            state.mj_model,
            state.data
        )
        return state, common_state, MujocoIKWrapperState(
            inner_actor_state=inner_actor_state,
            target_transform=current_transform,
            target_action=None,
            inner_actor_remaining_elapsed=self.actor.control_timestep,
            ik_state=ik_state
        )

    def onboard_reset(
        self,
        world: MujocoFuncWorld,
        state: MujocoFuncWorldState,
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]
    ]:
        state, common_state, actor_state = super().onboard_reset(world, state, common_state, actor_state)
        actor_state = dataclass_replace(
            actor_state,
            ik_state=self.ik.update_q(
                actor_state.ik_state,
                state.data.qpos[:self.ik_mj_model.nq]
            )
        )
        current_transform = self.ik.get_target_from_data(
            actor_state.ik_state,
            state.mj_model,
            state.data
        )
        return state, common_state, dataclass_replace(
            actor_state,
            target_transform=current_transform,
            inner_actor_remaining_elapsed=self.actor.control_timestep
        )
    
    def onboard_step(
        self,
        state: MujocoFuncWorldState,
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT],
        last_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]
    ]:
        state, common_state, actor_state = super().onboard_step(state, common_state, actor_state, last_step_elapsed)
        actor_state.inner_actor_remaining_elapsed -= last_step_elapsed
        ik_state, target_qpos, ik_converged = self.ik.step(
            state.mj_model,
            state.data.qpos[:self.ik_mj_model.nq],
            actor_state.ik_state,
            actor_state.target_transform,
            last_step_elapsed
        )
        actor_state.ik_state = ik_state
        if actor_state.inner_actor_remaining_elapsed > 0:
            state, common_state, inner_actor_state = self.actor.set_next_action(
                state,
                common_state,
                actor_state.inner_actor_state,
                action=self.fn_action_transform(
                    actor_state.target_action,
                    target_qpos
                ),
                last_control_step_elapsed=self.actor.control_timestep - actor_state.inner_actor_remaining_elapsed
            )
            actor_state.inner_actor_state = inner_actor_state
            actor_state.inner_actor_remaining_elapsed = self.actor.control_timestep
        return state, common_state, actor_state

    def set_next_extra_action(
        self,
        state: MujocoFuncWorldState,
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT],
        action: Any,
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]
    ]:
        return state, common_state, dataclass_replace(
            actor_state,
            target_action=action
        )
    
    def onboard_close(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        actor_state: MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator]
    ]:
        self.ik.close(actor_state.ik_state)
        return super().onboard_close(
            state, 
            common_state, 
            actor_state
        )

    def get_current_eef_position(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]
    ) -> np.ndarray:
        transform = self.ik.get_target_from_data(
            actor_state.ik_state,
            state.mj_model,
            state.data
        )

        return self.ik.translation_from_target(transform)

    def get_current_eef_euler(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        actor_state: MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]
    ) -> np.ndarray:
        transform = self.ik.get_target_from_data(
            actor_state.ik_state,
            state.mj_model,
            state.data
        )
        quat = self.ik.quaternion_from_target(transform)
        rotation = mink.SO3(quat).as_rpy_radians()
        return np.array([rotation.roll, rotation.pitch, rotation.yaw])
    
    def set_target_eef(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        actor_state: MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT], 
        position: np.ndarray, 
        euler: np.ndarray
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator], 
        MujocoIKWrapperState[ActorStateT, MujocoIKTargetT, MujocoIKStateT]
    ]:
        return state, common_state, dataclass_replace(
            actor_state,
            target_transform=mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3.from_rpy_radians(*euler),
                translation=position
            )
        )