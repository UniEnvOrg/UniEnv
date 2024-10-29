from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable
from unienv_interface.env_base.funcenv import FuncEnvCommonState
from unienv_interface.world.actor import FuncActorCombinedState, FuncActor, FuncActorWrapper, ActorStateT, ActorActT, ActorWrapperActT, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType, ActorWrapperStateT
from unienv_interface.world.actor_mixins import EndEffectorFuncActorMixin
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
class MujocoIKWrapperState(Generic[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]):
    inner_actor_state : ActorStateT
    target_transform : MujocoIKTargetT
    target_action : Optional[ActorWrapperActT]
    inner_actor_remaining_elapsed : float
    ik_state : MujocoIKStateT

class MujocoIKWrapper(Generic[
    ActorStateT, ActorWrapperActT, ActorActT, MujocoIKTargetT, MujocoIKStateT
], FuncActorWrapper[
    MujocoFuncWorldState, MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT], ActorWrapperActT, Any, np.dtype, np.random.Generator,
    ActorStateT, ActorActT, Any, np.dtype, np.random.Generator
], EndEffectorFuncActorMixin[
    MujocoFuncWorldState, MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT], Any, np.random.Generator
]):
    def __init__(
        self,
        actor: FuncActor[
            MujocoFuncWorldState, ActorStateT, ActorActT, Any, np.dtype, np.random.Generator
        ],
        ik: MujocoIKClass[MujocoIKStateT, MujocoIKTargetT],
        mj_model: mujoco.MjModel,
        new_action_space : Space[ActorWrapperActT, Any, Any, np.dtype, np.random.Generator],
        fn_target_transform : Callable[[ActorWrapperActT], MujocoIKTargetT],
        fn_action_transform : Callable[[Optional[ActorWrapperActT], np.ndarray], ActorActT],
    ):
        super().__init__(actor)
        self.ik = ik
        self.mj_model = mj_model
        self._action_space = new_action_space
        self.fn_target_transform = fn_target_transform
        self.fn_action_transform = fn_action_transform
    
    @property
    def is_eef_relative(self) -> bool:
        return self.ik.is_eef_relative

    @property
    def action_space(self) -> Space[ActorWrapperActT, Any, Any, np.dtype, np.random.Generator]:
        return self._action_space
    
    def _translate_to_inner_actor_state(self, state: MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]) -> ActorStateT:
        return state.inner_actor_state
    
    def _translate_to_outer_actor_state(self, state: ActorStateT, old_wrapper_state: MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]) -> MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]:
        return dataclass_replace(old_wrapper_state, inner_actor_state=state)
    
    def onboard_initial(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        *args, 
        seed: int, 
        **kwargs
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator], 
        MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]
    ]:
        state, common_state, inner_actor_state = self.actor.onboard_initial(
            state, 
            common_state, 
            *args, 
            seed=seed, 
            **kwargs
        )
        ik_state = self.ik.initial(self.mj_model, state.data.qpos[:self.mj_model.nq], seed=seed)
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
        state: MujocoFuncWorldState,
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]
    ]:
        state, common_state, actor_state = super().onboard_reset(state, common_state, actor_state)
        actor_state = dataclass_replace(
            actor_state,
            ik_state=self.ik.update_q(
                actor_state.ik_state,
                state.data.qpos[:self.mj_model.nq]
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
        actor_state: MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT],
        last_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]
    ]:
        state, common_state, actor_state = super().onboard_step(state, common_state, actor_state, last_step_elapsed)
        actor_state.inner_actor_remaining_elapsed -= last_step_elapsed
        ik_state, target_qpos, ik_converged = self.ik.step(
            state.mj_model,
            state.data.qpos[:self.mj_model.nq],
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

    def set_next_action(
        self,
        state: MujocoFuncWorldState,
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT],
        action: Optional[ActorWrapperActT],
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]
    ]:
        target_transform = self.fn_target_transform(action)
        return state, common_state, dataclass_replace(
            actor_state,
            target_transform=target_transform,
            target_action=action
        )
    
    def onboard_close(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        actor_state: MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]
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
        actor_state: MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]
    ) -> np.ndarray:
        trasform = self.ik.get_target_from_data(
            actor_state.ik_state,
            state.mj_model,
            state.data
        )

        return self.ik.translation_from_target(trasform)

    def get_current_eef_quaternion(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        actor_state: MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]
    ) -> np.ndarray:
        trasform = self.ik.get_target_from_data(
            actor_state.ik_state,
            state.mj_model,
            state.data
        )

        return self.ik.quaternion_from_target(trasform)
    
    def set_eef(
        self,
        state: MujocoFuncWorldState,
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT],
        position: Optional[np.ndarray],
        quaternion: Optional[np.ndarray]
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MujocoIKWrapperState[ActorStateT, ActorWrapperActT, MujocoIKTargetT, MujocoIKStateT]
    ]:
        assert not (position is None and quaternion is None)
        current_transform = self.ik.get_target_from_data(
            actor_state.ik_state,
            state.mj_model,
            state.data
        ) if position is None or quaternion is None else None
        target_transform = self.ik.target_from_se3(
            translation=self.ik.translation_from_target(current_transform) if position is None else position,
            quaternion=self.ik.quaternion_from_target(current_transform) if quaternion is None else quaternion
        )
        return state, common_state, dataclass_replace(
            actor_state,
            target_transform=target_transform
        )