from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable
from UniEnvPy.unienv_interface.env_base.funcenv import FuncEnvCommonState
from UniEnvPy.unienv_interface.world.actor import FuncActorCombinedState
from unienv_interface.world.actor import FuncActor, FuncActorWrapper, ActorStateT, ActorActT, ActorWrapperActT, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType, ActorWrapperStateT
from unienv_interface.backends.base import ComputeBackend, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, Box, Dict as DictSpace
from unienv_mujoco.base.world import MujocoFuncWorld, MujocoFuncWorldState
from dataclasses import dataclass, replace as dataclass_replace
import mink
import mujoco
import numpy as np
from . import ik_util

@dataclass
class MinkIKState:
    configuration : mink.Configuration
    limits : List[mink.Limit]
    eef_task : mink.FrameTask

class MinkIK:
    def __init__(
        self,
        collision_avoid_geom_pairs : Optional[List[Tuple[List[Union[str,int]], List[Union[str,int]]]]],
        max_velocity_per_joint : Optional[Dict[str, float]],
        frame_name : str,
        frame_type : str,

        pos_threshold : float = 1e-4,
        ori_threshold : float = 1e-4,
        solver : str = "quadprog",
        max_iterations : int = 20,
    ):
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.collision_avoid_geom_pairs = collision_avoid_geom_pairs
        self.max_velocity_per_joint = max_velocity_per_joint
        
        self.pos_threshold = pos_threshold
        self.ori_threshold = ori_threshold
        self.solver = solver
        self.max_iterations = max_iterations

    def initial(
        self,
        world_state : MujocoFuncWorldState
    ) -> MinkIKState:
        limits = [
            mink.ConfigurationLimit(
                world_state.mj_model
            ),
        ]
        eef_task = mink.FrameTask(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0
        )
        if self.collision_avoid_geom_pairs is not None:
            limits.append(mink.CollisionAvoidanceLimit(
                world_state.mj_model,
                geom_pairs=self.collision_avoid_geom_pairs
            ))
        if self.max_velocity_per_joint is not None:
            limits.append(mink.VelocityLimit(
                world_state.mj_model,
                max_velocity_per_joint=self.max_velocity_per_joint
            ))

        return MinkIKState(
            configuration=mink.Configuration(
                model=world_state.mj_model, 
                q=world_state.data.qpos
            ),
            limits=limits,
            eef_task=eef_task
        )

    def step(
        self,
        world_state : MujocoFuncWorldState,
        ik_state : MinkIKState,
        target_position : np.ndarray,
        target_orientation_euler : np.ndarray,
        elapsed_seconds : float
    ) -> Tuple[
        MujocoFuncWorldState,
        MinkIKState,
        np.ndarray
    ]:
        ik_state.eef_task.set_target(mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3.from_rpy_radians(*target_orientation_euler),
            translation=target_position
        ))
        for i in range(self.max_iterations):
            vel = mink.solve_ik(
                configuration=ik_state.configuration,
                tasks=[ik_state.eef_task],
                dt=elapsed_seconds,
                solver=self.solver,
                damping=1e-3,
                limits=ik_state.limits
            )
            ik_state.configuration.integrate_inplace(vel, elapsed_seconds)
            err = ik_state.eef_task.compute_error(ik_state.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= self.pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= self.ori_threshold
            if pos_achieved and ori_achieved:
                break
        
        return world_state, ik_state, ik_state.configuration.q


@dataclass(frozen=True)
class MinkIKWrapperState(Generic[ActorStateT, ActorWrapperActT]):
    inner_actor_state : ActorStateT
    target_position : np.ndarray
    target_orientation_euler : np.ndarray
    target_action : Optional[ActorWrapperActT]
    inner_actor_remaining_elapsed : float
    ik_state : MinkIKState

class MinkIKWrapper(FuncActorWrapper[
    MujocoFuncWorldState, MinkIKWrapperState[ActorStateT, ActorWrapperActT], ActorWrapperActT, Any, np.dtype, np.random.Generator,
    ActorStateT, ActorActT, Any, np.dtype, np.random.Generator
],Generic[
    ActorStateT, ActorWrapperActT, ActorActT
]):
    def __init__(
        self,
        actor: FuncActor[
            MujocoFuncWorldState, ActorStateT, ActorActT, Any, np.dtype, np.random.Generator
        ],
        ik: MinkIK,
        new_action_space : Space[ActorWrapperActT, Any, Any, np.dtype, np.random.Generator],
        fn_target_position_and_rotation : Callable[[ActorWrapperActT], Tuple[np.ndarray, np.ndarray]],
        fn_action_transform : Callable[[Optional[ActorWrapperActT], np.ndarray], ActorActT],
    ):
        super().__init__(actor)
        self.ik = ik
        self._action_space = new_action_space
        self.fn_target_position_and_rotation = fn_target_position_and_rotation
        self.fn_action_transform = fn_action_transform
    
    @property
    def action_space(self) -> Space[ActorWrapperActT, Any, Any, np.dtype, np.random.Generator]:
        return self._action_space
    
    def _translate_to_inner_actor_state(self, state: MinkIKWrapperState[ActorStateT, ActorWrapperActT]) -> ActorStateT:
        return state.inner_actor_state
    
    def _translate_to_outer_actor_state(self, state: ActorStateT, old_wrapper_state: MinkIKWrapperState[ActorStateT, ActorWrapperActT]) -> MinkIKWrapperState[ActorStateT, ActorWrapperActT]:
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
        MinkIKWrapperState[ActorStateT, ActorWrapperActT]
    ]:
        state, common_state, inner_actor_state = self.actor.onboard_initial(
            state, 
            common_state, 
            *args, 
            seed=seed, 
            **kwargs
        )
        ik_state = self.ik.initial(state)
        current_transform = ik_util.get_transform_frame_to_world(
            state.mj_model,
            state.data,
            self.ik.frame_name,
            self.ik.frame_type
        )
        current_position, current_orientation = current_transform.translation(), current_transform.rotation().as_rpy_radians()
        ik_state.eef_task.set_target(
            current_transform
        )
        return state, common_state, MinkIKWrapperState(
            inner_actor_state=inner_actor_state,
            target_position=current_position,
            target_orientation_euler=current_orientation,
            target_action=None,
            inner_actor_remaining_elapsed=self.actor.control_timestep,
            ik_state=ik_state
        )

    def onboard_reset(
        self,
        state: MujocoFuncWorldState,
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: MinkIKWrapperState[ActorStateT, ActorWrapperActT]
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MinkIKWrapperState[ActorStateT, ActorWrapperActT]
    ]:
        state, common_state, actor_state = super().onboard_reset(state, common_state, actor_state)
        current_transform = ik_util.get_transform_frame_to_world(
            state.mj_model,
            state.data,
            self.ik.frame_name,
            self.ik.frame_type
        )
        actor_state.ik_state.configuration.update(q=state.data.qpos)
        current_transform = ik_util.get_transform_frame_to_world(
            state.mj_model,
            state.data,
            self.ik.frame_name,
            self.ik.frame_type
        )
        
        current_position, current_orientation = current_transform.translation(), current_transform.rotation().as_rpy_radians()
        actor_state.ik_state.eef_task.set_target(
            current_transform
        )
        return state, common_state, dataclass_replace(
            actor_state,
            target_position=current_position,
            target_orientation_euler=current_orientation,
            inner_actor_remaining_elapsed=self.actor.control_timestep
        )
    
    def onboard_step(
        self,
        state: MujocoFuncWorldState,
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: MinkIKWrapperState[ActorStateT, ActorWrapperActT],
        last_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MinkIKWrapperState[ActorStateT, ActorWrapperActT]
    ]:
        state, common_state, actor_state = super().onboard_step(state, common_state, actor_state, last_step_elapsed)
        actor_state.inner_actor_remaining_elapsed -= last_step_elapsed
        state, ik_state, target_qpos = self.ik.step(
            state,
            actor_state.ik_state,
            actor_state.target_position,
            actor_state.target_orientation_euler,
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
        actor_state: MinkIKWrapperState[ActorStateT, ActorWrapperActT],
        action: Optional[ActorWrapperActT],
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MinkIKWrapperState[ActorStateT, ActorWrapperActT]
    ]:
        target_position, target_orientation = self.fn_target_position_and_rotation(action)
        actor_state.ik_state.eef_task.set_target(
            mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3.from_rpy_radians(*target_orientation),
                translation=target_position
            )
        )
        return state, common_state, dataclass_replace(
            actor_state,
            target_position=target_position,
            target_orientation_euler=target_orientation,
            target_action=action
        )
    


        