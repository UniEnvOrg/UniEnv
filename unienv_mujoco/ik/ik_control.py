from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable
from unienv_interface.env_base.funcenv import FuncEnvCommonState
from unienv_interface.world.actor import FuncActorCombinedState, FuncActor, FuncActorWrapper, ActorStateT, ActorActT, ActorWrapperActT, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType, ActorWrapperStateT
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
    home_task : Optional[mink.PostureTask] = None

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
        mj_model : mujoco.MjModel,
        mj_data : mujoco.MjData,
    ) -> MinkIKState:
        limits = [
            mink.ConfigurationLimit(
                mj_model
            ),
        ]
        eef_task = mink.FrameTask(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=2.0
        )
        try:
            q0 = mj_model.key("home").qpos
            home_task = mink.PostureTask(
                cost=0.1
            )
            home_task.set_target(q0)
        except:
            home_task = None

        if self.collision_avoid_geom_pairs is not None:
            limits.append(mink.CollisionAvoidanceLimit(
                mj_model,
                geom_pairs=self.collision_avoid_geom_pairs
            ))
        if self.max_velocity_per_joint is not None:
            limits.append(mink.VelocityLimit(
                mj_model,
                max_velocity_per_joint=self.max_velocity_per_joint
            ))

        return MinkIKState(
            configuration=mink.Configuration(
                model=mj_model, 
                q=mj_data.qpos
            ),
            limits=limits,
            eef_task=eef_task,
            home_task=home_task
        )

    def step(
        self,
        ik_state : MinkIKState,
        target_transform : mink.SE3,
        elapsed_seconds : float
    ) -> Tuple[
        MinkIKState,
        np.ndarray,
        bool
    ]:
        ik_state.eef_task.set_target(target_transform)
        converged = False
        for i in range(self.max_iterations):
            vel = mink.solve_ik(
                configuration=ik_state.configuration,
                tasks=[ik_state.eef_task] if ik_state.home_task is None else [ik_state.eef_task, ik_state.home_task],
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
                converged = True
                break
        
        return ik_state, ik_state.configuration.q, converged

    def update_q(
        self,
        ik_state : MinkIKState,
        q : np.ndarray
    ) -> MinkIKState:
        ik_state.configuration.update(q=q)
        return ik_state
    
    def get_target_from_data(
        self,
        ik_state : MinkIKState,
        mj_data : mujoco.MjData
    ) -> mink.SE3:
        return ik_util.get_transform_frame_to_world(
            ik_state.configuration.model,
            mj_data,
            self.frame_name,
            self.frame_type
        )

@dataclass
class MinkIKWrapperState(Generic[ActorStateT, ActorWrapperActT]):
    inner_actor_state : ActorStateT
    target_transform : mink.SE3
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
        fn_target_transform : Callable[[ActorWrapperActT], mink.SE3],
        fn_action_transform : Callable[[Optional[ActorWrapperActT], np.ndarray], ActorActT],
    ):
        super().__init__(actor)
        self.ik = ik
        self._action_space = new_action_space
        self.fn_target_transform = fn_target_transform
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
        ik_state = self.ik.initial(state.mj_model, state.data)
        current_transform = self.ik.get_target_from_data(
            ik_state,
            state.data
        )
        return state, common_state, MinkIKWrapperState(
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
        actor_state = dataclass_replace(
            actor_state,
            ik_state=self.ik.update_q(
                actor_state.ik_state,
                state.data.qpos
            )
        )
        current_transform = self.ik.get_target_from_data(
            actor_state.ik_state,
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
        actor_state: MinkIKWrapperState[ActorStateT, ActorWrapperActT],
        last_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MinkIKWrapperState[ActorStateT, ActorWrapperActT]
    ]:
        state, common_state, actor_state = super().onboard_step(state, common_state, actor_state, last_step_elapsed)
        actor_state.inner_actor_remaining_elapsed -= last_step_elapsed
        ik_state, target_qpos, ik_converged = self.ik.step(
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
        actor_state: MinkIKWrapperState[ActorStateT, ActorWrapperActT],
        action: Optional[ActorWrapperActT],
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        MinkIKWrapperState[ActorStateT, ActorWrapperActT]
    ]:
        target_transform = self.fn_target_transform(action)
        return state, common_state, dataclass_replace(
            actor_state,
            target_transform=target_transform,
            target_action=action
        )
    


        