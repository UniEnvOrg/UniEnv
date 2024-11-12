from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable
from dataclasses import dataclass, replace as dataclass_replace
from abc import ABC, abstractmethod
import mink
import mujoco
import numpy as np
from . import ik_util
import copy

MujocoIKStateT = TypeVar("IKStateT")
MujocoIKTargetT = TypeVar("IKTargetT")

class MujocoIKClass(ABC, Generic[MujocoIKStateT, MujocoIKTargetT]):
    is_eef_relative : bool = False
    @abstractmethod
    def initial(
        self,
        mj_model : mujoco.MjModel,
        qpos : np.ndarray
    ) -> MujocoIKStateT:
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self,
        mj_model : mujoco.MjModel,
        qpos : np.ndarray,
        ik_state : MujocoIKStateT,
        target_transform : MujocoIKTargetT,
        elapsed_seconds : float
    ) -> Tuple[
        MujocoIKStateT,
        np.ndarray,
        bool
    ]:
        raise NotImplementedError

    def update_q(
        self,
        ik_state : MujocoIKStateT,
        q : np.ndarray
    ) -> MujocoIKStateT:
        return ik_state

    @abstractmethod
    def get_target_from_data(
        self,
        ik_state : MujocoIKStateT,
        mj_model : mujoco.MjModel,
        mj_data : mujoco.MjData
    ) -> MujocoIKTargetT:
        raise NotImplementedError
    
    @abstractmethod
    def target_from_se3(
        self,
        translation : np.ndarray,
        quaternion : np.ndarray
    ) -> MujocoIKTargetT:
        raise NotImplementedError
    
    @abstractmethod
    def translation_from_target(
        self,
        target : MujocoIKTargetT
    ) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def quaternion_from_target(
        self,
        target : MujocoIKTargetT
    ) -> np.ndarray:
        raise NotImplementedError

    def close(
        self,
        ik_state : MujocoIKStateT
    ) -> None:
        pass

@dataclass
class MinkIKState:
    configuration : mink.Configuration
    limits : List[mink.Limit]
    eef_task : Union[mink.FrameTask, mink.RelativeFrameTask]
    home_task : Optional[mink.PostureTask] = None


class MinkIK(MujocoIKClass[MinkIKState, mink.SE3]):
    def __init__(
        self,
        collision_avoid_geom_pairs : Optional[List[Tuple[List[Union[str,int]], List[Union[str,int]]]]],
        max_velocity_per_joint : Optional[Dict[str, float]],
        frame_name : str,
        frame_type : str,
        relative_frame_name : Optional[str] = None,
        relative_frame_type : Optional[str] = None,

        pos_threshold : float = 1e-4,
        ori_threshold : float = 1e-4,
        solver : str = "quadprog",
        max_iterations : int = 20,
    ):
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.relative_frame_name = relative_frame_name
        self.relative_frame_type = relative_frame_type
        self.collision_avoid_geom_pairs = collision_avoid_geom_pairs
        self.max_velocity_per_joint = max_velocity_per_joint
        
        self.pos_threshold = pos_threshold
        self.ori_threshold = ori_threshold
        self.solver = solver
        self.max_iterations = max_iterations
    
    @property
    def is_eef_relative(self) -> bool:
        return self.relative_frame_name is not None

    def initial(
        self,
        mj_model : mujoco.MjModel,
        qpos : np.ndarray
    ) -> MinkIKState:
        limits = [
            mink.ConfigurationLimit(
                mj_model
            ),
        ]
        if self.relative_frame_name is None:
            eef_task = mink.FrameTask(
                frame_name=self.frame_name,
                frame_type=self.frame_type,
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=2.0
            )
        else:
            eef_task = mink.RelativeFrameTask(
                frame_name=self.frame_name,
                frame_type=self.frame_type,
                root_name=self.relative_frame_name,
                root_type=self.relative_frame_type,
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
                q=qpos
            ),
            limits=limits,
            eef_task=eef_task,
            home_task=home_task
        )

    def step(
        self,
        mj_model : mujoco.MjModel,
        qpos : np.ndarray,
        ik_state : MinkIKState,
        target_transform : mink.SE3,
        elapsed_seconds : float
    ) -> Tuple[
        MinkIKState,
        np.ndarray,
        bool
    ]:
        ik_state.eef_task.set_target(target_transform)
        ik_state.configuration.update(q=qpos)
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
            pos_err = np.linalg.norm(err[:3])
            ori_err = np.linalg.norm(err[3:])
            pos_achieved = pos_err <= self.pos_threshold
            ori_achieved = ori_err <= self.ori_threshold
            score = ik_state.eef_task.position_cost * pos_err + ik_state.eef_task.orientation_cost * ori_err
            if pos_achieved and ori_achieved:
                converged = True
                break
        
        return ik_state, ik_state.configuration.q, converged

    def get_target_from_data(
        self,
        ik_state : MinkIKState,
        mj_model : mujoco.MjModel,
        mj_data : mujoco.MjData
    ) -> mink.SE3:
        world_transform = ik_util.get_transform_frame_to_world(
            mj_model,
            mj_data,
            self.frame_name,
            self.frame_type
        )

        if self.relative_frame_name is None:
            return world_transform
        else:
            base_transform = ik_util.get_transform_frame_to_world(
                mj_model,
                mj_data,
                self.relative_frame_name,
                self.relative_frame_type
            )
            return ik_util.get_relative_transform(
                base_transform,
                world_transform
            )
        
    def target_from_se3(
        self,
        translation : np.ndarray,
        quaternion : np.ndarray
    ) -> mink.SE3:
        return mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(quaternion),
            translation=translation
        )
    
    def translation_from_target(
        self,
        target : mink.SE3
    ) -> np.ndarray:
        return target.translation().copy()
    
    def quaternion_from_target(
        self,
        target : mink.SE3
    ) -> np.ndarray:
        return target.rotation().wxyz.copy()

class MinkBulkIK(MujocoIKClass[MinkIKState, mink.SE3]):
    @staticmethod
    def solve_ik_with_score(
        ik: MinkIK,
        ik_state: MinkIKState,
        start_q: Optional[np.ndarray],
        elapsed_seconds: float
    ) -> Tuple[MinkIKState, np.ndarray, float, bool]:
        if start_q is not None:
            if start_q.shape[0] < ik_state.configuration.nq:
                start_q = np.concat([
                    start_q,
                    np.zeros(ik_state.configuration.nq - start_q.shape[0])
                ])
            ik_state.configuration.update(q=start_q)
        
        converged = False
        for i in range(ik.max_iterations):
            vel = mink.solve_ik(
                configuration=ik_state.configuration,
                tasks=[ik_state.eef_task] if ik_state.home_task is None else [ik_state.eef_task, ik_state.home_task],
                dt=elapsed_seconds,
                solver=ik.solver,
                damping=1e-3,
                limits=ik_state.limits
            )
            ik_state.configuration.integrate_inplace(vel, elapsed_seconds)
            err = ik_state.eef_task.compute_error(ik_state.configuration)
            pos_err = np.linalg.norm(err[:3])
            ori_err = np.linalg.norm(err[3:])
            pos_achieved = pos_err <= ik.pos_threshold
            ori_achieved = ori_err <= ik.ori_threshold
            score = ik_state.eef_task.position_cost * pos_err + ik_state.eef_task.orientation_cost * ori_err
            if pos_achieved and ori_achieved:
                converged = True
                break
        
        return ik_state, ik_state.configuration.q, score, converged

    def __init__(
        self,
        ik : MinkIK,

        # Additional starting qpos to try, shape (num_add, num_joints)
        additional_search_qpos : np.ndarray,
    ):
        self.ik = ik
        self.additional_search_qpos = additional_search_qpos

    @property
    def is_eef_relative(self) -> bool:
        return self.ik.is_eef_relative

    def initial(
        self,
        mj_model : mujoco.MjModel,
        qpos : np.ndarray
    ) -> MinkIKState:
        ik_state = self.ik.initial(mj_model, qpos)
        return ik_state
    
    def try_solve_ik(
        self,
        ik_state : MinkIKState,
        current_qpos : np.ndarray,
        elapsed_seconds : float
    ) -> Tuple[
        MinkIKState,
        np.ndarray,
        bool
    ]:
        previous_q = ik_state.configuration.q
        ik_state, target_q, score, converged = self.solve_ik_with_score(
            self.ik,
            ik_state,
            start_q=current_qpos,
            elapsed_seconds=elapsed_seconds
        )
        if converged:
            return ik_state, target_q, converged
        
        min_score_q = target_q
        min_score = score
        
        for start_qpos in self.additional_search_qpos:
            ik_state, target_q, score, converged = self.solve_ik_with_score(
                self.ik,
                ik_state,
                start_q=start_qpos,
                elapsed_seconds=elapsed_seconds
            )
            if converged:
                return ik_state, target_q, converged

            if score < min_score:
                min_score_q = target_q
                min_score = score
            
        return ik_state, min_score_q, False

    def step(
        self,
        mj_model : mujoco.MjModel,
        qpos : np.ndarray,
        ik_state : MinkIKState,
        target_transform : mink.SE3,
        elapsed_seconds : float
    ) -> Tuple[
        MinkIKState,
        np.ndarray,
        bool
    ]:
        current_qpos = qpos.copy()
        ik_state.eef_task.set_target(target_transform)
        ik_state, target_q, converged = self.try_solve_ik(
            ik_state,
            current_qpos,
            elapsed_seconds
        )
        return ik_state, target_q, converged
    
    def get_target_from_data(
        self,
        ik_state : MinkIKState,
        mj_model : mujoco.MjModel,
        mj_data : mujoco.MjData
    ) -> mink.SE3:
        return self.ik.get_target_from_data(ik_state, mj_model, mj_data)

    def target_from_se3(
        self,
        translation : np.ndarray,
        quaternion : np.ndarray
    ) -> mink.SE3:
        return self.ik.target_from_se3(translation, quaternion)
    
    def translation_from_target(
        self,
        target : mink.SE3
    ) -> np.ndarray:
        return self.ik.translation_from_target(target)
    
    def quaternion_from_target(
        self,
        target : mink.SE3
    ) -> np.ndarray:
        return self.ik.quaternion_from_target(target)