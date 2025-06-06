from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable, Literal
from abc import ABC, abstractmethod

from unienv_interface.world.sensors.lambda_sensor import FuncLambdaSensor
from xbarray import numpy as NumpyComputeBackend
from unienv_interface.space import DictSpace, BoxSpace
import mujoco
from dm_control import mjcf
from dataclasses import dataclass
import numpy as np
from unienv_mujoco import mjcf_util
from unienv_mujoco.base import *

class MujocoFuncJointPosSensor(
    FuncLambdaSensor[MujocoFuncWorldState, np.ndarray, Any, np.dtype, np.random.Generator]
):
    def __init__(
        self,
        world : MujocoFuncWorld,
        control_timestep : float,
        joint_names : Sequence[str],
    ):
        assert len(joint_names) > 0
        self.joint_names = joint_names
        observation_limits = np.zeros((len(joint_names), 2), dtype=np.float32)
        for i, joint_name in enumerate(joint_names):
            joint = world._mjmodel.joint(joint_name)
            observation_limits[i] = joint.range
        observation_space = BoxSpace(
            backend=NumpyComputeBackend,
            low=observation_limits[:, 0],
            high=observation_limits[:, 1],
            dtype=np.float32,
            device=None
        )
        super().__init__(
            observation_space=observation_space,
            control_timestep=control_timestep,
            data_fn=self.read_joint_positions
        )
    
    def read_joint_positions(self, state : MujocoFuncWorldState) -> np.ndarray:
        qpos = np.zeros(len(self.joint_names), dtype=np.float32)
        for i, joint_name in enumerate(self.joint_names):
            qpos[i] = state.data.joint(joint_name).qpos[0]
        return qpos
    
class MujocoFuncJointVelSensor(
    FuncLambdaSensor[MujocoFuncWorldState, np.ndarray, Any, np.dtype, np.random.Generator]
):
    def __init__(
        self,
        control_timestep : float,
        joint_names : Sequence[str],
    ):
        assert len(joint_names) > 0
        self.joint_names = joint_names

        observation_space = BoxSpace(
            backend=NumpyComputeBackend,
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            device=None,
            shape=(len(joint_names),)
        )
        super().__init__(
            observation_space=observation_space,
            control_timestep=control_timestep,
            data_fn=self.read_joint_velocities
        )
    
    def read_joint_velocities(self, state : MujocoFuncWorldState) -> np.ndarray:
        qvel = np.zeros(len(self.joint_names), dtype=np.float32)
        for i, joint_name in enumerate(self.joint_names):
            qvel[i] = state.data.joint(joint_name).qvel[0]
        return qvel
    
class MujocoFuncJointAccSensor(
    FuncLambdaSensor[MujocoFuncWorldState, np.ndarray, Any, np.dtype, np.random.Generator]
):
    def __init__(
        self,
        control_timestep : float,
        joint_names : Sequence[str],
    ):
        assert len(joint_names) > 0
        self.joint_names = joint_names

        observation_space = BoxSpace(
            backend=NumpyComputeBackend,
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            device=None,
            shape=(len(joint_names),)
        )
        super().__init__(
            observation_space=observation_space,
            control_timestep=control_timestep,
            data_fn=self.read_joint_accelerations
        )
    
    def read_joint_accelerations(self, state : MujocoFuncWorldState) -> np.ndarray:
        qacc = np.zeros(len(self.joint_names), dtype=np.float32)
        for i, joint_name in enumerate(self.joint_names):
            qacc[i] = state.data.joint(joint_name).qacc[0]
        return qacc