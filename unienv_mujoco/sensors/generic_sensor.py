from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable, Literal
from abc import ABC, abstractmethod

import mujoco.viewer

from unienv_interface.env_base.funcenv import FuncEnvCommonState
from unienv_interface.world.sensor import FuncSensor
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import Space, Box
import mujoco
from dm_control import mjcf
from dataclasses import dataclass, replace as dataclass_replace
import numpy as np
from .. import mjcf_util
from ..base.world import MujocoFuncWorldState, MujocoFuncWorld

class MujocoFuncGenericSensor(
    FuncSensor[MujocoFuncWorldState, None, np.ndarray, Any, np.dtype, np.random.Generator]
):
    def __init__(
        self,
        mjmodel : mujoco.MjModel,
        sensor_name : str,
        control_timestep : float,
    ):
        assert control_timestep > 0.0
        self.control_timestep = control_timestep

        self._sensor_name = sensor_name
        ndim = mjmodel.sensor(sensor_name).dim[0]
        self.observation_space = Box(
            backend=NumpyComputeBackend,
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            device=None,
            shape=(ndim,)
        )
    
    @property
    def sensor_name(self) -> str:
        return self._sensor_name
    
    def initial(
        self,
        world : MujocoFuncWorld,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        seed : int
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        None
    ]:
        return state, common_state, None

    def reset(
        self,
        world : MujocoFuncWorld,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        sensor_state : None
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        None
    ]:
        return state, common_state, None

    def step(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        sensor_state : None,
        last_step_elapsed : float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        None
    ]:
        return state, common_state, None
    
    def get_data(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        sensor_state: None,
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator], 
        None, 
        np.ndarray
    ]:
        sensor_data = state.data.sensor(self.sensor_name).data.astype(np.float32).copy()
        return state, common_state, None, sensor_data

    def close(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        sensor_state: None
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator]
    ]:
        return state, common_state