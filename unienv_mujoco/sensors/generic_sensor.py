from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable, Literal
from abc import ABC, abstractmethod

import mujoco.viewer

from unienv_interface.env_base.funcenv import FuncEnvCommonState
from unienv_interface.world.sensor import FuncSensorSingleState, FuncSensor
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
        model : mujoco.MjModel,
        sensor_name : str,
        control_timestamp : float,
        seed : Optional[int] = None,
    ):
        assert control_timestamp > 0.0
        self.control_timestamp = control_timestamp

        self._sensor_name = sensor_name
        ndim = model.sensor(sensor_name).dim
        self.observation_space = Box(
            backend=NumpyComputeBackend,
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            device=None,
            shape=(ndim,),
            seed=seed
        )
    
    @property
    def sensor_name(self) -> str:
        return self._sensor_name
    
    def initial(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        seed : int
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        FuncSensorSingleState[None]
    ]:
        return state, common_state, FuncSensorSingleState(
            sensor_state=None,
            remaining_time_until_read=0
        )

    def reset(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        sensor_single_state : FuncSensorSingleState[None]
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        FuncSensorSingleState[None]
    ]:
        return state, common_state, dataclass_replace(
            sensor_single_state,
            remaining_time_until_read=0
        )

    def step(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        sensor_single_state : FuncSensorSingleState[None],
        last_step_elapsed : float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        FuncSensorSingleState[None]
    ]:
        return state, common_state, dataclass_replace(
            sensor_single_state,
            remaining_time_until_read=sensor_single_state.remaining_time_until_read - last_step_elapsed
        )
    
    def get_data(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        sensor_single_state: FuncSensorSingleState[None]
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator], 
        FuncSensorSingleState[None], 
        np.ndarray
    ]:
        sensor_data = state.data.sensor(self.sensor_name).data.astype(np.float32).copy()
        return state, common_state, dataclass_replace(
            sensor_single_state,
            remaining_time_until_read=self.control_timestamp
        ), sensor_data

    def close(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        sensor_single_state: FuncSensorSingleState[None]
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator]
    ]:
        return state, common_state