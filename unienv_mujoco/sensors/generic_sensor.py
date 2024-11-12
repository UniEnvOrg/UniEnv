from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable, Literal
from abc import ABC, abstractmethod

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
        world : MujocoFuncWorld,
        sensor_name : str,
        control_timestep : float,
    ):
        assert control_timestep > 0.0
        self.control_timestep = control_timestep

        self._sensor_name = sensor_name
        ndim = world._mjmodel.sensor(sensor_name).dim[0]
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
        rng : np.random.Generator,
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        None
    ]:
        return state, rng, None

    def reset(
        self,
        world : MujocoFuncWorld,
        state : MujocoFuncWorldState,
        rng : np.random.Generator,
        sensor_state : None
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        None
    ]:
        return state, rng, None

    def step(
        self,
        state : MujocoFuncWorldState,
        rng : np.random.Generator,
        sensor_state : None,
        last_step_elapsed : float
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        None
    ]:
        return state, rng, None
    
    def get_data(
        self, 
        state: MujocoFuncWorldState, 
        rng: np.random.Generator, 
        sensor_state: None,
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState, 
        np.random.Generator, 
        None, 
        np.ndarray
    ]:
        sensor_data = state.data.sensor(self.sensor_name).data.astype(np.float32).copy()
        return state, rng, None, sensor_data
