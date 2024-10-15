from typing import Optional, Any, Dict, Generic, TypeVar, Type, Tuple
from .actor import Actor
from .sensor import Sensor
from ..backends.base import ComputeBackend, BDeviceType, BDtypeType, BRNGType
from ..env_base.funcenv import FuncEnvCommonState, FuncEnv, StateType
from abc import ABC, abstractmethod

class World(ABC):

    """The world timestep in seconds, if None, the world is asynchronous (real-time)"""
    world_timestep : Optional[float]

    """The last step elapsed time in seconds"""
    last_step_elapsed : float

    """If the world is simulated or real"""
    is_real : bool

    @abstractmethod
    def step(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError

class FuncWorld(
    ABC,
    Generic[StateType, BDeviceType, BRNGType]
):
    world_timestep : Optional[float]
    is_real : bool

    backend : Type[ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]]
    
    @abstractmethod
    def initial(
        self, *, seed : int
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType]
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def reset(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType]
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType]
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType]
    ) -> Tuple[
        float, # elapsed time
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType]
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def close(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType]
    ) -> None:
        raise NotImplementedError