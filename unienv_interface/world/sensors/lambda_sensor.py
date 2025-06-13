from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable
from ..sensor import Sensor, SensorDataT, FuncSensor, SensorStateT
from ..world import FuncWorld
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from unienv_interface.env_base.funcenv import StateType


class LambdaSensor(
    Sensor[SensorDataT, BDeviceType, BDtypeType, BRNGType],
    Generic[SensorDataT, BDeviceType, BDtypeType, BRNGType]
):
    def __init__(
        self,
        observation_space : Space[SensorDataT, BDeviceType, BDtypeType, BRNGType],
        control_timestep : float,
        data_fn : Callable[[], SensorDataT],
    ):
        self.observation_space = observation_space
        self.control_timestep = control_timestep
        self.data_fn = data_fn
    
    def update(self, last_step_elapsed : float) -> None:
        pass

    def get_data(self) -> SensorDataT:
        return self.data_fn()
    
    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

class FuncLambdaSensor(
    FuncSensor[StateType, None, SensorDataT, BDeviceType, BDtypeType, BRNGType],
    Generic[StateType, SensorDataT, BDeviceType, BDtypeType, BRNGType]
):
    def __init__(
        self,
        observation_space : Space[SensorDataT, BDeviceType, BDtypeType, BRNGType],
        control_timestep : float,
        data_fn : Callable[[StateType], SensorDataT],
    ):
        self.observation_space = observation_space
        self.control_timestep = control_timestep
        self.data_fn = data_fn
    
    def initial(
        self, 
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        state: StateType, 
        rng: BRNGType, 
        *, 
        seed: int
    ) -> Tuple[
        StateType, 
        BRNGType, 
        None
    ]:
        return state, rng, None
    
    def reset(
        self, 
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        state: StateType, 
        rng: BRNGType, 
        sensor_state: None
    ) -> Tuple[
        StateType, 
        BRNGType, 
        None
    ]:
        return state, rng, None
    
    def step(
        self, 
        state: StateType, 
        rng: BRNGType, 
        sensor_state: None, 
        last_step_elapsed: float
    ) -> Tuple[
        StateType, 
        BRNGType, 
        None
    ]:
        return state, rng, None
    
    def get_data(
        self,
        state: StateType,
        rng: BRNGType,
        sensor_state: None,
        last_control_step_elapsed: float
    ) -> Tuple[
        StateType,
        BRNGType,
        None,
        SensorDataT
    ]:
        return state, rng, None, self.data_fn(state)