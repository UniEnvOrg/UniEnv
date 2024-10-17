from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..space import Space
from ..backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from ..env_base.funcenv import FuncEnvCommonState, FuncEnv, StateType

SensorDataT = TypeVar("SensorDataT", covariant=True)
class Sensor(ABC, Generic[SensorDataT, BDeviceType, BDtypeType, BRNGType]):
    observation_space : Space[SensorDataT, Any, BDeviceType, BDtypeType, BRNGType]
    control_timestep : float

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.observation_space.device
    
    @property
    def backend(self) -> Type[ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]]:
        return self.observation_space.backend

    @abstractmethod
    def update(self, last_step_elapsed : float) -> None:
        """Update the sensor with a new state (e.g. from the environment)."""
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> SensorDataT:
        """Get the data from the sensor."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the sensor."""
        raise NotImplementedError
    
    @abstractmethod
    def close(self) -> None:
        """Close the sensor."""
        raise NotImplementedError

    def __del__(self) -> None:
        self.close()

SensorStateT = TypeVar("SensorStateT")

class FuncSensor(
    ABC,
    Generic[StateType, SensorStateT, SensorDataT, BDeviceType, BDtypeType, BRNGType]
):
    observation_space : Space[SensorDataT, Any, BDeviceType, BDtypeType, BRNGType]
    control_timestep : float

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.observation_space.device
    
    @property
    def backend(self) -> Type[ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]]:
        return self.observation_space.backend
    
    @abstractmethod
    def initial(
        self,
        state : StateType, 
        common_state : FuncEnvCommonState[BDeviceType, BRNGType], 
        *, 
        seed : int
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        SensorStateT
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def reset(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        sensor_state : SensorStateT
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        SensorStateT
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        sensor_state : SensorStateT,
        last_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        SensorStateT
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def get_data(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        sensor_state : SensorStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        SensorStateT,
        SensorDataT
    ]:
        """Get the data from the sensor if the sensor is readable."""
        raise NotImplementedError

    @abstractmethod
    def close(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        sensor_state : SensorStateT
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType]
    ]:
        raise NotImplementedError