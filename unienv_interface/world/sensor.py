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

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "Sensor":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["Sensor"]:
        return None
    
    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the attribute `name` exists in the environment."""
        return hasattr(self, name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        """Gets the attribute `name` from the environment."""
        return getattr(self, name)
    
    def set_wrapper_attr(self, name: str, value: Any):
        """Sets the attribute `name` on the environment with `value`."""
        setattr(self, name, value)

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

    def close(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        sensor_state : SensorStateT
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType]
    ]:
        return state, common_state
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncSensor":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["FuncSensor"]:
        return None
    
    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the attribute `name` exists in the environment."""
        return hasattr(self, name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        """Gets the attribute `name` from the environment."""
        return getattr(self, name)
    
    def set_wrapper_attr(self, name: str, value: Any):
        """Sets the attribute `name` on the environment with `value`."""
        setattr(self, name, value)

WrapperDataT = TypeVar("WrapperDataT")
WrapperDeviceT = TypeVar("WrapperDeviceT")
WrapperDtypeT = TypeVar("WrapperDtypeT")
WrapperRNGT = TypeVar("WrapperRNGT")

class SensorWrapper(
    Generic[
        WrapperDataT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT,
        SensorDataT, BDeviceType, BDtypeType, BRNGType
    ],
    Sensor[SensorDataT, BDeviceType, BDtypeType, BRNGType]
):
    def __init__(
        self,
        sensor : Sensor[SensorDataT, BDeviceType, BDtypeType, BRNGType]
    ):
        self.sensor = sensor
        self._observation_space : Optional[Space[WrapperDataT, Any, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]] = None
    
    @property
    def observation_space(self) -> Space[WrapperDataT, Any, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]:
        return self._observation_space or self.sensor.observation_space
    
    @observation_space.setter
    def observation_space(self, value : Space[WrapperDataT, Any, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]) -> None:
        self._observation_space = value

    @property
    def control_timestep(self) -> float:
        return self.sensor.control_timestep
    
    @control_timestep.setter
    def control_timestep(self, value : float) -> None:
        self.sensor.control_timestep = value
    
    def update(self, last_step_elapsed: float) -> None:
        self.sensor.update(last_step_elapsed)
    
    def get_data(self) -> WrapperDataT:
        return self.sensor.get_data()
    
    def reset(self) -> None:
        self.sensor.reset()

    def close(self) -> None:
        self.sensor.close()

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "Sensor":
        return self.sensor.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> "Sensor[SensorDataT, BDeviceType, BDtypeType, BRNGType]":
        return self.sensor
    
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or self.sensor.has_wrapper_attr(name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.sensor.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        sub_sensor = self
        attr_set = False

        while attr_set is False and sub_sensor is not None:
            if hasattr(sub_sensor, name):
                setattr(sub_sensor, name, value)
                attr_set = True
            else:
                sub_sensor = sub_sensor.prev_wrapper_layer

        if attr_set is False and sub_sensor is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )
        
WrapperStateT = TypeVar("WrapperStateT")
class FuncSensorWrapper(
    Generic[
        StateType,
        WrapperStateT, WrapperDataT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT,
        SensorStateT, SensorDataT, BDeviceType, BDtypeType, BRNGType
    ],
    FuncSensor[StateType, WrapperStateT, WrapperDataT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]
):
    def __init__(
        self,
        sensor : FuncSensor[StateType, SensorStateT, SensorDataT, BDeviceType, BDtypeType, BRNGType]
    ):
        self.sensor = sensor
        self._observation_space : Optional[Space[WrapperDataT, Any, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]] = None
    
    @property
    def observation_space(self) -> Space[WrapperDataT, Any, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]:
        return self._observation_space or self.sensor.observation_space
    
    @observation_space.setter
    def observation_space(self, value : Space[WrapperDataT, Any, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]) -> None:
        self._observation_space = value

    @property
    def control_timestep(self) -> float:
        return self.sensor.control_timestep
    
    @control_timestep.setter
    def control_timestep(self, value : float) -> None:
        self.sensor.control_timestep = value

    def initial(
        self,
        state : StateType, 
        common_state : FuncEnvCommonState[WrapperDeviceT, WrapperRNGT], 
        *args, 
        seed : int,
        **kwargs
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[WrapperDeviceT, WrapperRNGT],
        WrapperStateT
    ]:
        return self.sensor.initial(
            state,
            common_state,
            *args,
            seed=seed,
            **kwargs
        )

    def reset(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[WrapperDeviceT, WrapperRNGT],
        sensor_state : WrapperStateT
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[WrapperDeviceT, WrapperRNGT],
        WrapperStateT
    ]:
        return self.sensor.reset(
            state,
            common_state,
            sensor_state
        )
    
    def step(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[WrapperDeviceT, WrapperRNGT],
        sensor_state : WrapperStateT,
        last_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[WrapperDeviceT, WrapperRNGT],
        WrapperStateT
    ]:
        return self.sensor.step(
            state,
            common_state,
            sensor_state,
            last_step_elapsed
        )
    
    def get_data(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[WrapperDeviceT, WrapperRNGT],
        sensor_state : WrapperStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[WrapperDeviceT, WrapperRNGT],
        WrapperStateT,
        WrapperDataT
    ]:
        return self.sensor.get_data(
            state,
            common_state,
            sensor_state,
            last_control_step_elapsed
        )
    
    def close(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[WrapperDeviceT, WrapperRNGT],
        sensor_state : WrapperStateT
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[WrapperDeviceT, WrapperRNGT]
    ]:
        return self.sensor.close(
            state,
            common_state,
            sensor_state
        )

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncSensor":
        return self.sensor.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> "FuncSensor[StateType, SensorStateT, SensorDataT, BDeviceType, BDtypeType, BRNGType]":
        return self.sensor
    
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or self.sensor.has_wrapper_attr(name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.sensor.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        sub_sensor = self
        attr_set = False

        while attr_set is False and sub_sensor is not None:
            if hasattr(sub_sensor, name):
                setattr(sub_sensor, name, value)
                attr_set = True
            else:
                sub_sensor = sub_sensor.prev_wrapper_layer

        if attr_set is False and sub_sensor is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )