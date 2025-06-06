from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from xbarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space import Space
from unienv_interface.env_base.funcenv import FuncEnv, StateType
from .world import FuncWorld

SensorDataT = TypeVar("SensorDataT", covariant=True)
class Sensor(ABC, Generic[SensorDataT, BDeviceType, BDtypeType, BRNGType]):
    observation_space : Space[SensorDataT, BDeviceType, BDtypeType, BRNGType]
    control_timestep : float
    
    @property
    def backend(self) -> ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]:
        return self.observation_space.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.observation_space.device

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
    observation_space : Space[SensorDataT, BDeviceType, BDtypeType, BRNGType]
    control_timestep : float

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.observation_space.device
    
    @property
    def backend(self) -> ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]:
        return self.observation_space.backend
    
    @abstractmethod
    def initial(
        self,
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        state : StateType, 
        rng : BRNGType, 
    ) -> Tuple[
        StateType,
        BRNGType,
        SensorStateT
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def reset(
        self,
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        state : StateType,
        rng : BRNGType,
        sensor_state : SensorStateT
    ) -> Tuple[
        StateType,
        BRNGType,
        SensorStateT
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self,
        state : StateType,
        rng : BRNGType,
        sensor_state : SensorStateT,
        last_step_elapsed : float
    ) -> Tuple[
        StateType,
        BRNGType,
        SensorStateT
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def get_data(
        self,
        state : StateType,
        rng : BRNGType,
        sensor_state : SensorStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        BRNGType,
        SensorStateT,
        SensorDataT
    ]:
        """Get the data from the sensor if the sensor is readable."""
        raise NotImplementedError

    def close(
        self,
        state : StateType,
        rng : BRNGType,
        sensor_state : SensorStateT
    ) -> Tuple[
        StateType,
        BRNGType
    ]:
        return state, rng
    
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
        self._observation_space : Optional[Space[WrapperDataT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]] = None
    
    @property
    def observation_space(self) -> Space[WrapperDataT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]:
        return self._observation_space or self.sensor.observation_space
    
    @observation_space.setter
    def observation_space(self, value : Space[WrapperDataT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]) -> None:
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
        self._observation_space : Optional[Space[WrapperDataT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]] = None
    
    @property
    def observation_space(self) -> Space[WrapperDataT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]:
        return self._observation_space or self.sensor.observation_space
    
    @observation_space.setter
    def observation_space(self, value : Space[WrapperDataT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]) -> None:
        self._observation_space = value

    @property
    def control_timestep(self) -> float:
        return self.sensor.control_timestep
    
    @control_timestep.setter
    def control_timestep(self, value : float) -> None:
        self.sensor.control_timestep = value

    def initial(
        self,
        world : FuncWorld[StateType, WrapperDeviceT, WrapperDtypeT, WrapperRNGT],
        state : StateType, 
        rng : WrapperRNGT, 
        *args, 
        **kwargs
    ) -> Tuple[
        StateType,
        WrapperRNGT,
        WrapperStateT
    ]:
        return self.sensor.initial(
            world,
            state,
            rng,
            *args,
            **kwargs
        )

    def reset(
        self,
        world : FuncWorld[StateType, WrapperDeviceT, WrapperDtypeT, WrapperRNGT],
        state : StateType,
        rng : WrapperRNGT,
        sensor_state : WrapperStateT
    ) -> Tuple[
        StateType,
        WrapperRNGT,
        WrapperStateT
    ]:
        return self.sensor.reset(
            world,
            state,
            rng,
            sensor_state
        )
    
    def step(
        self,
        state : StateType,
        rng : WrapperRNGT,
        sensor_state : WrapperStateT,
        last_step_elapsed : float
    ) -> Tuple[
        StateType,
        WrapperRNGT,
        WrapperStateT
    ]:
        return self.sensor.step(
            state,
            rng,
            sensor_state,
            last_step_elapsed
        )
    
    def get_data(
        self,
        state : StateType,
        rng : WrapperRNGT,
        sensor_state : WrapperStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        WrapperRNGT,
        WrapperStateT,
        WrapperDataT
    ]:
        return self.sensor.get_data(
            state,
            rng,
            sensor_state,
            last_control_step_elapsed
        )
    
    def close(
        self,
        state : StateType,
        rng : WrapperRNGT,
        sensor_state : WrapperStateT
    ) -> Tuple[
        StateType,
        WrapperRNGT
    ]:
        return self.sensor.close(
            state,
            rng,
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