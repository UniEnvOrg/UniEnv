from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type
from abc import ABC, abstractmethod
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space import Space

class Sensor(ABC, Generic[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    observation_space : Space[Any, BDeviceType, BDtypeType, BRNGType]
    control_timestep : float
    
    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.observation_space.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.observation_space.device

    def pre_environment_step(self, dt : float) -> None:
        """
        This method is called before the environment step
        Args:
            dt (float): The time elapsed between the last world step and the current step.
        """
        pass

    def update(self, dt : float) -> None:
        """
        This method is called after the environment step to update the sensor's internal state.
        Args:
            dt (float): The time elapsed between the last world step and the current step.
        """
        pass

    @abstractmethod
    def get_observation(self) -> Any:
        """
        Get the current observation from the sensor.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset the sensor and update its internal state.
        When this method is called, the `update` method will not be called until the next environment step.
        """
        self.update()
    
    def close(self) -> None:
        """
        Close the sensor.
        This method should be called to clean up any resources used by the actor.
        """
        pass

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

WrapperArrayT = TypeVar("WrapperBArrayT")
WrapperDeviceT = TypeVar("WrapperDeviceT")
WrapperDtypeT = TypeVar("WrapperDtypeT")
WrapperRNGT = TypeVar("WrapperRNGT")

class SensorWrapper(
    Generic[
        WrapperArrayT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT,
        BArrayType, BDeviceType, BDtypeType, BRNGType
    ],
    Sensor[WrapperArrayT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]
):
    def __init__(
        self,
        sensor : Sensor[BArrayType, BDeviceType, BDtypeType, BRNGType]
    ):
        self.sensor = sensor
        self._observation_space : Optional[Space[WrapperArrayT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]] = None
    
    @property
    def observation_space(self) -> Space[WrapperArrayT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]:
        return self._observation_space or self.sensor.observation_space
    
    @observation_space.setter
    def observation_space(self, value : Space[WrapperArrayT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]) -> None:
        self._observation_space = value

    @property
    def control_timestep(self) -> float:
        return self.sensor.control_timestep
    
    @control_timestep.setter
    def control_timestep(self, value : float) -> None:
        self.sensor.control_timestep = value
    
    def get_observation(self):
        return self.sensor.get_observation()

    def pre_environment_step(self, dt) -> None:
        self.sensor.pre_environment_step(dt)

    def update(self, dt) -> None:
        self.sensor.update(dt)

    def reset(self) -> None:
        self.sensor.reset()
    
    def close(self) -> None:
        self.sensor.close()

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "Sensor":
        return self.sensor.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> "Sensor[BArrayType, BDeviceType, BDtypeType, BRNGType]":
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