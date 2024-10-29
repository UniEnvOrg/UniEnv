from typing import Optional, Any, Dict, Generic, TypeVar, Type, Tuple
from ..backends.base import ComputeBackend, BDeviceType, BDtypeType, BRNGType
from ..env_base.funcenv import FuncEnvCommonState, FuncEnv, StateType
from abc import ABC, abstractmethod

class World(ABC, Generic[BDeviceType, BDtypeType, BRNGType]):
    backend : Type[ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]]

    """The physical timestep in seconds, if None, the world is asynchronous (real-time)"""
    world_subtimestep : Optional[float]

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
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "World":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["World"]:
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

class FuncWorld(
    ABC,
    Generic[StateType, BDeviceType, BDtypeType, BRNGType]
):
    world_subtimestep : Optional[float]
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
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncWorld":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["FuncWorld"]:
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

WrapperBDeviceType = TypeVar("WrapperBDeviceType")
WrapperBDtypeType = TypeVar("WrapperBDtypeType")
WrapperBRNGType = TypeVar("WrapperBRNGType")

class WorldWrapper(
    Generic[
        WrapperBDeviceType, WrapperBDtypeType, WrapperBRNGType,
        BDeviceType, BDtypeType, BRNGType
    ],
    World[WrapperBDeviceType, WrapperBDtypeType, WrapperBRNGType]
):
    def __init__(
        self,
        world : World[BDeviceType, BDtypeType, BRNGType]
    ):
        self.world = world
        self._backend : Optional[ComputeBackend[WrapperBDeviceType, WrapperBDtypeType, WrapperBRNGType]] = None
    
    @property
    def backend(self) -> Type[ComputeBackend[WrapperBDeviceType, WrapperBDtypeType, WrapperBRNGType]]:
        return self._backend or self.world.backend
    
    @backend.setter
    def backend(self, value : Type[ComputeBackend[WrapperBDeviceType, WrapperBDtypeType, WrapperBRNGType]]) -> None:
        self._backend = value
    
    @property
    def world_subtimestep(self) -> Optional[float]:
        return self.world.world_subtimestep
    
    @world_subtimestep.setter
    def world_subtimestep(self, value : Optional[float]) -> None:
        self.world.world_subtimestep = value

    @property
    def world_timestep(self) -> Optional[float]:
        return self.world.world_timestep
    
    @world_timestep.setter
    def world_timestep(self, value : Optional[float]) -> None:
        self.world.world_timestep = value

    @property
    def last_step_elapsed(self) -> float:
        return self.world.last_step_elapsed
    
    @property
    def is_real(self) -> bool:
        return self.world.is_real
    
    def step(self):
        return self.world.step()
    
    def reset(self):
        return self.world.reset()
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "World":
        return self.world.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> "World[BDeviceType, BDtypeType, BRNGType]":
        return self.world
    
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or self.world.has_wrapper_attr(name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.world.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        sub_world = self
        attr_set = False

        while attr_set is False and sub_world is not None:
            if hasattr(sub_world, name):
                setattr(sub_world, name, value)
                attr_set = True
            else:
                sub_world = sub_world.prev_wrapper_layer

        if attr_set is False and sub_world is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )

WrapperStateType = TypeVar("WrapperStateType")
class FuncWorldWrapper(
    Generic[
        WrapperStateType, WrapperBDeviceType, WrapperBDtypeType, WrapperBRNGType,
        StateType, BDeviceType, BDtypeType, BRNGType
    ],
    FuncWorld[WrapperStateType, WrapperBDeviceType, WrapperBDtypeType, WrapperBRNGType]
):
    def __init__(
        self,
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType]
    ):
        self.world = world
        self._backend : Optional[ComputeBackend[WrapperBDeviceType, WrapperBDtypeType, WrapperBRNGType]] = None
    
    @property
    def backend(self) -> Type[ComputeBackend[WrapperBDeviceType, WrapperBDtypeType, WrapperBRNGType]]:
        return self._backend or self.world.backend
    
    @backend.setter
    def backend(self, value : Type[ComputeBackend[WrapperBDeviceType, WrapperBDtypeType, WrapperBRNGType]]) -> None:
        self._backend = value

    @property
    def world_subtimestep(self) -> Optional[float]:
        return self.world.world_subtimestep
    
    @world_subtimestep.setter
    def world_subtimestep(self, value : Optional[float]) -> None:
        self.world.world_subtimestep = value

    @property
    def world_timestep(self) -> Optional[float]:
        return self.world.world_timestep
    
    @world_timestep.setter
    def world_timestep(self, value : Optional[float]) -> None:
        self.world.world_timestep = value

    @property
    def is_real(self) -> bool:
        return self.world.is_real
    
    def initial(
        self,
        *args,
        seed : int,
        **kwargs
    ) -> Tuple[
        WrapperStateType,
        FuncEnvCommonState[WrapperBDeviceType, WrapperBRNGType]
    ]:
        return self.world.initial(*args, seed=seed, **kwargs)
    
    def reset(
        self,
        state : WrapperStateType,
        common_state : FuncEnvCommonState[WrapperBDeviceType, WrapperBRNGType]
    ) -> Tuple[
        WrapperStateType,
        FuncEnvCommonState[WrapperBDeviceType, WrapperBRNGType]
    ]:
        return self.world.reset(state, common_state)
    
    def step(
        self,
        state : WrapperStateType,
        common_state : FuncEnvCommonState[WrapperBDeviceType, WrapperBRNGType]
    ) -> Tuple[
        float, # elapsed time
        WrapperStateType,
        FuncEnvCommonState[WrapperBDeviceType, WrapperBRNGType]
    ]:
        return self.world.step(state, common_state)
    
    def close(
        self,
        state : WrapperStateType,
        common_state : FuncEnvCommonState[WrapperBDeviceType, WrapperBRNGType]
    ) -> None:
        return self.world.close(state, common_state)
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncWorld":
        return self.world.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> "FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType]":
        return self.world
    
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or self.world.has_wrapper_attr(name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.world.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        sub_world = self
        attr_set = False

        while attr_set is False and sub_world is not None:
            if hasattr(sub_world, name):
                setattr(sub_world, name, value)
                attr_set = True
            else:
                sub_world = sub_world.prev_wrapper_layer

        if attr_set is False and sub_world is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )