from typing import Optional, Any, Dict, Generic, TypeVar, Type, Tuple
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from abc import ABC, abstractmethod

class World(ABC, Generic[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType]

    """The world timestep in seconds, if None, the world is asynchronous (real-time)"""
    world_timestep : Optional[float]

    @abstractmethod
    def step(self) -> float:
        """
        Step the world by one timestep.
        Returns:
            float: The elapsed time since the last step.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
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

WrapperArrayT = TypeVar("WrapperArrayT")
WrapperDeviceT = TypeVar("WrapperDeviceT")
WrapperDtypeT = TypeVar("WrapperDtypeT")
WrapperRNGT = TypeVar("WrapperRNGT")

class WorldWrapper(
    Generic[
        WrapperArrayT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT,
        BArrayType, BDeviceType, BDtypeType, BRNGType
    ],
    World[WrapperArrayT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]
):
    def __init__(
        self,
        world : World[BArrayType, BDeviceType, BDtypeType, BRNGType]
    ):
        self.world = world

    @property
    def world_timestep(self) -> Optional[float]:
        return self.world.world_timestep
    
    @world_timestep.setter
    def world_timestep(self, value : Optional[float]) -> None:
        self.world.world_timestep = value
    
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
