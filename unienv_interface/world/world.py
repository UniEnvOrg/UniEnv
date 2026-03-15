from typing import Optional, Any, Dict, Generic, TypeVar, Type, Tuple, Union
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from abc import ABC, abstractmethod
import time

class World(ABC, Generic[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """Mutable world/simulator interface used by node-based environments.

    A ``World`` owns the shared simulation state while ``WorldNode`` instances
    contribute observations, actions, rewards, and reset/reload logic around it.
    """
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType]

    """The world timestep in seconds, if None, the world is asynchronous (real-time)"""
    world_timestep : Optional[float]

    """The world's physical timestep in seconds, there might be multiple world sub-steps inside a world step. If none this means it is not known"""
    world_subtimestep : Optional[float] = None

    """The number of parallel environments in this world"""
    batch_size : Optional[int] = None

    @abstractmethod
    def step(self) -> Union[float, BArrayType]:
        """
        Step the world by one timestep.
        Returns:
            float: The elapsed time since the last step.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> None:
        """Reset the world state in-place."""
        raise NotImplementedError

    def reload(
        self,
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> None:
        """
        Prepare the world for reloading. This is called BEFORE nodes add entities.
        
        Use this hook to:
        - Clear old scene data
        - Load asset definitions
        - Prepare for entities to be added by nodes
        
        DO NOT compile the scene here. Compilation happens in `after_reload()` after
        all nodes have had a chance to add their entities.
        
        By default, this just calls `reset` with the same parameters.
        Simulation environments should override this to prepare for entity addition.
        
        The reload flow is:
            1. world.reload()       # World prepares (clear, load assets)
            2. node.reload()        # Nodes add entities to scene
            3. world.after_reload() # World compiles scene with all entities
            4. node.after_reload()  # Nodes cache references
        """
        self.reset(seed=seed, mask=mask, **kwargs)

    def after_reset(
        self,
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> None:
        """
        Called after `reset()` completes.
        
        Use this hook to perform post-reset initialization that requires
        the simulation to be in a valid state (e.g., caching entity indices,
        validating scene state).
        
        This is called by the environment composer after all nodes have been reset.
        """
        pass

    def after_reload(
        self,
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> None:
        """
        Compile the world scene AFTER all nodes have added their entities.
        
        This is where the actual scene compilation should happen, since nodes
        have already added their entities during the `reload()` phase.
        
        Use this hook to:
        - Compile the simulation scene
        - Cache references to entities
        - Perform validation that requires the compiled scene
        - Initialize state that depends on the final scene configuration
        
        This is called by the environment composer after all nodes have been reloaded.
        """
        pass

    def close(self) -> None:
        """Release any resources owned by the world."""
        pass

    # ========== Helper Methods ==========
    def is_control_timestep_compatible(self, control_timestep : Optional[float]) -> bool:
        """Check whether a node control period aligns with ``world_timestep``."""
        if control_timestep is None or self.world_timestep is None:
            return True
        return (control_timestep % self.world_timestep) == 0

    # ========== Wrapper methods ==========
    # @property
    # def unwrapped(self) -> "World":
    #     return self
    
    # @property
    # def prev_wrapper_layer(self) -> Optional["World"]:
    #     return None
    
    # def has_wrapper_attr(self, name: str) -> bool:
    #     """Checks if the attribute `name` exists in the environment."""
    #     return hasattr(self, name)
    
    # def get_wrapper_attr(self, name: str) -> Any:
    #     """Gets the attribute `name` from the environment."""
    #     return getattr(self, name)
    
    # def set_wrapper_attr(self, name: str, value: Any):
    #     """Sets the attribute `name` on the environment with `value`."""
    #     setattr(self, name, value)

# WrapperArrayT = TypeVar("WrapperArrayT")
# WrapperDeviceT = TypeVar("WrapperDeviceT")
# WrapperDtypeT = TypeVar("WrapperDtypeT")
# WrapperRNGT = TypeVar("WrapperRNGT")

# class WorldWrapper(
#     Generic[
#         WrapperArrayT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT,
#         BArrayType, BDeviceType, BDtypeType, BRNGType
#     ],
#     World[WrapperArrayT, WrapperDeviceT, WrapperDtypeT, WrapperRNGT]
# ):
#     def __init__(
#         self,
#         world : World[BArrayType, BDeviceType, BDtypeType, BRNGType]
#     ):
#         self.world = world

#     @property
#     def world_timestep(self) -> Optional[float]:
#         return self.world.world_timestep
    
#     @world_timestep.setter
#     def world_timestep(self, value : Optional[float]) -> None:
#         self.world.world_timestep = value
    
#     def step(self):
#         return self.world.step()
    
#     def reset(self):
#         return self.world.reset()
    
#     # ========== Wrapper methods ==========
#     @property
#     def unwrapped(self) -> "World":
#         return self.world.unwrapped
    
#     @property
#     def prev_wrapper_layer(self) -> "World[BDeviceType, BDtypeType, BRNGType]":
#         return self.world
    
#     def has_wrapper_attr(self, name: str) -> bool:
#         return hasattr(self, name) or self.world.has_wrapper_attr(name)
    
#     def get_wrapper_attr(self, name: str) -> Any:
#         if hasattr(self, name):
#             return getattr(self, name)
#         else:
#             try:
#                 return self.world.get_wrapper_attr(name)
#             except AttributeError as e:
#                 raise AttributeError(
#                     f"wrapper {type(self).__name__} has no attribute {name!r}"
#                 ) from e

#     def set_wrapper_attr(self, name: str, value: Any):
#         sub_world = self
#         attr_set = False

#         while attr_set is False and sub_world is not None:
#             if hasattr(sub_world, name):
#                 setattr(sub_world, name, value)
#                 attr_set = True
#             else:
#                 sub_world = sub_world.prev_wrapper_layer

#         if attr_set is False and sub_world is None:
#             raise AttributeError(
#                 f"wrapper {type(self).__name__} has no attribute {name!r}"
#             )

class RealWorld(World[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """
    ``World`` implementation backed by wall-clock elapsed time.
    """

    def __init__(
        self,
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device: Optional[BDeviceType] = None,
        world_timestep: Optional[float] = None,
        world_subtimestep: Optional[float] = None,
        batch_size: Optional[int] = None
    ):
        self.backend = backend
        self.device = device
        self.world_timestep = world_timestep
        self.world_subtimestep = world_subtimestep
        self.batch_size = batch_size
        self._last_step_time : Optional[None] = None

    def step(self) -> float:
        """Return the wall-clock delta since the previous step."""
        assert self._last_step_time is not None, "World must be reset before stepping."
        current_time = time.monotonic()
        elapsed_time = current_time - self._last_step_time
        self._last_step_time = current_time
        return elapsed_time

    def reset(self) -> None:
        """Start a new wall-clock timing sequence."""
        self._last_step_time = time.monotonic()
