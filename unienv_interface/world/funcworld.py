from typing import Optional, Any, Dict, Generic, TypeVar, Type, Tuple, Union
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from abc import ABC, abstractmethod
import time

WorldStateT = TypeVar("WorldStateT")

class FuncWorld(ABC, Generic[WorldStateT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """Functional counterpart to :class:`World` with explicit state passing."""
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType]

    """The world timestep in seconds. If None, the world is asynchronous (real-time)"""
    world_timestep : Optional[float]

    """The world's physical timestep in seconds. There might be multiple world sub-steps inside a world step. If none this means it is not known"""
    world_subtimestep : Optional[float] = None

    """The number of parallel environments in this world"""
    batch_size : Optional[int] = None

    @abstractmethod
    def initial(
        self,
        *,
        seed : Optional[int] = None,
        **kwargs
    ) -> WorldStateT:
        """Construct the initial world state."""
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self,
        state : WorldStateT
    ) -> Tuple[WorldStateT, Union[float, BArrayType]]:
        """Advance ``state`` by one world step."""
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        state : WorldStateT,
        *,
        seed: Optional[int] = None,
        mask: Optional[BArrayType] = None,
        **kwargs
    ) -> WorldStateT:
        """
        Perform reset on the selected environments with the given mask
        Note that the state input and output should be with the full batch size
        """
        raise NotImplementedError

    def reload(
        self,
        state : WorldStateT,
        *,
        seed : Optional[int] = None,
        **kwargs
    ) -> WorldStateT:
        """
        Prepare the world for reloading. This is called BEFORE nodes add entities.
        
        Use this hook to:
        - Clear old scene data
        - Load asset definitions
        - Prepare for entities to be added by nodes
        
        DO NOT compile the scene here. Compilation happens in `after_reload()` after
        all nodes have had a chance to add their entities.
        
        By default, this just calls `initial` with the same parameters.
        Simulation environments should override this to prepare for entity addition.
        
        The reload flow is:
            1. world.reload()       # World prepares (clear, load assets)
            2. node.reload()        # Nodes add entities to scene
            3. world.after_reload() # World compiles scene with all entities
            4. node.after_reload()  # Nodes cache references
        """
        return self.initial(seed=seed, **kwargs)

    def after_reset(
        self,
        state : WorldStateT,
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> WorldStateT:
        """
        Called after `reset()` completes.
        
        Use this hook to perform post-reset initialization.
        Returns the (possibly modified) world state.
        """
        return state

    def after_reload(
        self,
        state : WorldStateT,
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> WorldStateT:
        """
        Compile the world scene AFTER all nodes have added their entities.
        
        This is where the actual scene compilation should happen, since nodes
        have already added their entities during the `reload()` phase.
        
        Use this hook to:
        - Compile the simulation scene
        - Cache references to entities
        - Perform validation that requires the compiled scene
        - Initialize state that depends on the final scene configuration
        
        Returns the (possibly modified) world state.
        """
        return state

    def close(
        self,
        state : WorldStateT
    ) -> None:
        """Release any resources referenced by ``state``."""
        pass

    # ========== Helper Methods ==========
    def is_control_timestep_compatible(self, control_timestep : Optional[float]) -> bool:
        """Check whether a node control period aligns with ``world_timestep``."""
        if control_timestep is None or self.world_timestep is None:
            return True
        return (control_timestep % self.world_timestep) == 0
