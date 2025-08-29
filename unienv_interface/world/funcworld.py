from typing import Optional, Any, Dict, Generic, TypeVar, Type, Tuple, Union
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from abc import ABC, abstractmethod
import time

WorldStateT = TypeVar("WorldStateT")

class FuncWorld(ABC, Generic[WorldStateT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
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
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self,
        state : WorldStateT
    ) -> Tuple[WorldStateT, Union[float, BArrayType]]:
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

    def close(
        self,
        state : WorldStateT
    ) -> None:
        pass

    # ========== Helper Methods ==========
    def is_control_timestep_compatible(self, control_timestep : Optional[float]) -> bool:
        if control_timestep is None or self.world_timestep is None:
            return True
        return (control_timestep % self.world_timestep) == 0
