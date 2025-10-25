import abc
import os
from typing import Generic, TypeVar, Optional, Any, Dict, Union, Tuple, Sequence, Callable, Type
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space import Space
from .common import BatchBase, BatchT, IndexableType

class SpaceStorage(abc.ABC, Generic[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """
    SpaceStorage is an abstract base class for storages that hold instances of a specific space.
    It provides a common interface for creating, loading, and managing the storage of instances of a given space.
    Note that if you want your space storage to support multiprocessing, you need to check / implement `__getstate__` and `__setstate__` methods to ensure that the storage can be pickled and unpickled correctly.
    """
    # ========== Class Creation and Loading Methods ==========
    @classmethod
    def create(
        cls,
        single_instance_space : Space[BatchT, BDeviceType, BDtypeType, BRNGType],
        *args,
        capacity : Optional[int],
        cache_path : Optional[Union[str, os.PathLike]] = None,
        **kwargs
    ) -> "SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        raise NotImplementedError

    @classmethod
    def load_from(
        cls,
        path: Union[str, os.PathLike],
        single_instance_space: Space[BatchT, BDeviceType, BDtypeType, BRNGType],
        *,
        capacity : Optional[int] = None,
        read_only : bool = True,
        **kwargs
    ) -> "SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        raise NotImplementedError

    # ========== Class Attributes ==========
    
    """
    The file extension (e.g. `.pt`) used for saving a single instance of the space.
    If this is None, it means the storage stores files in a folder
    """
    single_file_ext : Optional[str] = None

    # ======== Instance Attributes ==========
    """
    The total capacity (number of single instances) of the storage.
    If None, the storage has unlimited capacity.
    """
    capacity : Optional[int] = None

    """
    The cache path for the storage.
    If None, the storage will not use caching.
    """
    cache_filename : Optional[Union[str, os.PathLike]] = None

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.single_instance_space.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.single_instance_space.device

    def __init__(
        self,
        single_instance_space : Space[BatchT, BDeviceType, BDtypeType, BRNGType],
    ):
        self.single_instance_space = single_instance_space
    
    def extend_length(self, length : int) -> None:
        """
        This is used by capacity = None storages to extend the length of the storage
        If this is called on a storage with a fixed capacity, we will simply ignore the call.
        """
        pass

    def shrink_length(self, length : int) -> None:
        """
        This is used by capacity = None storages to shrink the length of the storage
        If this is called on a storage with a fixed capacity, we will simply ignore the call.
        """
        pass

    def __len__(self) -> int:
        """
        Returns the number of instances in the storage
        Storages with unlimited capacity should implement this method to return the current length of the storage.
        """
        if self.capacity is None:
            raise NotImplementedError(f"__len__ is not implemented for class {type(self).__name__}")
        return self.capacity

    # We don't define them here, since they are optional and the `ReplayBuffer` checks if they are implemented
    # by using hasattr(self, "get_flattened") and hasattr(self, "set_flattened").
    # def get_flattened(self, index : Union[IndexableType, BArrayType]) -> BArrayType:
    #     raise NotImplementedError
    
    # def set_flattened(self, index : Union[IndexableType, BArrayType], value : BArrayType) -> None:
    #     raise NotImplementedError

    @abc.abstractmethod
    def get(self, index : Union[IndexableType, BArrayType]) -> BatchT:
        raise NotImplementedError
    
    @abc.abstractmethod
    def set(self, index : Union[IndexableType, BArrayType], value : BatchT) -> None:
        raise NotImplementedError
    
    def clear(self) -> None:
        """
        Clear all data inside the storage and set the length to 0 if the storage has unlimited capacity.
        For storages with fixed capacity, this should reset the storage to its initial state.
        """
        if self.capacity is None:
            self.shrink_length(len(self))

    @abc.abstractmethod
    def dumps(self, path : Union[str, os.PathLike]) -> None:
        """
        Dumps the storage to the specified path.
        This is used for storages that have a single file extension (e.g. `.pt` for PyTorch).
        """
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()