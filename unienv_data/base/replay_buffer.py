import abc
import os
import dataclasses
from typing import Generic, TypeVar, Optional, Any, Dict, Union, Tuple, Sequence, Callable, Type
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space import Space
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.space.space_utils import serialization_utils as bsu
from unienv_interface.utils.symbol_util import get_class_from_full_name, get_full_class_name
from .common import BatchBase, BatchT, IndexableType
import json
import pickle

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
        cache_path : Optional[Union[str, os.PathLike]] = None,
        capacity : Optional[int],
        default_value : Optional[Any] = None,
        **kwargs
    ) -> "SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        raise NotImplementedError

    @classmethod
    def load_from(
        cls,
        path: Union[str, os.PathLike],
        single_instance_space: Space[BatchT, BDeviceType, BDtypeType, BRNGType],
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
        default_value : Optional[Any] = None,
    ):
        self.single_instance_space = single_instance_space
        self._default_value = default_value
    
    @property
    def default_value(self) -> Optional[Any]:
        return self._default_value

    @default_value.setter
    def default_value(self, value : Optional[Any]):
        assert value is None or isinstance(value, (int, float)) or self.single_instance_space.contains(value), \
            f"Default value {value} must be None or of type int, float or match the single instance space {self.single_instance_space}"
        self._default_value = value
    
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

    @abc.abstractmethod
    def get(self, index : Union[IndexableType, BArrayType]) -> BArrayType:
        raise NotImplementedError
    
    @abc.abstractmethod
    def set(self, index : Union[IndexableType, BArrayType], value : BArrayType) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def clear(self) -> None:
        """
        Clear all data inside the storage and set the length to 0 if the storage has unlimited capacity.
        For storages with fixed capacity, this should reset the storage to its initial state.
        """
        raise NotImplementedError

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

def index_with_offset(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    index : Union[int, slice, BArrayType],
    len_transitions : int,
    offset : int,
    device : Optional[BDeviceType] = None,
) -> Union[int, BArrayType]:
    """
    Helpful function to convert round robin indices to data indices in the SpaceStorage
    Returns:
        - data_index: The index in the storage that corresponds to the given index with the specified offset.
    """
    if index is Ellipsis:
        nonzero_index = backend.arange(len_transitions, device=device)
        data_index = (nonzero_index + offset) % len_transitions
        return data_index
    elif isinstance(index, int):
        assert -len_transitions <= index < len_transitions, f"Index {index} is out of bounds for length {len_transitions}"
        return (index + len_transitions + offset) % len_transitions
    elif isinstance(index, slice):
        nonzero_index = backend.arange(*index.indices(len_transitions), device=device)
        data_index = (nonzero_index + offset) % len_transitions
        return data_index
    else:
        assert len(index.shape) == 1, f"Index shape {index.shape} is not 1D"
        if index.shape == (len_transitions, ) and backend.dtype_is_boolean(
            index.dtype
        ):
            # Boolean mask, rotate the mask by offset
            nonzero_index = backend.nonzero(index)[0]
        else:
            assert backend.dtype_is_real_integer(index.dtype), f"Index dtype {index.dtype} is not an integer"
            nonzero_index = index + len_transitions
        data_index = (nonzero_index + offset) % len_transitions
        return data_index

class ReplayBuffer(BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    is_mutable = True
    # =========== Class Attributes ==========
    @staticmethod
    def create(
        storage : SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType],
    ) -> "ReplayBuffer[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        return ReplayBuffer(
            storage, 
            "storage" + (storage.single_file_ext or ""),
            0, 
            0
        )
    
    @staticmethod
    def is_loadable_from(
        path : Union[str, os.PathLike]
    ) -> bool:
        return os.path.exists(os.path.join(path, "metadata.json"))

    @staticmethod
    def load_from(
        path : Union[str, os.PathLike],
        *,
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType] = None,
        device: Optional[BDeviceType] = None,
        **storage_kwargs
    ) -> "ReplayBuffer[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        assert metadata['type'] == __class__.__name__, f"Metadata type {metadata['type']} does not match expected type {__class__.__name__}"
        offset = int(metadata["offset"])
        count = int(metadata["count"])
        single_instance_space = bsu.json_to_space(metadata["single_instance_space"], backend, device)
        
        storage_cls : Type[SpaceStorage] = get_class_from_full_name(metadata["storage_cls"])
        storage_path = os.path.join(path, metadata["storage_path_relative"])

        storage = storage_cls.load_from(
            storage_path,
            single_instance_space,
            **storage_kwargs
        )
        return ReplayBuffer(storage, metadata["storage_path_relative"], count, offset)

    def dumps(self, path : Union[str, os.PathLike]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        storage_path = os.path.join(path, self.storage_path_relative)
        self.storage.dumps(storage_path)
        metadata = {
            "type": __class__.__name__,
            "count": self.count,
            "offset": self.offset,
            "storage_cls": get_full_class_name(type(self.storage)),
            "storage_path_relative": self.storage_path_relative,
            "single_instance_space": bsu.space_to_json(self.storage.single_instance_space),
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    # =========== Instance Attributes and Methods ==========
    def __init__(
        self,
        storage : SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType],
        storage_path_relative : Union[str, os.PathLike],
        count : int = 0,
        offset : int = 0,
    ):
        self.storage = storage
        self.count = count
        self.offset = offset
        self.storage_path_relative = storage_path_relative
        super().__init__(
            storage.single_instance_space,
            None
        )

    def __len__(self) -> int:
        return self.count
    
    @property
    def capacity(self) -> Optional[int]:
        return self.storage.capacity

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.storage.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.storage.device

    def get_flattened_at(self, idx):
        return self.get_flattened_at_with_metadata(idx)[0]

    def get_flattened_at_with_metadata(self, idx: Union[IndexableType, BArrayType]) -> BArrayType:
        data, metadata = self.get_at_with_metadata(idx)
        if isinstance(idx, int):
            data = sfu.flatten_data(self.single_space, data)
        else:
            data = sfu.flatten_data(self._batched_space, data, start_dim=1)
        return data, metadata

    def get_at(self, idx):
        return self.get_at_with_metadata(idx)[0]

    def get_at_with_metadata(self, idx):
        data_index = index_with_offset(
            self.backend,
            idx,
            self.count,
            self.offset,
            self.device
        )
        data = self.storage.get(data_index)
        return data, None
    
    def set_flattened_at(self, idx: Union[IndexableType, BArrayType], value: BArrayType) -> None:
        if isinstance(idx, int):
            value = sfu.unflatten_data(self.single_space, value)
        else:
            value = sfu.unflatten_data(self._batched_space, value, start_dim=1)
        self.set_at(idx, value)

    def set_at(self, idx, value):
        self.storage.set(index_with_offset(
            self.backend,
            idx,
            self.count,
            self.offset,
            self.device
        ), value)

    def extend_flattened(
        self,
        value: BArrayType
    ):
        unflattened_data = sfu.unflatten_data(self._batched_space, value, start_dim=1)
        self.extend(unflattened_data)
        
    def extend(self, value):
        B = sbu.batch_size_data(value)
        if B == 0:
            return
        if self.capacity is None:
            assert self.offset == 0, "Offset must be 0 when capacity is None"
            self.storage.extend_length(B)
            self.storage.set(slice(self.count, self.count + B), value)
            self.count += B
            return
        
        # We have a fixed capacity, only keep the last `capacity` elements
        if B >= self.capacity:
            self.storage.set(None, sbu.get_at(self._batched_space, value, slice(-self.capacity)))
            self.count = self.capacity
            self.offset = 0
            return
        
        # Otherwise, perform round-robin writes
        indexes = (self.backend.arange(B, device=self.device) + self.offset) % self.capacity
        self.storage.set(indexes, value)
        outflow = max(0, self.count + B - self.capacity)
        if outflow > 0:
            self.offset = (self.offset + outflow) % self.capacity
        self.count = min(self.count + B, self.capacity)

    def clear(self):
        self.count = 0
        self.offset = 0
        self.storage.clear()

    def close(self) -> None:
        self.storage.close()
