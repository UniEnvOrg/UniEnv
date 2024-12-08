import abc
import os
import dataclasses
from typing import Generic, TypeVar, Optional, Any, Dict, Union, Tuple, Sequence, Callable
from unienv_interface.space import Space, Box, flatten_utils as sfu, batch_utils as sbu
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from .common import BatchBase, BatchT
import json

class TensorStorage(abc.ABC, Generic[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    save_ext : str = ".storage"

    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType] = None
    dtype : BDtypeType
    capacity : Optional[int] = None
    
    def __init__(
        self,
        single_instance_shape : Tuple[int, ...],
        default_value : Optional[BArrayType] = None,
    ):
        self.single_instance_shape = single_instance_shape
        self._default_value = default_value

    @property
    def default_value(self) -> BArrayType:
        return self._default_value or self.backend.array_api_namespace.zeros(
            self.single_instance_shape,
            dtype=self.dtype,
            device=self.device
        )

    @default_value.setter
    def default_value(self, value : Optional[BArrayType]):
        if self.backend.array_api_namespace.all(value == 0):
            self._default_value = None
        else:
            self._default_value = value
    
    def extend_length(self, length : int) -> None:
        """
        This is used by capacity = None storages to extend the length of the storage
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, index : Union[int, slice, BArrayType, None]) -> BArrayType:
        raise NotImplementedError
    
    @abc.abstractmethod
    def set(self, index : Union[int, slice, BArrayType, None], value : BArrayType) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def dumps(self, path : Union[str, os.PathLike]) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def loads(self, path : Union[str, os.PathLike]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()

def index_with_offset(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    index : Union[int, slice, BArrayType, None],
    len_transitions : int,
    capacity : Optional[int],
    offset : int
) -> Union[int, BArrayType]:
    """
    Helpful function to convert replay buffer indices to data indices in the TensorStorage
    """
    if capacity is None:
        assert offset == 0, "Offset must be 0 for unbounded storage"
        capacity = len_transitions
    if index is None:
        nonzero_index = backend.array_api_namespace.arange(len_transitions)
        data_index = (nonzero_index + offset) % capacity
        return data_index
    elif isinstance(index, int):
        assert index < len_transitions and index >= -len_transitions, f"Index {index} is out of bounds for length {len_transitions}"
        nonzero_index = (index + len_transitions) % len_transitions
        data_index = (nonzero_index + offset) % capacity
        return data_index
    elif isinstance(index, slice):
        nonzero_index = backend.array_api_namespace.asarray(range(*index.indices(len_transitions)))
        data_index = (nonzero_index + offset) % capacity
        return data_index
    else:
        assert len(index.shape) == 1, f"Index shape {index.shape} is not 1D"
        if index.shape == (len_transitions, ) and backend.dtype_is_boolean(
            index.dtype
        ):
            # Boolean mask, rotate the mask by offset
            rotated_mask = backend.array_api_namespace.roll(
                index,
                shift=offset,
                axis=0
            )
            return rotated_mask
        else:
            assert backend.array_api_namespace.min(index) >= -len_transitions and backend.array_api_namespace.max(index) < len_transitions, f"Index {index} is out of bounds for length {len_transitions}"
            assert backend.dtype_is_real_integer(index.dtype), f"Index dtype {index.dtype} is not an integer"
            nonzero_index = (index + len_transitions) % len_transitions
            data_index = (nonzero_index + offset) % capacity
            return data_index

class ReplayBuffer(BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    is_mutable = True

    def __init__(
        self,
        storage : TensorStorage[BArrayType, BDeviceType, BDtypeType, BRNGType],
        single_space : Space[Any, Any, BDeviceType, BDtypeType, BRNGType],
    ):
        self.storage = storage
        self.count = 0
        self.offset = 0
        assert storage.single_instance_shape == sfu.flatten_space(single_space).shape, "Storage shape must match single instance shape"
        super().__init__(single_space.to_device(storage.device))

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

    def get_flattened_at(self, idx: Optional[Union[int, slice, BArrayType]] = None) -> BArrayType:
        return self.storage.get(index_with_offset(
            self.backend,
            idx,
            self.count,
            self.capacity,
            self.offset
        ))
    
    def set_flattened_at(self, idx: Optional[Union[int, slice, BArrayType]], value: BArrayType) -> None:
        self.storage.set(index_with_offset(
            self.backend,
            idx,
            self.count,
            self.capacity,
            self.offset
        ), value)

    def extend_flattened(
        self,
        value: BArrayType
    ):
        b = value.shape[0]
        if b == 0:
            return

        start_storage_idx = self.offset + self.count
        if self.capacity is None:
            self.storage.set(slice(start_storage_idx, start_storage_idx + b), value)
            self.count = self.count + b
            return
        
        start_storage_idx = start_storage_idx % self.capacity
        if b >= self.capacity:
            self.storage.set(None, value[-self.capacity:])
            self.count = self.capacity
            self.offset = 0
            return
        
        remaining_capacity = self.capacity - start_storage_idx
        if b <= remaining_capacity:
            self.storage.set(slice(start_storage_idx, start_storage_idx + b), value)
            outflow = max(0, self.count + b - self.capacity)
            if outflow > 0:
                self.offset = (self.offset + outflow) % self.capacity
            self.count = min(self.count + b, self.capacity)
            return
        else:
            self.storage.set(slice(start_storage_idx, self.capacity), value[:remaining_capacity])
            self.count = min(self.count + remaining_capacity, self.capacity)
            self.offset = 0
            self.extend_flattened(value[remaining_capacity:])
            return
    
    def clear(self):
        self.count = 0
        self.offset = 0
        self.storage.clear()

    def dumps(self, path : Union[str, os.PathLike]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.storage.dumps(os.path.join(path, f"storage{self.storage.save_ext}"))
        metadata = {
            "count": self.count,
            "offset": self.offset
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    
    def loads(self, path : Union[str, os.PathLike]):
        self.storage.loads(os.path.join(path, f"storage{self.storage.save_ext}"))
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
            self.count = metadata["count"]
            self.offset = metadata["offset"]


    def close(self) -> None:
        self.storage.close()

    def __del__(self) -> None:
        self.close()