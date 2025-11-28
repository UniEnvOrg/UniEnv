import abc
import os
import dataclasses
import multiprocessing as mp
from contextlib import nullcontext

from typing import Generic, TypeVar, Optional, Any, Dict, Union, Tuple, Sequence, Callable, Type
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space import Space
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.space.space_utils import serialization_utils as bsu
from unienv_interface.utils.symbol_util import get_class_from_full_name, get_full_class_name

from unienv_data.base import BatchBase, BatchT, IndexableType, SpaceStorage
import json

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
            assert backend.all(index >= -len_transitions) and backend.all(index < len_transitions), \
                f"Index values {index} are out of bounds for length {len_transitions}"
            nonzero_index = index + len_transitions
        data_index = (nonzero_index + offset) % len_transitions
        return data_index

class ReplayBuffer(BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    # =========== Class Attributes ==========
    @staticmethod
    def create(
        storage_cls : Type[SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        single_instance_space : Space[BatchT, BDeviceType, BDtypeType, BRNGType],
        *args,
        cache_path : Optional[Union[str, os.PathLike]] = None,
        capacity : Optional[int] = None,
        multiprocessing : bool = False,
        **kwargs
    ) -> "ReplayBuffer[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        storage_path_relative = "storage" + (storage_cls.single_file_ext or "")
        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)
        storage = storage_cls.create(
            single_instance_space,
            *args,
            cache_path=None if cache_path is None else os.path.join(cache_path, storage_path_relative),
            capacity=capacity,
            multiprocessing=multiprocessing,
            **kwargs
        )
        return ReplayBuffer(
            storage,
            storage_path_relative,
            0,
            0,
            cache_path=cache_path,
            multiprocessing=multiprocessing
        )
    
    @staticmethod
    def is_loadable_from(
        path : Union[str, os.PathLike]
    ) -> bool:
        if os.path.exists(os.path.join(path, "metadata.json")):
            with open(os.path.join(path, "metadata.json"), "r") as f:
                metadata = json.load(f)
            return metadata.get('type', None) == __class__.__name__
        return False

    @staticmethod
    def load_from(
        path : Union[str, os.PathLike],
        *,
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device: Optional[BDeviceType] = None,
        read_only : bool = True,
        multiprocessing : bool = False,
        **storage_kwargs
    ) -> "ReplayBuffer[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        assert metadata['type'] == __class__.__name__, f"Metadata type {metadata['type']} does not match expected type {__class__.__name__}"
        offset = int(metadata["offset"])
        count = int(metadata["count"])
        capacity = int(metadata["capacity"]) if "capacity" in metadata else None
        single_instance_space = bsu.json_to_space(
            metadata["single_instance_space"], backend, device
        )
        
        storage_cls : Type[SpaceStorage] = get_class_from_full_name(metadata["storage_cls"])
        storage_path = os.path.join(path, metadata["storage_path_relative"])

        storage = storage_cls.load_from(
            storage_path,
            single_instance_space,
            capacity=capacity,
            read_only=read_only,
            multiprocessing=multiprocessing,
            **storage_kwargs
        )
        return ReplayBuffer(
            storage,
            metadata["storage_path_relative"],
            count,
            offset,
            cache_path=path,
            multiprocessing=multiprocessing
        )

    # =========== Instance Attributes and Methods ==========
    def dumps(self, path : Union[str, os.PathLike]):
        with self._lock_scope():
            os.makedirs(path, exist_ok=True)
            storage_path = os.path.join(path, self.storage_path_relative)
            self.storage.dumps(storage_path)
            metadata = {
                "type": __class__.__name__,
                "count": self.count,
                "offset": self.offset,
                "capacity": self.storage.capacity,
                "storage_cls": get_full_class_name(type(self.storage)),
                "storage_path_relative": self.storage_path_relative,
                "single_instance_space": bsu.space_to_json(self.storage.single_instance_space),
            }
            with open(os.path.join(path, "metadata.json"), "w") as f:
                json.dump(metadata, f)

    def __init__(
        self,
        storage : SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType],
        storage_path_relative : str,
        count : int = 0,
        offset : int = 0,
        cache_path : Optional[Union[str, os.PathLike]] = None,
        multiprocessing : bool = False,
    ):
        self.storage = storage
        self._storage_path_relative = storage_path_relative
        self._cache_path = cache_path
        self._multiprocessing = multiprocessing
        if multiprocessing:
            assert storage.is_multiprocessing_safe, "Storage is not multiprocessing safe"
            self._lock = mp.Lock()
            self._count_value = mp.Value("q", int(count))
            self._offset_value = mp.Value("q", int(offset))
        else:
            self._lock = None
            self._count_value = int(count)
            self._offset_value = int(offset)
        
        super().__init__(
            storage.single_instance_space,
            None
        )

    def _lock_scope(self):
        if self._lock is not None:
            return self._lock
        else:
            return nullcontext()

    @property
    def cache_path(self) -> Optional[Union[str, os.PathLike]]:
        return self._cache_path

    @property
    def storage_path_relative(self) -> str:
        return self._storage_path_relative

    def __len__(self) -> int:
        return self.count

    @property
    def count(self) -> int:
        return self._count_value.value if self._multiprocessing else self._count_value

    @count.setter
    def count(self, value: int) -> None:
        if self._multiprocessing:
            self._count_value.value = int(value)
        else:
            self._count_value = int(value)
    
    @property
    def offset(self) -> int:
        return self._offset_value.value if self._multiprocessing else self._offset_value

    @offset.setter
    def offset(self, value: int) -> None:
        if self._multiprocessing:
            self._offset_value.value = int(value)
        else:
            self._offset_value = int(value)

    @property
    def capacity(self) -> Optional[int]:
        return self.storage.capacity

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.storage.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.storage.device

    @property
    def is_mutable(self) -> bool:
        return self.storage.is_mutable
    
    @property
    def is_multiprocessing_safe(self) -> bool:
        return self._multiprocessing

    def get_flattened_at(self, idx):
        return self.get_flattened_at_with_metadata(idx)[0]

    def get_flattened_at_with_metadata(self, idx: Union[IndexableType, BArrayType]) -> BArrayType:
        if hasattr(self.storage, "get_flattened"):
            with self._lock_scope():
                data = self.storage.get_flattened(idx)
            return data, None

        data, metadata = self.get_at_with_metadata(idx)
        if isinstance(idx, int):
            data = sfu.flatten_data(self.single_space, data)
        else:
            data = sfu.flatten_data(self._batched_space, data, start_dim=1)
        return data, metadata

    def get_at(self, idx):
        return self.get_at_with_metadata(idx)[0]

    def get_at_with_metadata(self, idx):
        with self._lock_scope():
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
        if hasattr(self.storage, "set_flattened"):
            with self._lock_scope():
                self.storage.set_flattened(idx, value)
            return

        if isinstance(idx, int):
            value = sfu.unflatten_data(self.single_space, value)
        else:
            value = sfu.unflatten_data(self._batched_space, value, start_dim=1)
        self.set_at(idx, value)

    def set_at(self, idx, value):
        with self._lock_scope():
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
        with self._lock_scope():
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
                self.storage.set(Ellipsis, sbu.get_at(self._batched_space, value, slice(-self.capacity, None)))
                self.count = self.capacity
                self.offset = 0
                return
            
            # Otherwise, perform round-robin writes
            indexes = (self.backend.arange(B, device=self.device) + self.offset + self.count) % self.capacity
            self.storage.set(indexes, value)
            outflow = max(0, self.count + B - self.capacity)
            if outflow > 0:
                self.offset = (self.offset + outflow) % self.capacity
            self.count = min(self.count + B, self.capacity)

    def clear(self):
        with self._lock_scope():
            self.count = 0
            self.offset = 0
            self.storage.clear()

    def close(self) -> None:
        self.storage.close()
