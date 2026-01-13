from importlib import metadata
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable, Type

from unienv_interface.space import Space
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils.symbol_util import *

from unienv_data.base import SpaceStorage, BatchT, IndexableType

import os
import shutil
from abc import abstractmethod

def batched_index_to_list(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    batched_index : Union[BArrayType, IndexableType],
    length : int
) -> List[int]:
    if isinstance(batched_index, slice):
        return list(range(*batched_index.indices(length)))
    elif batched_index is Ellipsis or batched_index is None:
        return list(range(length))
    else: # backend.is_backendarray
        assert backend.is_backendarray(batched_index)
        assert len(batched_index.shape) == 1
        if backend.dtype_is_boolean(batched_index.dtype):
            return [i for i in range(batched_index.shape[0]) if batched_index[i]]
        elif backend.dtype_is_real_integer(batched_index.dtype):
            return [batched_index[i] for i in range(batched_index.shape[0])]
        else:
            raise ValueError(f"Unsupported index type {type(batched_index)}")

class ListStorageBase(SpaceStorage[
    BatchT,
    BArrayType,
    BDeviceType,
    BDtypeType,
    BRNGType,
]):
    # ========== Instance Implementations ==========
    single_file_ext = None

    def __init__(
        self,
        single_instance_space: Space[Any, BDeviceType, BDtypeType, BRNGType],
        file_ext : str,
        cache_filename : Union[str, os.PathLike],
        mutable : bool = True,
        capacity : Optional[int] = None,
        length : int = 0,
    ):
        assert cache_filename is not None, "ListStorage requires a cache filename"
        super().__init__(single_instance_space)
        self._batched_single_space = sbu.batch_space(self.single_instance_space, 1)
        self.file_ext = file_ext
        self._cache_path = cache_filename
        self.is_mutable = mutable
        self.capacity = capacity
        self.length = length if capacity is None else capacity

    @property
    def cache_filename(self) -> Union[str, os.PathLike]:
        return self._cache_path

    @property
    def is_multiprocessing_safe(self) -> bool:
        return True
    
    def extend_length(self, length):
        assert self.capacity is None, "Cannot extend length of a fixed-capacity storage"
        self.length += length

    def shrink_length(self, length):
        assert self.is_mutable, "Cannot shrink length of a read-only storage"
        assert self.capacity is None, "Cannot shrink length of a fixed-capacity storage"
        from_len = self.length
        to_len = max(from_len - length, 0)
        all_files = os.listdir(self._cache_path)
        for i in range(to_len, from_len):
            if f"{i}.{self.file_ext}" in all_files:
                os.remove(os.path.join(self._cache_path, f"{i}.{self.file_ext}"))
        self.length = to_len
    
    def __len__(self):
        return self.length if self.capacity is None else self.capacity

    @abstractmethod
    def get_from_file(self, filename : str) -> BatchT:
        raise NotImplementedError

    @abstractmethod
    def set_to_file(self, filename : str, value : BatchT):
        raise NotImplementedError

    def get_single(self, index : int) -> BatchT:
        assert 0 <= index < self.length, f"Index {index} out of bounds for storage of length {self.length}"
        filename = os.path.join(self._cache_path, f"{index}.{self.file_ext}")
        return self.get_from_file(filename)

    def set_single(self, index : int, value : BArrayType):
        assert self.is_mutable, "Storage is not mutable"
        assert 0 <= index < self.length, f"Index {index} out of bounds for storage of length {self.length}"
        filename = os.path.join(self._cache_path, f"{index}.{self.file_ext}")
        self.set_to_file(filename, value)

    def get(self, index):
        if isinstance(index, int):
            result = self.get_single(index)
        else:
            result = sbu.concatenate(
                self._batched_single_space,    
                [
                    self.get_single(i) for i in batched_index_to_list(self.backend, index, len(self))
                ]
            )
        return result

    def set(self, index, value):
        assert self.is_mutable, "Storage is not mutable"
        if isinstance(index, int):
            self.set_single(index, value)
        else:
            indexes = batched_index_to_list(self.backend, index, len(self))
            for i, ind in enumerate(indexes):
                self.set_single(ind, sbu.get_at(self._batched_single_space, value, i))

    def clear(self):
        assert self.is_mutable, "Cannot clear a read-only storage"
        if self.capacity is None:
            self.length = 0
        shutil.rmtree(self._cache_path)
        os.makedirs(self._cache_path, exist_ok=True)

    def close(self):
        pass
