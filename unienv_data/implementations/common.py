from unienv_data.base import *
from unienv_interface.space import Space
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List

class ListStorage(TensorStorage[
    BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    save_ext : str = ".lststorage"

    def __init__(
        self,
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device : Optional[BDeviceType],
        dtype : BDtypeType,
        single_instance_shape : Tuple[int, ...],
        default_value : Optional[BArrayType] = None,
        capacity : Optional[int] = None,
    ):
        super().__init__(
            single_instance_shape=single_instance_shape,
            default_value=default_value
        )
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.capacity = capacity
        self.data : List[Optional[BArrayType]] = []

    def __len__(self):
        return len(self.data)
    
    def _extend_to_length(self, length : int):
        len_storage = len(self)
        if self.capacity is not None:
            length = min(length, self.capacity)
        if len_storage >= length:
            return
        self.data.extend([None] * (length - len_storage))
    
    def _get_data_at(self, index : int) -> BArrayType:
        if self.capacity is not None and index >= self.capacity:
            raise IndexError("Index out of range")
        if index < 0:
            raise IndexError("Index out of range")
        if index >= len(self):
            return self.default_value
        else:
            dat = self.data[index]
            if dat is None:
                return self.default_value
            else:
                return dat

    def get(self, index : Union[int, slice, BArrayType, None]) -> BArrayType:
        to_stack = []
        if isinstance(index, int):
            return self._get_data_at(index)
        elif isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            to_stack = [self._get_data_at(i) for i in indices]
        elif isinstance(index, BArrayType):
            len_storage = len(self)
            if index.shape == (len_storage, ) and self.backend.dtype_is_boolean(
                index.dtype
            ):
                to_stack = [self._get_data_at(i) for i in range(len_storage) if index[i]]
            else:
                to_stack = [self._get_data_at(i) for i in index]
        else:
            to_stack = self.data
        return self.backend.array_api_namespace.stack(
            to_stack, axis=0
        )

    def set(self, index : Union[int, slice, BArrayType, None], value : BArrayType):
        if isinstance(index, int):
            self._extend_to_length(index + 1)
            self.data[index] = value
            return
        len_storage = len(self)
        if isinstance(index, slice):
            list_idx_iter = range(*index.indices(len_storage))
        elif isinstance(index, BArrayType):
            if index.shape == (len_storage, ) and self.backend.dtype_is_boolean(
                index.dtype
            ):
                list_idx_iter = [i for i in range(len_storage) if index[i]]
            else:
                list_idx_iter = index
        else:
            list_idx_iter = range(len_storage)
        
        max_idx = max(list_idx_iter)
        self._extend_to_length(max_idx + 1)
        for i, list_index in enumerate(list_idx_iter):
            self.data[list_index] = value[i]
    
    def extend(
        self,
        value : BArrayType
    ) -> None:
        assert value.shape[1:] == self.single_instance_shape
        if self.capacity is None:
            for i in range(value.shape[0]):
                self.data.append(value[i])
            self.next_cursor += value.shape[0]
        else:
            len_storage = len(self)
            if len_storage >= self.capacity:
                super().extend(value)
            else:
                remaining_capacity = self.capacity - len_storage
                extend_length = min(remaining_capacity, value.shape[0])
                for i in range(extend_length):
                    self.data.append(value[i])
                self.next_cursor = (self.next_cursor + extend_length) % self.capacity
                remaining_extend_length = value.shape[0] - extend_length
                if remaining_extend_length > 0:
                    super().extend(value[extend_length:])