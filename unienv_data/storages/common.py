import os
from unienv_data.base import *
from unienv_interface.space import Space
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable
import h5py
import numpy as np

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
        if self.capacity is not None and length > self.capacity:
            raise IndexError("Index out of range")
        if len_storage >= length:
            return
        self.data.extend([None] * (length - len_storage))

    def _get_data_at(self, index_positive : int) -> BArrayType:
        len_storage = len(self)
        if index_positive >= len_storage:
            return self.default_value

        dat = self.data[index_positive]
        if dat is None:
            return self.default_value
        else:
            return dat

    def get(self, index : Union[int, slice, BArrayType, None]) -> BArrayType:
        len_storage = len(self)
        if index is None:
            if self.capacity is None:
                indices = range(len_storage)
            else:
                indices = range(self.capacity)
        elif isinstance(index, int):
            return self._get_data_at(index)
        elif isinstance(index, slice):
            if self.capacity is None:
                indices = range(*index.indices(len_storage))
            else:
                indices = range(*index.indices(self.capacity))
            return indices
        elif isinstance(index, BArrayType):
            if self.backend.dtype_is_boolean(
                index.dtype
            ):
                if self.capacity is not None:
                    assert index.shape == (self.capacity, ), f"Expected boolean mask shape to match capacity {(self.capacity, )}, got {index.shape}"
                else:
                    assert index.shape == (len_storage, ), f"Expected boolean mask shape to match length {(len_storage, )}, got {index.shape}"
                
                indices = self.backend.array_api_namespace.nonzero(index)[0]
            else:
                indices = index
            indices[indices < 0] += len_storage
            assert (
                self.capacity is None and self.backend.array_api_namespace.all(
                    (indices >= 0) & (indices < len_storage)
                )
            ) or (
                self.capacity is not None and self.backend.array_api_namespace.all(
                    (indices >= 0) & (indices < self.capacity)
                )
            ), "Index out of range"

        to_stack = [
            self._get_data_at(index_positive) for index_positive in indices
        ]
        return self.backend.array_api_namespace.stack(
            to_stack, axis=0
        )

    def set(self, index : Union[int, slice, BArrayType, None], value : BArrayType):
        len_storage = len(self)
        if index is None:
            b_value = value.shape[0]
            if self.capacity is not None:
                assert b_value <= self.capacity, f"Expected batch size {b_value} to be less than capacity {self.capacity}"
            self.data = [
                value[i] for i in range(b_value)
            ]
            return
        elif isinstance(index, int):
            if index < 0:
                index += len_storage
            if index < 0 or (self.capacity is not None and index >= self.capacity):
                raise IndexError("Index out of range")
            indices = [index]
            value = [value]
        elif isinstance(index, slice):
            if self.capacity is None:
                start = 0 if index.start is None else index.start
                stop = index.stop
                step = 1 if index.step is None else index.step

                indices = range(start, stop, step)
            else:
                indices = range(*index.indices(self.capacity))
            return indices
        elif isinstance(index, BArrayType):
            if self.backend.dtype_is_boolean(
                index.dtype
            ):
                if self.capacity is not None:
                    assert index.shape == (self.capacity, ), f"Expected boolean mask shape to match capacity {(self.capacity, )}, got {index.shape}"
                else:
                    assert index.shape == (len_storage, ), f"Expected boolean mask shape to match length {(len_storage, )}, got {index.shape}"
                
                indices = self.backend.array_api_namespace.nonzero(index)[0]
            else:
                indices = index
            if self.capacity is not None:
                indices[indices < 0] += self.capacity
                assert self.backend.array_api_namespace.all(
                    (indices >= 0) & (indices < self.capacity)
                ), "Index out of range"
            else:
                indices[indices < 0] += len_storage
                assert self.backend.array_api_namespace.all(
                    (indices >= 0)
                ), "Index out of range"

        max_idx = max(indices)
        self._extend_to_length(max_idx + 1)
        for i, list_index in enumerate(indices):
            self.data[list_index] = value[i]

    def dumps(self, path: Union[str, os.PathLike]) -> None:
        with h5py.File(path, 'w') as f:
            actual_data = [
                (i, dat) for i, dat in enumerate(self.data) if dat is not None
            ]
            dtype = self.backend.to_numpy(self.default_value).dtype
            datset = f.create_dataset(
                'data',
                data=np.stack([self.backend.to_numpy(
                    dat
                ) for _, dat in actual_data], axis=0),
            )
            dataset_idx = f.create_dataset(
                'indices',
                data=np.array([i for i, _ in actual_data], dtype=np.int64)
            )
            dataset_idx.attrs['length'] = len(self)
    
    def loads(self, path: Union[str, os.PathLike]) -> None:
        with h5py.File(path, 'r') as f:
            data = f['data']
            indices = f['indices']
            self.data = [None] * indices.attrs['length']
            for i, dat in zip(indices, data):
                self.data[i] = self.backend.from_numpy(dat)
