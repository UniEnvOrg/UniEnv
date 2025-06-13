import os
from unienv_data.base import *
from unienv_interface.space import Space
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend
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

    @classmethod
    def create(
        cls,
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device : Optional[BDeviceType],
        dtype : BDtypeType,
        single_instance_shape : Tuple[int, ...],
        *,
        capacity : Optional[int],
        default_value : Optional[BArrayType] = None,
    ) -> "ListStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        storage = ListStorage(
            backend,
            device,
            dtype,
            single_instance_shape,
            default_value=default_value,
            capacity=capacity,
        )
        return storage

    @classmethod
    def load_from(
        cls,
        path: Union[str, os.PathLike],
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device : Optional[BDeviceType],
        dtype : BDtypeType,
        single_instance_shape : Tuple[int, ...],
        *,
        capacity : Optional[int],
        default_value : Optional[BArrayType] = None,
    ) -> "ListStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        storage = cls.create(
            backend,
            device,
            dtype,
            single_instance_shape,
            capacity=capacity,
            default_value=default_value
        )
        with h5py.File(path, 'r') as f:
            data = f['data']
            indices = f['indices']
            storage.data = [None] * indices.attrs['length']
            for i, dat in zip(indices, data):
                storage.data[i] = backend.from_numpy(dat, dtype=dtype, device=device)

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

    def __len__(self):
        return len(self.data)
    
    def extend_length(self, length : int):
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
        elif self.backend.is_backendarray(index):
            if self.backend.dtype_is_boolean(
                index.dtype
            ):
                if self.capacity is not None:
                    assert index.shape == (self.capacity, ), f"Expected boolean mask shape to match capacity {(self.capacity, )}, got {index.shape}"
                else:
                    assert index.shape == (len_storage, ), f"Expected boolean mask shape to match length {(len_storage, )}, got {index.shape}"
                
                indices = self.backend.nonzero(index)[0]
            else:
                indices = index
            indices = self.backend.at(indices)[indices < 0].add(len_storage)
            assert (
                self.capacity is None and self.backend.all(
                    (indices >= 0) & (indices < len_storage)
                )
            ) or (
                self.capacity is not None and self.backend.all(
                    (indices >= 0) & (indices < self.capacity)
                )
            ), "Index out of range"

        to_stack = [
            self._get_data_at(index_positive) for index_positive in indices
        ]
        return self.backend.stack(
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
                start = 0 if index.start is None else (
                    index.start if index.start >= 0 else index.start + len_storage
                )
                stop = len_storage if index.stop is None else (
                    index.stop if index.stop >= 0 else index.stop + len_storage
                )
                step = 1 if index.step is None else index.step

                indices = range(start, stop, step)
            else:
                indices = range(*index.indices(self.capacity))
        elif self.backend.is_backendarray(index):
            if self.backend.dtype_is_boolean(
                index.dtype
            ):
                if self.capacity is not None:
                    assert index.shape == (self.capacity, ), f"Expected boolean mask shape to match capacity {(self.capacity, )}, got {index.shape}"
                else:
                    assert index.shape == (len_storage, ), f"Expected boolean mask shape to match length {(len_storage, )}, got {index.shape}"
                
                indices = self.backend.nonzero(index)[0]
            else:
                indices = index
            if self.capacity is not None:
                indices[indices < 0] += self.capacity
                assert self.backend.all(
                    (indices >= 0) & (indices < self.capacity)
                ), "Index out of range"
            else:
                indices[indices < 0] += len_storage
                assert self.backend.all(
                    (indices >= 0)
                ), "Index out of range"

        max_idx = max(indices)
        self.extend_length(max_idx + 1)
        for i, list_index in enumerate(indices):
            self.data[list_index] = value[i]

    def clear(self) -> None:
        self.data = []


class HDF5Storage(TensorStorage[
    NumpyComputeBackend.ARRAY_TYPE, 
    NumpyComputeBackend.DEVICE_TYPE, 
    NumpyComputeBackend.DTYPE_TYPE, 
    NumpyComputeBackend.RNG_TYPE,
]):
    backend = NumpyComputeBackend
    device = None
    save_ext : str = ".hdf5"

    def __init__(
        self,
        dtype : NumpyComputeBackend.DTYPE_TYPE,
        single_instance_shape : Tuple[int, ...],
        capacity : Optional[int], # None can only be used when mmap_load is True
        file_location : str,
        load : bool = False,
        default_value : Optional[NumpyComputeBackend.ARRAY_TYPE] = None,
    ):
        super().__init__(
            single_instance_shape=single_instance_shape,
            default_value=default_value
        )

        assert capacity is not None and capacity > 0, "Capacity must be a positive integer"
        
        self.dtype = dtype
        self.capacity = capacity

        if os.path.exists(file_location) and load:    
            self.file = h5py.File(file_location, 'r+')
            assert 'data' in self.file, "No data found in file"
            self.data = self.file['data']
            assert self.data.shape[1:] == single_instance_shape, "Mismatched shape"
            assert capacity is None or self.data.shape[0] == capacity, "Capacity mismatch"
            capacity = capacity or self.data.shape[0]
        else:
            if os.path.exists(file_location):
                os.remove(file_location)
            self.file = h5py.File(file_location, 'w')
            self.data = self.file.create_dataset(
                'data',
                shape=(capacity, *single_instance_shape),
                dtype=self.dtype,
            )

        if self._default_value is not None:
            self._default_value = self._default_value.astype(self.dtype)

    @classmethod
    def create(
        cls,
        backend : NumpyComputeBackend,
        device : Optional[NumpyComputeBackend.DEVICE_TYPE],
        dtype : NumpyComputeBackend.DTYPE_TYPE,
        single_instance_shape : Tuple[int, ...],
        path : Union[str, os.PathLike],
        *,
        capacity : int,
        default_value : Optional[NumpyComputeBackend.ARRAY_TYPE] = None,
    ) -> "HDF5Storage":
        assert backend is NumpyComputeBackend, "Only NumpyComputeBackend is supported"
        storage = HDF5Storage(
            dtype,
            single_instance_shape,
            capacity,
            path,
            load=False,
            default_value=default_value
        )
        return storage

    @classmethod
    def load_from(
        cls,
        path : Union[str, os.PathLike],
        backend : NumpyComputeBackend,
        device : Optional[NumpyComputeBackend.DEVICE_TYPE],
        dtype : NumpyComputeBackend.DTYPE_TYPE,
        single_instance_shape : Tuple[int, ...],
        *,
        capacity : int,
        default_value : Optional[NumpyComputeBackend.ARRAY_TYPE] = None,
    ) -> "HDF5Storage":
        assert backend is NumpyComputeBackend, "Only NumpyComputeBackend is supported"
        storage = HDF5Storage(
            dtype,
            single_instance_shape,
            capacity,
            path,
            load=True,
            default_value=default_value
        )
        return storage

    def get(self, index : Union[int, slice, np.ndarray, None]) -> np.ndarray:
        if index is None:
            return self.data
        return self.data[index]

    def set(self, index : Union[int, slice, np.ndarray, None], value : np.ndarray) -> None:
        if index is None:
            self.data[:] = value
        else:
            self.data[index] = value

    def clear(self) -> None:
        if self._default_value is None:
            self.data[:] = 0
        else:
            self.data[:] = self._default_value[np.newaxis]
    
    def dumps(self, path: Union[str, os.PathLike]) -> None:
        if os.path.exists(path) and os.path.samefile(self.file.filename, path):
            self.file.flush()     
        else:
            target_file = h5py.File(path, 'w')
            for key in self.file:
                self.file.copy(key, target_file)
    
    def loads(self, path: Union[str, os.PathLike]) -> None:
        assert os.path.exists(path), "File does not exist"
        if os.path.samefile(self.file.filename, path):
            return
        
        source_file = h5py.File(path, 'r')
        assert 'data' in source_file, "No data found in file"
        source_file.copy('data', self.file)
        self.data = self.file['data']
        source_file.close()
    
    def close(self):
        self.file.close()