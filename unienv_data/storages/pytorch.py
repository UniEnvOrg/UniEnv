import os
import torch
from unienv_data.base import *
from unienv_interface.space import Space
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.backends.base import ComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from tensordict.memmap import MemoryMappedTensor
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List

class PytorchTensorStorage(TensorStorage[
    PyTorchComputeBackend.ARRAY_TYPE, 
    PyTorchComputeBackend.DEVICE_TYPE, 
    PyTorchComputeBackend.DTYPE_TYPE, 
    PyTorchComputeBackend.RNG_TYPE,
]):
    backend = PyTorchComputeBackend
    save_ext : str = ".memmap"

    def __init__(
        self,
        device : Optional[PyTorchComputeBackend.DEVICE_TYPE],
        dtype : PyTorchComputeBackend.DTYPE_TYPE,
        single_instance_shape : Tuple[int, ...],
        capacity : Optional[int], # None can only be used when mmap_load is True
        default_value : Optional[PyTorchComputeBackend.ARRAY_TYPE] = None,
        use_mmap : bool = False,
        mmap_location : Optional[str] = None,
        mmap_load : bool = False,
    ):
        super().__init__(
            single_instance_shape=single_instance_shape,
            default_value=default_value
        )

        device = None if device is None else torch.device(device)
        self.device = device
        self.dtype = dtype
        self.capacity = capacity
        if use_mmap:
            assert device is None or device.type == 'cpu', "Memory mapping is only supported for CPU tensors"
            if mmap_load:
                assert os.path.exists(mmap_location), "Memory-mapped file does not exist"
                self.data = MemoryMappedTensor.from_filename(
                    mmap_location,
                    dtype=dtype,
                    shape=(capacity, *single_instance_shape),
                )
                assert self.data.shape[1:] == single_instance_shape, "Mismatched shape"
                if self.data.shape[0] != capacity:
                    assert capacity is None, "Capacity mismatch"
                    capacity = self.data.shape[0]
            else:
                assert capacity is not None, "Capacity must be specified when creating a new memory-mapped tensor"
                self.data = MemoryMappedTensor.zeros(
                    (capacity, *single_instance_shape),
                    dtype=dtype,
                    device=device,
                    filename=mmap_location,
                    existsok=True
                )
        else:
            assert capacity is not None, "Capacity must be specified when creating a new tensor"
            self.data = torch.zeros(
                (capacity, *single_instance_shape),
                dtype=dtype,
                device=device
            )
        
        if self._default_value is not None:
            if device is not None:
                self._default_value = self._default_value.to(device)
            self._default_value = PyTorchComputeBackend.array_api_namespace.astype(self._default_value, dtype)
        
        self.capacity = capacity
        # Fill storage with default value
        if self.default_value is not None and not mmap_load:
            self.data.copy_(self.default_value.expand((self.capacity, *self.single_instance_shape)))
        self.memap = use_mmap

    def get(self, index : Union[int, slice, torch.Tensor, None]) -> torch.Tensor:
        if index is None:
            return self.data
        return self.data[index]

    def set(self, index : Union[int, slice, torch.Tensor, None], value : torch.Tensor) -> None:
        if index is None:
            self.data[:] = value
        else:
            self.data[index] = value

    def clear(self) -> None:
        if self._default_value is None:
            self.data.fill_(0)
        else:
            self.data[:] = self._default_value.unsqueeze(0)
    
    def dumps(self, path: Union[str, os.PathLike]) -> None:
        if os.path.exists(path):
            if self.memap and os.path.samefile(self.data.filename, path):
                return
            MemoryMappedTensor.from_filename(
                shape=self.data.shape,
                filename=path,
                dtype=self.data.dtype,
            ).copy_(self.data)
        else:
            parent_dir = os.path.dirname(path)
            os.makedirs(parent_dir, exist_ok=True)
            MemoryMappedTensor.from_tensor(
                self.data,
                filename=path,
                copy_existing=True,
                copy_data=True,
            )
    
    def loads(self, path: Union[str, os.PathLike]) -> None:
        if os.path.exists(path) and self.memap and os.path.samefile(self.data.filename, path):
            return
        
        data_memap = MemoryMappedTensor.from_filename(
            shape=self.data.shape,
            filename=path,
            dtype=self.data.dtype,
        )
        self.data.copy_(data_memap)