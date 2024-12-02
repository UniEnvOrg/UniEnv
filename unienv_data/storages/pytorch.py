import os
import torch
from unienv_data.base import *
from unienv_interface.space import Space
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.backends.base import ComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from tensordict.memmap import MemoryMappedTensor
from torchrl.data.replay_buffers import LazyMemmapStorage
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List

class PytorchTensorStorage(TensorStorage[
    PyTorchComputeBackend.ARRAY_TYPE, 
    PyTorchComputeBackend.DEVICE_TYPE, 
    PyTorchComputeBackend.DTYPE_TYPE, 
    PyTorchComputeBackend.RNG_TYPE,
]):
    save_ext : str = ".memmap"

    def __init__(
        self,
        device : Optional[PyTorchComputeBackend.DEVICE_TYPE],
        dtype : PyTorchComputeBackend.DTYPE_TYPE,
        single_instance_shape : Tuple[int, ...],
        default_value : Optional[PyTorchComputeBackend.ARRAY_TYPE] = None,
        capacity : int = None,
        use_mmap : bool = False,
        mmap_location : Optional[str] = None,
        mmap_existsok : bool = False
    ):
        super().__init__(
            single_instance_shape=single_instance_shape,
            default_value=default_value
        )

        assert capacity is not None and capacity > 0, "Capacity must be a positive integer"
        self.backend = PyTorchComputeBackend
        self.device = device
        self.dtype = dtype
        self.capacity = capacity
        if use_mmap:
            assert device is None or device.type == 'cpu', "Memory mapping is only supported for CPU tensors"
            self.data = MemoryMappedTensor.zeros(
                (capacity, *single_instance_shape),
                dtype=dtype,
                device=device,
                filename=mmap_location,
                existsok=mmap_existsok
            )
            if self._default_value is not None:
                self._default_value = self._default_value.to(device) if device is not None else self._default_value.cpu()
        else:
            self.data = torch.zeros(
                (capacity, *single_instance_shape),
                dtype=dtype,
                device=device
            )
            if device is not None and self._default_value is not None:
                self._default_value = self._default_value.to(device)
        # Fill storage with default value
        if self.default_value is not None:
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
    
    def dumps(self, path: Union[str, os.PathLike]) -> None:
        if self.memap:
            data : MemoryMappedTensor = self.data
            if os.path.samefile(data.filename, path):
                return
        if os.path.exists(path):
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
        self.data = MemoryMappedTensor.from_filename(
            shape=self.data.shape,
            filename=path,
            dtype=self.data.dtype,
        )