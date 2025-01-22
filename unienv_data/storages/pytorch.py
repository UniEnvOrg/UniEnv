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
        data : Union[torch.Tensor, MemoryMappedTensor],
        default_value : Optional[PyTorchComputeBackend.ARRAY_TYPE] = None,
    ):
        super().__init__(
            single_instance_shape=data.shape[1:],
            default_value=default_value
        )
        self.data = data

    @property
    def device(self) -> PyTorchComputeBackend.DEVICE_TYPE:
        return self.data.device

    @property
    def dtype(self) -> PyTorchComputeBackend.DTYPE_TYPE:
        return self.data.dtype
    
    @property
    def capacity(self) -> int:
        return self.data.shape[0]

    @property
    def is_memmap(self) -> bool:
        return isinstance(self.data, MemoryMappedTensor)

    @classmethod
    def create(
        cls,
        backend : PyTorchComputeBackend,
        device : Optional[PyTorchComputeBackend.DEVICE_TYPE],
        dtype : PyTorchComputeBackend.DTYPE_TYPE,
        single_instance_shape : Tuple[int, ...],
        *,
        memmap : bool = False,
        memmap_path : Optional[str] = None,
        memmap_existok : bool = True,
        capacity : Optional[int],
        default_value : Optional[PyTorchComputeBackend.ARRAY_TYPE] = None,
    ) -> "PytorchTensorStorage":
        assert backend is PyTorchComputeBackend, "PytorchTensorStorage only supports PyTorch backend"
        assert capacity is not None, "Capacity must be specified when creating a new tensor"
        
        target_shape = (capacity, *single_instance_shape)
        if memmap:
            real_device = None if device is None else torch.device(device)
            assert real_device is None or real_device.type == 'cpu', "Memory mapping is only supported for CPU tensors"
            assert memmap_path is not None, "Memory-mapped file path must be specified (and should be the dumps path)"
            
            data = MemoryMappedTensor.zeros(
                target_shape,
                dtype=dtype,
                device=device,
                filename=memmap_path,
                existsok=memmap_existok
            )
        else:
            data = torch.zeros(
                target_shape,
                dtype=dtype,
                device=device
            )
        
        if default_value is not None:
            if device is not None:
                default_value = default_value.to(device)
            
            data.copy_(default_value.unsqueeze(0).expand(target_shape))
        
        return PytorchTensorStorage(data, default_value=default_value)

    @classmethod
    def load_from(
        cls,
        path: Union[str, os.PathLike],
        backend : PyTorchComputeBackend,
        device : Optional[PyTorchComputeBackend.DEVICE_TYPE],
        dtype : PyTorchComputeBackend.DTYPE_TYPE,
        single_instance_shape : Tuple[int, ...],
        *,
        memmap : bool = False,
        capacity : Optional[int],
        default_value : Optional[PyTorchComputeBackend.ARRAY_TYPE] = None,
    ) -> "PytorchTensorStorage":
        assert backend is PyTorchComputeBackend, "PytorchTensorStorage only supports PyTorch backend"
        assert capacity is not None, "Capacity must be specified when creating a new tensor"
        assert os.path.exists(path), "File does not exist"
        
        target_shape = (capacity, *single_instance_shape)
        target_data = MemoryMappedTensor.from_filename(
            path,
            dtype=dtype,
            shape=target_shape
        )
        
        if memmap:
            real_device = None if device is None else torch.device(device)
            assert real_device is None or real_device.type == 'cpu', "Memory mapping is only supported for CPU tensors"
            data = target_data
        else:
            data = torch.zeros(
                (capacity, *single_instance_shape),
                dtype=dtype,
                device=device
            )
            data.copy_(target_data)
        
        return PytorchTensorStorage(data, default_value=default_value)

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
            if self.is_memmap and os.path.samefile(self.data.filename, path):
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
