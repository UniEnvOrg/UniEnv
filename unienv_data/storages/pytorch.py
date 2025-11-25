import os
import torch
from unienv_interface.space import Space, BoxSpace
from unienv_interface.backends import ComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType
from unienv_data.base import SpaceStorage
from tensordict.memmap import MemoryMappedTensor
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Type

class PytorchTensorStorage(SpaceStorage[
    PyTorchArrayType,
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
]):
    @classmethod
    def create(
        cls,
        single_instance_space : BoxSpace[PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType],
        *,
        capacity : Optional[int],
        is_memmap : bool = False,
        cache_path : Optional[str] = None,
        multiprocessing : bool = False,
        memmap_existok : bool = True,
    ) -> "PytorchTensorStorage":
        assert single_instance_space.backend is PyTorchComputeBackend, \
            f"Single instance space must be of type PyTorchComputeBackend, got {single_instance_space.backend}"
        assert isinstance(single_instance_space, BoxSpace), \
            f"Single instance space must be a BoxSpace, got {type(single_instance_space)}"
        assert capacity is not None, "Capacity must be specified when creating a new tensor"

        target_shape = (capacity, *single_instance_space.shape)
        if is_memmap:
            real_device = None if single_instance_space.device is None else torch.device(single_instance_space.device)
            assert real_device is None or real_device.type == 'cpu', "Memory mapping is only supported for CPU tensors"
            assert cache_path is not None, "Memory-mapped file path (`cache_path`) must be specified (and should be the dumps path)"
            
            # Ensure the directory exists for the memory-mapped file
            parent_dir = os.path.dirname(cache_path)
            os.makedirs(parent_dir, exist_ok=True)

            data = MemoryMappedTensor.empty(
                target_shape,
                dtype=single_instance_space.dtype,
                device=single_instance_space.device,
                filename=cache_path,
                existsok=memmap_existok
            )
        else:
            data = torch.zeros(
                target_shape,
                dtype=single_instance_space.dtype,
                device=single_instance_space.device
            )
            if multiprocessing:
                data = data.share_memory_()

        return PytorchTensorStorage(single_instance_space, data, mutable=True)

    @classmethod
    def load_from(
        cls,
        path: Union[str, os.PathLike],
        single_instance_space: BoxSpace[PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType],
        *,
        is_memmap : bool = False,
        capacity : Optional[int] = None,
        read_only : bool = True,
        multiprocessing : bool = False,
    ) -> "PytorchTensorStorage":
        assert single_instance_space.backend is PyTorchComputeBackend, "PytorchTensorStorage only supports PyTorch backend"
        assert capacity is not None, "Capacity must be specified when creating a new tensor"
        assert os.path.exists(path), "File does not exist"

        if is_memmap and not read_only:
            assert os.access(path, os.W_OK), "File is not writable, cannot open in read-write mode"

        target_shape = (capacity, *single_instance_space.shape)
        target_data = MemoryMappedTensor.from_filename(
            path,
            dtype=single_instance_space.dtype,
            shape=target_shape
        )

        if is_memmap:
            real_device = None if single_instance_space.device is None else torch.device(single_instance_space.device)
            assert real_device is None or real_device.type == 'cpu', "Memory mapping is only supported for CPU tensors"
            data = target_data
        else:
            data = torch.empty(
                target_shape,
                dtype=single_instance_space.dtype,
                device=single_instance_space.device
            )
            if multiprocessing:
                data = data.share_memory_()
            data = data.copy_(target_data)

        return PytorchTensorStorage(
            single_instance_space,
            data,
            mutable=not read_only
        )

    # ========== Instance Implementations ==========

    backend = PyTorchComputeBackend
    single_file_ext : str = ".memmap"

    def __init__(
        self,
        single_instance_space : BoxSpace[PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType],
        data : Union[torch.Tensor, MemoryMappedTensor],
        mutable : bool = True,
    ):
        assert single_instance_space.shape == data.shape[1:], \
            f"Single instance space shape {single_instance_space.shape} does not match data shape {data.shape[1:]}"
        super().__init__(
            single_instance_space
        )
        self.data = data
        self._mutable = mutable

    @property
    def device(self) -> Optional[PyTorchDeviceType]:
        return self.single_instance_space.device

    @property
    def cache_filename(self) -> Optional[Union[str, os.PathLike]]:
        if isinstance(self.data, MemoryMappedTensor):
            return self.data.filename
        return None
    
    @property
    def is_mutable(self) -> bool:
        return self._mutable

    @property
    def is_multiprocessing_safe(self) -> bool:
        return self.data.is_shared()
    
    @property
    def capacity(self) -> int:
        return self.data.shape[0]

    @property
    def is_memmap(self) -> bool:
        return isinstance(self.data, MemoryMappedTensor)

    def get(self, index : Union[int, slice, torch.Tensor]) -> torch.Tensor:
        return self.data[index]

    def set(self, index : Union[int, slice, torch.Tensor], value : torch.Tensor) -> None:
        assert self.is_mutable, "Storage is not mutable"
        self.data[index] = value

    def clear(self) -> None:
        assert self.is_mutable, "Storage is not mutable"
        pass
    
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

    def close(self):
        self.data = None