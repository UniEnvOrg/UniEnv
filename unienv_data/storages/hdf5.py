from importlib import metadata
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable, Type, Literal, Mapping, Callable

from unienv_interface.space import Space, BoxSpace, DictSpace, TextSpace, BinarySpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.utils.symbol_util import *

from unienv_data.base import SpaceStorage, BatchT

import h5py
import numpy as np
import os
import json

HDF5BatchType = Union[Dict[str, Any], NumpyArrayType, str]
HDF5SpaceType = Union[DictSpace, BoxSpace, TextSpace]
class HDF5Storage(SpaceStorage[
    HDF5BatchType,
    NumpyArrayType,
    NumpyDeviceType,
    NumpyDtypeType,
    NumpyRNGType,
]):
    # ========== Class Attributes ==========
    single_file_ext : Optional[str] = ".hdf5"
    DEFAULT_KEY : str = "data"

    @staticmethod
    def _check_hdf5_file(
        root : Union[h5py.Group, h5py.Dataset],
        single_instance_space : HDF5SpaceType,
        capacity : Optional[int],
    ):
        assert single_instance_space.backend == NumpyComputeBackend, \
            f"Expected NumpyComputeBackend, got {single_instance_space.backend}"

        if not isinstance(single_instance_space, DictSpace) and isinstance(root, h5py.Group):
            assert __class__.DEFAULT_KEY in root, \
                f"Expected key '{__class__.DEFAULT_KEY}' in group, got keys {list(root.keys())}"
            root = root[__class__.DEFAULT_KEY]

        if isinstance(single_instance_space, BoxSpace) or isinstance(single_instance_space, BinarySpace):
            assert isinstance(root, h5py.Dataset), \
                f"Expected h5py.Dataset for BoxSpace / BinarySpace, got {type(root)}"
            assert root.maxshape[0] == capacity, \
                f"Expected maxshape[0] to be {capacity}, got {root.maxshape[0]}"
            assert root.shape[1:] == single_instance_space.shape, \
                f"Expected shape[1:] to be {single_instance_space.shape}, got {root.shape[1:]}"
            assert root.dtype == (single_instance_space.dtype or NumpyComputeBackend.default_boolean_dtype), \
                f"Expected dtype {(single_instance_space.dtype or NumpyComputeBackend.default_boolean_dtype)}, got {root.dtype}"
        elif isinstance(single_instance_space, TextSpace):
            assert isinstance(root, h5py.Dataset), \
                f"Expected h5py.Dataset for TextSpace, got {type(root)}"
            assert root.maxshape[0] == capacity, \
                f"Expected maxshape[0] to be {capacity}, got {root.maxshape[0]}"
            
            assert root.dtype.kind == "O" or root.dtype.kind == "S", \
                f"Expected dtype 'O' or 'S' for TextSpace, got {root.dtype}"
            if root.dtype.kind == "S":
                assert root.dtype.itemsize >= single_instance_space.max_length, \
                    f"Expected itemsize to be {single_instance_space.max_length}, got {root.dtype.itemsize}"
        elif isinstance(single_instance_space, DictSpace):
            assert isinstance(root, h5py.Group), \
                f"Expected h5py.Group for DictSpace, got {type(root)}"
            
            for key, space in single_instance_space.spaces.items():
                assert key in root, f"Key '{key}' not found in group"
                sub_root = root[key]
                __class__._check_hdf5_file(
                    sub_root,
                    space,
                    capacity
                )
    
    @staticmethod
    def _construct_hdf5_file(
        root : h5py.Group,
        single_instance_space : HDF5SpaceType,
        capacity : Optional[int] = None,
        initial_capacity : Optional[int] = None,
        compression : Union[
            Dict[str, Any],
            Optional[str]
        ] = None, # 'gzip', 'lzf', etc.
        compression_level : Union[
            Dict[str, Any],
            Optional[int]
        ] = None, # 0-9 for gzip
        chunks : Union[
            Dict[str, Any],
            Optional[Union[bool, Tuple[int, ...]]]
        ] = None,
    ) -> None:
        assert not (initial_capacity is None and capacity is None), \
            "If `capacity` is None, `initial_capacity` must be provided"
        assert capacity is None or initial_capacity is None or initial_capacity == capacity, \
            "If `capacity` is provided, `initial_capacity` must be equal to `capacity`"

        if not isinstance(single_instance_space, DictSpace):
            return __class__._construct_hdf5_file(
                root,
                DictSpace(
                    single_instance_space.backend,
                    {
                        __class__.DEFAULT_KEY: single_instance_space
                    },
                    device=single_instance_space.device,
                ),
                capacity=capacity,
                initial_capacity=initial_capacity,
            )

        initial_capacity = initial_capacity or capacity
        for key, space in single_instance_space.spaces.items():
            assert key not in root, f"Key '{key}' already exists in group"
            if not isinstance(space, DictSpace):
                if isinstance(space, BoxSpace) or isinstance(space, BinarySpace):
                    shape = (initial_capacity, *space.shape)
                    maxshape = (capacity, *space.shape)
                    dtype = space.dtype or NumpyComputeBackend.default_boolean_dtype
                elif isinstance(space, TextSpace):
                    shape = (initial_capacity,)
                    maxshape = (capacity,) if capacity is not None else (None,)
                    dtype = h5py.string_dtype(encoding='utf-8', length=space.max_length)
                else:
                    raise ValueError(f"Unsupported space type: {type(space)}")
            
                root.create_dataset(
                    key,
                    shape=shape,
                    maxshape=maxshape,
                    dtype=dtype,
                    compression=compression if not isinstance(compression, Mapping) else compression.get(key, None),
                    compression_opts=compression_level if not isinstance(compression_level, Mapping) else compression_level.get(
                        key, None
                    ),
                    chunks=chunks if not isinstance(chunks, Mapping) else chunks.get(key, None),
                )
            else:
                sub_group = root.create_group(key)
                __class__._construct_hdf5_file(
                    sub_group,
                    space,
                    capacity=capacity,
                    initial_capacity=initial_capacity,
                    compression=compression if not isinstance(compression, Mapping) else compression.get(
                        key, None
                    ),
                    compression_level=compression_level if not isinstance(compression_level, Mapping) else compression_level.get(
                        key, None
                    ),
                    chunks=chunks if not isinstance(chunks, Mapping) else chunks.get(key, None),
                )

    @staticmethod
    def call_function_on_first_dataset(
        root : h5py.Group,
        function : Callable[[h5py.Dataset], Any],
    ) -> Any:
        groups = []
        for key, item in root.items():
            if isinstance(item, h5py.Dataset):
                return function(item)
            elif isinstance(item, h5py.Group):
                groups.append(item)
        for group in groups:
            try:
                return __class__.call_function_on_first_dataset(group, function)
            except ValueError:
                continue
        raise ValueError("No dataset found in the HDF5 group")

    @staticmethod
    def call_function_on_every_dataset(
        root : h5py.Group,
        function : Callable[[h5py.Dataset], None],
    ) -> None:
        for key, item in root.items():
            if isinstance(item, h5py.Dataset):
                function(item)
            elif isinstance(item, h5py.Group):
                __class__.call_function_on_every_dataset(item, function)

    @staticmethod
    def get_from(
        root : Union[h5py.Group, h5py.Dataset],
        single_instance_space : HDF5SpaceType,
        index : Union[int, slice, Sequence[int], BArrayType],
    ) -> Any:
        if not isinstance(single_instance_space, DictSpace) and isinstance(root, h5py.Group):
            assert __class__.DEFAULT_KEY in root, \
                f"Expected key '{__class__.DEFAULT_KEY}' in group, got keys {list(root.keys())}"
            root = root[__class__.DEFAULT_KEY]
        
        if isinstance(single_instance_space, DictSpace):
            result = {}
            for key, space in single_instance_space.spaces.items():
                sub_root = root[key]
                result[key] = __class__.get_from(sub_root, space, index)
        else:
            result = root[index]
        return result

    @staticmethod
    def set_to(
        root : Union[h5py.Group, h5py.Dataset],
        single_instance_space : HDF5SpaceType,
        index : Union[int, slice, Sequence[int], BArrayType],
        value : HDF5BatchType,
    ) -> None:
        if not isinstance(single_instance_space, DictSpace) and isinstance(root, h5py.Group):
            assert __class__.DEFAULT_KEY in root, \
                f"Expected key '{__class__.DEFAULT_KEY}' in group, got keys {list(root.keys())}"
            root = root[__class__.DEFAULT_KEY]
        
        if isinstance(single_instance_space, DictSpace):
            for key, space in single_instance_space.spaces.items():
                sub_root = root[key]
                __class__.set_to(sub_root, space, index, value[key])
        else:
            root[index] = value

    @classmethod
    def create(
        cls, 
        single_instance_space, 
        capacity, 
        cache_path = None, 
        initial_capacity : Optional[int] = None,
        compression : Union[
            Dict[str, Any],
            Optional[str]
        ] = None, # 'gzip', 'lzf', etc.
        compression_level : Union[
            Dict[str, Any],
            Optional[int]
        ] = None, # 0-9 for gzip
        chunks : Union[
            Dict[str, Any],
            Optional[Union[bool, Tuple[int, ...]]]
        ] = None,
        **kwargs
    ) -> "HDF5Storage":
        assert cache_path is not None, \
            "cache_path must be provided for HDF5Storage"
        root = h5py.File(
            cache_path,
            "w"
        )
        __class__._construct_hdf5_file(
            root,
            single_instance_space,
            capacity=capacity,
            initial_capacity=initial_capacity,
            compression=compression,
            compression_level=compression_level,
            chunks=chunks,
        )
        return cls(
            single_instance_space,
            root,
            capacity=capacity,
        )
    
    @classmethod
    def load_from(
        cls, 
        path, 
        single_instance_space, 
        *, 
        capacity = None, 
    ) -> "HDF5Storage":
        assert os.path.exists(path), \
            f"Path {path} does not exist"
        
        assert os.access(path, os.R_OK), \
            f"Path {path} is not readable"
    
        # Check file permissions
        can_write = os.access(path, os.W_OK)

        root = h5py.File(
            path,
            "r+" if can_write else "r"
        )
        return cls(
            single_instance_space,
            root,
            capacity=capacity
        )

    # ========== Instance Methods ==========

    def __init__(
        self,
        single_instance_space : HDF5SpaceType,
        root : h5py.Group,
        capacity : Optional[int] = None,
    ):
        __class__._check_hdf5_file(
            root,
            single_instance_space,
            capacity
        )
        super().__init__(
            single_instance_space
        )
        self.root = root
        self.capacity = capacity
        self._len = self.call_function_on_first_dataset(
            root,
            lambda dataset: dataset.shape[0]
        )
        assert self.capacity is None or self._len == self.capacity, \
            f"If the storage has a fixed capacity, the length must match the capacity. Expected {self.capacity}, got {self._len}"
    
    def extend_length(self, length):
        assert self.capacity is None, \
            "Cannot extend length of a storage with fixed capacity"
        assert length > 0, "Length must be greater than 0"
        new_length = self._len + length
        __class__.call_function_on_every_dataset(
            self.root,
            lambda dataset: dataset.resize(new_length, axis=0)
        )
        self._len = new_length
    
    def shrink_length(self, length):
        assert self.capacity is None, \
            "Cannot shrink length of a storage with fixed capacity"
        assert length > 0, "Length must be greater than 0"
        new_length = self._len - length
        assert new_length >= 0, "New length must be non-negative"
        __class__.call_function_on_every_dataset(
            self.root,
            lambda dataset: dataset.resize(new_length, axis=0)
        )
        self._len = new_length
    
    def __len__(self):
        return self._len
    
    def get(self, index):
        return __class__.get_from(
            self.root,
            self.single_instance_space,
            index
        )

    def set(self, index, value):
        return __class__.set_to(
            self.root,
            self.single_instance_space,
            index,
            value
        )

    def dumps(self, path):
        if isinstance(self.root, h5py.File) and os.path.samefile(self.root.filename, path):
            self.root.flush()
        else:
            target_file = h5py.File(path, 'w')
            target_file.copy(
                self.root,
                target_file,
            )
            target_file.flush()
            target_file.close()

    def close(self):
        if isinstance(self.root, h5py.File):
            self.root.close()
        self.root = None