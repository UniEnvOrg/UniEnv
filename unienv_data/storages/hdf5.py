from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable, Type, Literal, Mapping, Callable

from unienv_interface.space import Space, BoxSpace, DictSpace, TextSpace, BinarySpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.utils.symbol_util import *

from unienv_data.base import SpaceStorage, BatchT
from unienv_data.replay_buffer import ReplayBuffer

import h5py
import numpy as np
import os
import json
import copy

def is_fancy_index(
    index : Any
) -> bool:
    """
    Check if the given index is fancy indexing.
    Args:
        index (Any): Index to check
    Returns:
        bool: True if fancy indexing, False otherwise
    """
    if not isinstance(index, np.ndarray):
        return False
    if index.dtype.kind not in {'i', 'u'} or index.shape == (1,):
        return False
    return True

def fancy_indexing_to_supported_indexing(
    index : np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert fancy indexing to limited fancy indexing that h5py supports.
    Args:
        index (np.ndarray): Fancy indexing array of shape (N,)
    Returns:
        np.ndarray: Limited fancy indexing array of shape (M,)
        np.ndarray: Indices to reconstruct the resulting data to original order from the fetched data, shape (N,)
    """
    unique_indices, unique_reverse_index = np.unique(index, return_inverse=True)
    unique_indices_sorted_ids = np.argsort(unique_indices)
    # Construct reverse mapping from index `unique_indices[unique_indices_sorted_ids]` to original index
    # Create inverse mapping: for each original unique index position, find its new sorted position
    inverse_mapping = np.empty_like(unique_indices_sorted_ids)
    inverse_mapping[unique_indices_sorted_ids] = np.arange(len(unique_indices_sorted_ids))
    # Apply the inverse mapping to the reverse index to get sorted reverse index
    sorted_unique_reverse_index = inverse_mapping[unique_reverse_index]
    return unique_indices[unique_indices_sorted_ids], sorted_unique_reverse_index

def fancy_indexing_to_supported_set_indexing(
    index : np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert fancy indexing to limited fancy indexing that h5py supports for setting values.
    Args:
        index (np.ndarray): Fancy indexing array of shape (N,)
    Returns:
        np.ndarray: Limited fancy indexing array of shape (M,)
        np.ndarray: Indices to read data required to construct the limited set data, shape (M,)
    """
    unique_indices, unique_idx = np.unique(index, return_index=True)
    unique_indices_sorted_ids = np.argsort(unique_indices)
    return unique_indices[unique_indices_sorted_ids], unique_idx[unique_indices_sorted_ids]

def io_reduced_indexing(
    original_index : np.ndarray,
    max_gap_size : int = 8
) -> List[Union[slice, np.ndarray, int]]:
    """
    Convert a supported fancy-index array (sorted, unique values) to a list of slices and fancy indices to reduce IO operations.
    Args:
        original_index (np.ndarray): Original indexing array of shape (N,)
    Returns:
        List[Union[slice, np.ndarray, int]]: List of slices and fancy indices
    """
    if original_index.ndim != 1:
        raise ValueError("Only 1D indexing is supported")
    
    if original_index.size == 0:
        return []
    
    result: List[Union[slice, np.ndarray, int]] = []
    pending: List[int] = []
    dtype = original_index.dtype
    n = original_index.size

    def flush_pending() -> None:
        if not pending:
            return
        if len(pending) == 1:
            result.append(pending[0])
        else:
            result.append(np.array(pending, dtype=dtype))
        pending.clear()

    pos = 0
    while pos < n:
        if pos == n - 1:
            pending.append(int(original_index[pos]))
            break

        stride = int(original_index[pos + 1] - original_index[pos])
        if stride <= 0:
            raise ValueError("Indices must be strictly increasing")
        if stride > max_gap_size:
            pending.append(int(original_index[pos]))
            pos += 1
            continue

        end = pos + 1
        while end + 1 < n:
            next_gap = int(original_index[end + 1] - original_index[end])
            if next_gap != stride or next_gap > max_gap_size:
                break
            end += 1

        run_length = end - pos + 1
        if run_length >= 2:
            flush_pending()
            start_val = int(original_index[pos])
            stop_val = int(original_index[end] + stride)
            result.append(slice(start_val, stop_val, stride))
        else:
            pending.append(int(original_index[pos]))
        pos = end + 1

    flush_pending()
    return result

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
    def build_space_from_hdf5_file(
        root : Union[h5py.Group, h5py.Dataset],
    ) -> Tuple[int, Optional[int], HDF5SpaceType]:
        if isinstance(root, h5py.Dataset):
            capacity = root.maxshape[0]
            count = root.shape[0]
            if root.dtype.kind == 'O' or root.dtype.kind == 'S':
                space = TextSpace(
                    NumpyComputeBackend,
                    max_length=root.dtype.itemsize if root.dtype.itemsize is not None else 4096,
                    dtype=str,
                    device=None,
                )
            elif NumpyComputeBackend.dtype_is_boolean(root.dtype):
                space = BinarySpace(
                    NumpyComputeBackend,
                    shape=root.shape[1:],
                    dtype=root.dtype,
                    device=None,
                )
            elif NumpyComputeBackend.dtype_is_real_floating(root.dtype) or NumpyComputeBackend.dtype_is_real_integer(root.dtype):
                space = BoxSpace(
                    NumpyComputeBackend,
                    low=-np.inf,
                    high=np.inf,
                    shape=root.shape[1:],
                    dtype=root.dtype,
                    device=None,
                )
            else:
                raise ValueError(f"Unsupported dataset dtype: {root.dtype}")
        elif isinstance(root, h5py.Group):
            if __class__.DEFAULT_KEY in root:
                assert len(root) == 1, \
                    f"If key '{__class__.DEFAULT_KEY}' is present in a group, it must be the only key. Found keys: {list(root.keys())}"
                return __class__.build_space_from_hdf5_file(root[__class__.DEFAULT_KEY])
            
            capacity = None
            count = 0
            capacity_determined = False
            spaces = {}
            for key, item in root.items():
                if key.startswith('.') or key.startswith('_'):
                    continue
                sub_count, sub_capacity, sub_space = __class__.build_space_from_hdf5_file(item)
                if not capacity_determined:
                    capacity = sub_capacity
                    count = sub_count
                    capacity_determined = True
                else:
                    assert capacity == sub_capacity, \
                        f"All datasets in a group must have the same capacity. Expected {capacity}, got {sub_capacity} in key '{key}'"
                    assert count == sub_count, \
                        f"All datasets in a group must have the same count. Expected {count}, got {sub_count} in key '{key}'"
                spaces[key] = sub_space
            space = DictSpace(
                NumpyComputeBackend,
                spaces,
                device=None,
            )
        else:
            raise ValueError(f"Unsupported HDF5 item type: {type(root)}")
        return count, capacity, space

    @staticmethod
    def load_replay_buffer_from_raw_hdf5(
        path : Union[str, os.PathLike],
    ) -> ReplayBuffer[HDF5BatchType, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]:
        assert os.path.exists(path), \
            f"Path {path} does not exist"
        assert os.access(path, os.R_OK), \
            f"Path {path} is not readable"
        root = h5py.File(
            path,
            "r"
        )
        count, capacity, single_instance_space = __class__.build_space_from_hdf5_file(root)
        storage = __class__(
            single_instance_space,
            root,
            capacity=capacity,
        )
        return ReplayBuffer(
            storage,
            storage_path_relative="storage" + (__class__.single_file_ext or ""),
            count=count,
            offset=0,
            cache_path=None
        )

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

                current_chunks = chunks if not isinstance(chunks, Mapping) else chunks.get(key, chunks.get("*", None))
                if current_chunks is not None and isinstance(current_chunks, Sequence) and len(current_chunks) < len(shape):
                    next_chunks = list(shape).copy()
                    for i, c_size in enumerate(current_chunks):
                        next_chunks[i] = c_size
                    current_chunks = tuple(next_chunks)
                    
                root.create_dataset(
                    key,
                    shape=shape,
                    maxshape=maxshape,
                    dtype=dtype,
                    compression=compression if not isinstance(compression, Mapping) else compression.get(key, None),
                    compression_opts=compression_level if not isinstance(compression_level, Mapping) else compression_level.get(
                        key, None
                    ),
                    chunks=current_chunks,
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
                    chunks=chunks if not isinstance(chunks, Mapping) else chunks.get(key, chunks.get("*", None)),
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
        reduce_io : bool = True,
    ) -> Any:
        if not isinstance(single_instance_space, DictSpace) and isinstance(root, h5py.Group):
            assert __class__.DEFAULT_KEY in root, \
                f"Expected key '{__class__.DEFAULT_KEY}' in group, got keys {list(root.keys())}"
            root = root[__class__.DEFAULT_KEY]
        
        if isinstance(single_instance_space, DictSpace):
            result = {}
            for key, space in single_instance_space.spaces.items():
                sub_root = root[key]
                result[key] = __class__.get_from(sub_root, space, index, reduce_io=reduce_io)
        else:
            if reduce_io and is_fancy_index(index):
                indexes = io_reduced_indexing(index)
                result_parts = []
                for idx in indexes:
                    if isinstance(idx, int):
                        result_parts.append(root[idx][None])
                    else:
                        result_parts.append(root[idx])
                result = np.concatenate(result_parts, axis=0)
            else:
                result = root[index]
            # Convert to numpy array if it's a scalar
            if isinstance(result, (int, float)):
                result = np.array(result)
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
        reduce_io : bool = True,
        **kwargs
    ) -> "HDF5Storage":
        assert cache_path is not None, \
            "cache_path must be provided for HDF5Storage"
        root = h5py.File(
            cache_path,
            "w",
            **kwargs
        )
        __class__._construct_hdf5_file(
            root,
            single_instance_space,
            capacity=capacity,
            initial_capacity=initial_capacity,
            compression=compression,
            compression_level=compression_level,
            chunks=chunks,
            **kwargs
        )
        return cls(
            single_instance_space,
            root,
            capacity=capacity,
            reduce_io=reduce_io,
        )
    
    @classmethod
    def load_from(
        cls, 
        path, 
        single_instance_space, 
        *, 
        capacity = None, 
        read_only = True,
        reduce_io : bool = True,
        **kwargs
    ) -> "HDF5Storage":
        assert os.path.exists(path), \
            f"Path {path} does not exist"
        
        assert os.access(path, os.R_OK), \
            f"Path {path} is not readable"
    
        # Check file permissions
        can_write = os.access(path, os.W_OK)

        root = h5py.File(
            path,
            "r+" if can_write and not read_only else "r",
            **kwargs
        )
        return cls(
            single_instance_space,
            root,
            capacity=capacity,
            reduce_io=reduce_io,
        )

    # ========== Instance Methods ==========

    def __init__(
        self,
        single_instance_space : HDF5SpaceType,
        root : h5py.Group,
        capacity : Optional[int] = None,
        reduce_io : bool = True,
    ):
        __class__._check_hdf5_file(
            root,
            single_instance_space,
            capacity
        )
        super().__init__(
            single_instance_space
        )
        self._batched_instance_space = sbu.batch_space(
            self.single_instance_space,
            1
        )
        self.root = root
        self.capacity = capacity
        self._len = self.call_function_on_first_dataset(
            root,
            lambda dataset: dataset.shape[0]
        )
        self.reduce_io = reduce_io
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
        if not is_fancy_index(index):
            return __class__.get_from(
                self.root,
                self.single_instance_space,
                index
            )
        else:
            unique_indices, reverse_mapping = fancy_indexing_to_supported_indexing(index)
            get_result_unique = __class__.get_from(
                self.root,
                self.single_instance_space,
                unique_indices
            )
            return sbu.get_at(
                self._batched_instance_space,
                get_result_unique,
                reverse_mapping
            )

    def set(self, index, value):
        if not is_fancy_index(index):
            return __class__.set_to(
                self.root,
                self.single_instance_space,
                index,
                value
            )
        else:
            unique_indices, unique_idx = fancy_indexing_to_supported_set_indexing(index)
            return __class__.set_to(
                self.root,
                self.single_instance_space,
                unique_indices,
                sbu.get_at(
                    self._batched_instance_space,
                    value,
                    unique_idx
                )
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