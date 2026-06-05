import abc
import os
import dataclasses
import multiprocessing as mp
import ctypes
from contextlib import nullcontext

from typing import Generic, TypeVar, Optional, Any, Dict, Union, Tuple, Sequence, Callable, Type, List
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space import Space
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.space.space_utils import serialization_utils as bsu
from unienv_interface.utils.symbol_util import get_class_from_full_name, get_full_class_name

from unienv_data.base import BatchBase, BatchT, IndexableType, SpaceStorage
import json

def index_with_offset(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    index : Union[int, slice, BArrayType],
    len_transitions : int,
    offset : int,
    device : Optional[BDeviceType] = None,
) -> Union[int, BArrayType]:
    """
    Helpful function to convert round robin indices to data indices in the SpaceStorage
    Returns:
        - data_index: The index in the storage that corresponds to the given index with the specified offset.
    """
    if index is Ellipsis:
        nonzero_index = backend.arange(len_transitions, device=device)
        data_index = (nonzero_index + offset) % len_transitions
        return data_index
    elif isinstance(index, int):
        assert -len_transitions <= index < len_transitions, f"Index {index} is out of bounds for length {len_transitions}"
        return (index + len_transitions + offset) % len_transitions
    elif isinstance(index, slice):
        nonzero_index = backend.arange(*index.indices(len_transitions), device=device)
        data_index = (nonzero_index + offset) % len_transitions
        return data_index
    else:
        assert len(index.shape) == 1, f"Index shape {index.shape} is not 1D"
        if index.shape == (len_transitions, ) and backend.dtype_is_boolean(
            index.dtype
        ):
            # Boolean mask, rotate the mask by offset
            nonzero_index = backend.nonzero(index)[0]
        else:
            assert backend.dtype_is_real_integer(index.dtype), f"Index dtype {index.dtype} is not an integer"
            assert backend.all(index >= -len_transitions) and backend.all(index < len_transitions), \
                f"Index values {index} are out of bounds for length {len_transitions}"
            nonzero_index = index + len_transitions
        data_index = (nonzero_index + offset) % len_transitions
        return data_index

def _chunk_to_logical_range(chunk: List[int], count: int) -> Optional[Tuple[int, int]]:
    if len(chunk) == 0:
        return None
    if len(chunk) >= count:
        return (0, count)
    return (int(chunk[0]), int((chunk[-1] + 1) % count))

def _map_physical_segments_to_logical(
    physical_segments: Sequence[Tuple[int, int]],
    *,
    offset: int,
    count: int,
    capacity: Optional[int],
) -> Optional[List[Tuple[int, int]]]:
    if capacity is None:
        return sorted(((int(start), int(end) + 1) for start, end in physical_segments), key=lambda item: item[0])
    if count == 0:
        return []

    logical_segments: List[Tuple[int, int]] = []
    for start, end in physical_segments:
        if start <= end:
            physical_indexes = range(int(start), int(end) + 1)
        else:
            physical_indexes = list(range(int(start), capacity)) + list(range(0, int(end) + 1))

        logical_positions: List[int] = []
        for physical_index in physical_indexes:
            logical_index = (physical_index - offset) % capacity
            if logical_index < count:
                logical_positions.append(int(logical_index))

        if not logical_positions:
            continue

        chunk = [logical_positions[0]]
        for logical_index in logical_positions[1:]:
            if (chunk[-1] + 1) % count != logical_index:
                chunk_range = _chunk_to_logical_range(chunk, count)
                if chunk_range is not None:
                    logical_segments.append(chunk_range)
                chunk = [logical_index]
            else:
                chunk.append(logical_index)

        chunk_range = _chunk_to_logical_range(chunk, count)
        if chunk_range is not None:
            logical_segments.append(chunk_range)

    logical_segments.sort(key=lambda item: item[0])
    return logical_segments

class ReplayBuffer(BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """Ring-buffer style batch storage built on top of a ``SpaceStorage``.

    The buffer exposes the standard ``BatchBase`` interface while handling
    round-robin overwrite semantics, persistence metadata, and optional
    multiprocessing-safe counters.
    """
    # =========== Class Attributes ==========
    @staticmethod
    def create(
        storage_cls : Type[SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        single_instance_space : Space[BatchT, BDeviceType, BDtypeType, BRNGType],
        *args,
        cache_path : Optional[Union[str, os.PathLike]] = None,
        capacity : Optional[int] = None,
        multiprocessing : bool = False,
        **kwargs
    ) -> "ReplayBuffer[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        """Create a new replay buffer and its backing storage."""
        storage_path_relative = "storage" + (storage_cls.single_file_ext or "")
        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)
        storage = storage_cls.create(
            single_instance_space,
            *args,
            cache_path=None if cache_path is None else os.path.join(cache_path, storage_path_relative),
            capacity=capacity,
            multiprocessing=multiprocessing,
            **kwargs
        )
        return ReplayBuffer(
            storage,
            storage_path_relative,
            0,
            0,
            cache_path=cache_path,
            multiprocessing=multiprocessing
        )
    
    @staticmethod
    def is_loadable_from(
        path : Union[str, os.PathLike]
    ) -> bool:
        if os.path.exists(os.path.join(path, "metadata.json")):
            with open(os.path.join(path, "metadata.json"), "r") as f:
                metadata = json.load(f)
            return metadata.get('type', None) == __class__.__name__
        return False
    
    @staticmethod
    def get_length_from_path(
        path : Union[str, os.PathLike]
    ) -> Optional[int]:
        if os.path.exists(os.path.join(path, "metadata.json")):
            with open(os.path.join(path, "metadata.json"), "r") as f:
                metadata = json.load(f)
            if metadata.get('type', None) != __class__.__name__:
                return None
            return int(metadata["count"])
        return None
    
    @staticmethod
    def get_capacity_from_path(
        path : Union[str, os.PathLike]
    ) -> Optional[int]:
        if os.path.exists(os.path.join(path, "metadata.json")):
            with open(os.path.join(path, "metadata.json"), "r") as f:
                metadata = json.load(f)
            if metadata.get('type', None) != __class__.__name__:
                return None
            capacity = metadata.get("capacity", None)
            return None if capacity is None else int(capacity)
        return None
    
    @staticmethod
    def get_space_from_path(
        path : Union[str, os.PathLike],
        *,
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device: Optional[BDeviceType] = None,
    ) -> Optional[Space[BatchT, BDeviceType, BDtypeType, BRNGType]]:
        if os.path.exists(os.path.join(path, "metadata.json")):
            with open(os.path.join(path, "metadata.json"), "r") as f:
                metadata = json.load(f)
            if metadata.get('type', None) != __class__.__name__:
                return None
            single_instance_space = bsu.json_to_space(
                metadata["single_instance_space"], backend, device
            )
            return single_instance_space
        return None

    @staticmethod
    def load_from(
        path : Union[str, os.PathLike],
        *,
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device: Optional[BDeviceType] = None,
        read_only : bool = True,
        multiprocessing : bool = False,
        single_instance_space : Optional[Space[BatchT, BDeviceType, BDtypeType, BRNGType]] = None,
        **storage_kwargs
    ) -> "ReplayBuffer[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        """Load a replay buffer plus its backing storage from disk.
        
        If ``single_instance_space`` is provided, it is validated against the
        space reconstructed from the on-disk metadata and, if compatible,
        reused instead of keeping the reconstructed copy.  This avoids
        allocating duplicate ``Space`` objects when many buffers with
        identical spaces are loaded simultaneously (e.g. for
        ``CombinedBatch``).
        """
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        assert metadata['type'] == __class__.__name__, f"Metadata type {metadata['type']} does not match expected type {__class__.__name__}"
        offset = int(metadata["offset"])
        count = int(metadata["count"])
        capacity = metadata.get("capacity", None)
        if capacity is not None:
            capacity = int(capacity)
        loaded_space = bsu.json_to_space(
            metadata["single_instance_space"], backend, device
        )

        if single_instance_space is not None:
            assert single_instance_space == loaded_space, \
                f"Provided single_instance_space does not match the space loaded from metadata. " \
                f"Provided: {single_instance_space}, Loaded: {loaded_space}"
            del loaded_space
            use_space = single_instance_space
        else:
            use_space = loaded_space
        
        storage_cls : Type[SpaceStorage] = get_class_from_full_name(metadata["storage_cls"])
        storage_path = os.path.join(path, metadata["storage_path_relative"])

        storage = storage_cls.load_from(
            storage_path,
            use_space,
            capacity=capacity,
            read_only=read_only,
            multiprocessing=multiprocessing,
            **storage_kwargs
        )
        return ReplayBuffer(
            storage,
            metadata["storage_path_relative"],
            count,
            offset,
            cache_path=path,
            multiprocessing=multiprocessing
        )

    # =========== Instance Attributes and Methods ==========
    def dumps(self, path : Union[str, os.PathLike]):
        """Persist the replay buffer metadata and its backing storage."""
        with self._lock_scope():
            os.makedirs(path, exist_ok=True)
            storage_path = os.path.join(path, self.storage_path_relative)
            self.storage.dumps(storage_path)
            metadata = {
                "type": __class__.__name__,
                "count": self.count,
                "offset": self.offset,
                "capacity": self.storage.capacity,
                "storage_cls": get_full_class_name(type(self.storage)),
                "storage_path_relative": self.storage_path_relative,
                "single_instance_space": bsu.space_to_json(self.storage.single_instance_space),
            }
            with open(os.path.join(path, "metadata.json"), "w") as f:
                json.dump(metadata, f)

    def __init__(
        self,
        storage : SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType],
        storage_path_relative : str,
        count : int = 0,
        offset : int = 0,
        cache_path : Optional[Union[str, os.PathLike]] = None,
        multiprocessing : bool = False,
    ):
        """Wrap a storage object with replay-buffer indexing semantics."""
        self.storage = storage
        self._storage_path_relative = storage_path_relative
        self._cache_path = cache_path
        self._multiprocessing = multiprocessing
        if multiprocessing and storage.is_mutable:
            assert storage.is_multiprocessing_safe, "Storage is not multiprocessing safe"
            self._lock = mp.RLock()
            self._count_value = mp.Value(ctypes.c_long, int(count))
            self._offset_value = mp.Value(ctypes.c_long, int(offset))
        else:
            self._lock = None
            self._count_value = int(count)
            self._offset_value = int(offset)
        
        super().__init__(
            storage.single_instance_space,
            None
        )

    def _lock_scope(self):
        if self._lock is not None:
            return self._lock
        else:
            return nullcontext()

    @property
    def cache_path(self) -> Optional[Union[str, os.PathLike]]:
        return self._cache_path

    @property
    def storage_path_relative(self) -> str:
        return self._storage_path_relative

    def __len__(self) -> int:
        return self.count

    @property
    def count(self) -> int:
        return self._count_value.value if not isinstance(self._count_value, int) else self._count_value

    @count.setter
    def count(self, value: int) -> None:
        if not isinstance(self._count_value, int):
            self._count_value.value = int(value)
        else:
            self._count_value = int(value)
    
    @property
    def offset(self) -> int:
        return self._offset_value.value if not isinstance(self._offset_value, int) else self._offset_value

    @offset.setter
    def offset(self, value: int) -> None:
        if not isinstance(self._offset_value, int):
            self._offset_value.value = int(value)
        else:
            self._offset_value = int(value)

    @property
    def capacity(self) -> Optional[int]:
        return self.storage.capacity

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.storage.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.storage.device

    @property
    def is_mutable(self) -> bool:
        return self.storage.is_mutable
    
    @property
    def is_multiprocessing_safe(self) -> bool:
        return self._multiprocessing

    def get_flattened_at(self, idx):
        return self.get_flattened_at_with_metadata(idx)[0]

    def get_flattened_at_with_metadata(self, idx: Union[IndexableType, BArrayType]) -> BArrayType:
        if hasattr(self.storage, "get_flattened"):
            data = self.storage.get_flattened(idx)
            return data, None

        data, metadata = self.get_at_with_metadata(idx)
        if isinstance(idx, int):
            data = sfu.flatten_data(self.single_space, data)
        else:
            data = sfu.flatten_data(self._batched_space, data, start_dim=1)
        return data, metadata

    def get_at(self, idx):
        return self.get_at_with_metadata(idx)[0]

    def get_at_with_metadata(self, idx):
        data_index = index_with_offset(
            self.backend,
            idx,
            self.count,
            self.offset,
            self.device
        )
        data = self.storage.get(data_index)
        return data, None
    
    def set_flattened_at(self, idx: Union[IndexableType, BArrayType], value: BArrayType) -> None:
        if hasattr(self.storage, "set_flattened"):
            self.storage.set_flattened(
                index_with_offset(
                    self.backend,
                    idx,
                    self.count,
                    self.offset,
                    self.device
                ), 
                value
            )
            return

        if isinstance(idx, int):
            value = sfu.unflatten_data(self.single_space, value)
        else:
            value = sfu.unflatten_data(self._batched_space, value, start_dim=1)
        self.set_at(idx, value)

    def set_at(self, idx, value):
        self.storage.set(index_with_offset(
            self.backend,
            idx,
            self.count,
            self.offset,
            self.device
        ), value)

    def extend_flattened(
        self,
        value: BArrayType
    ):
        unflattened_data = sfu.unflatten_data(self._batched_space, value, start_dim=1)
        self.extend(unflattened_data)
        
    def extend(self, value):
        with self._lock_scope():
            B = sbu.batch_size_data(value)
            if B == 0:
                return
            if self.capacity is None:
                assert self.offset == 0, "Offset must be 0 when capacity is None"
                self.storage.extend_length(B)
                self.storage.set(slice(self.count, self.count + B), value)
                self.count += B
                return
            
            # We have a fixed capacity, only keep the last `capacity` elements
            if B >= self.capacity:
                self.storage.set(Ellipsis, sbu.get_at(self._batched_space, value, slice(-self.capacity, None)))
                self.count = self.capacity
                self.offset = 0
                return
            
            # Otherwise, perform round-robin writes
            indexes = (self.backend.arange(B, device=self.device) + self.offset + self.count) % self.capacity
            self.storage.set(indexes, value)
            outflow = max(0, self.count + B - self.capacity)
            if outflow > 0:
                self.offset = (self.offset + outflow) % self.capacity
            self.count = min(self.count + B, self.capacity)

    def append(self, value):
        with self._lock_scope():
            if self.capacity is None:
                physical_index = self.count
                self.storage.extend_length(1)
                opened_segment = False
                try:
                    if not self.storage.has_open_segment:
                        self.storage.mark_segment_start(physical_index)
                        opened_segment = True
                    self.storage.append(value)
                except Exception:
                    try:
                        self.storage.shrink_length(1)
                    finally:
                        if opened_segment:
                            try:
                                self.storage.abort_segment()
                            except Exception:
                                pass
                    raise
                self.count += 1
                return

            physical_index = (self.offset + self.count) % self.capacity
            if self.storage.has_open_segment and self.storage.pending_segment_length >= self.capacity:
                self.storage.mark_segment_end()

            opened_segment = False
            try:
                if not self.storage.has_open_segment:
                    self.storage.mark_segment_start(physical_index)
                    opened_segment = True
                self.storage.append(value)
            except Exception:
                if opened_segment:
                    try:
                        self.storage.abort_segment()
                    except Exception:
                        pass
                raise

            outflow = max(0, self.count + 1 - self.capacity)
            if outflow > 0:
                self.offset = (self.offset + outflow) % self.capacity
            self.count = min(self.count + 1, self.capacity)

    def mark_segment_end(self):
        with self._lock_scope():
            self.storage.mark_segment_end()

    def get_segments(self):
        with self._lock_scope():
            storage_segments = self.storage.get_segments()
            if storage_segments is None:
                return None
            logical_segments = _map_physical_segments_to_logical(
                storage_segments,
                offset=self.offset,
                count=self.count,
                capacity=self.capacity,
            )
            if logical_segments is None:
                return None
            return logical_segments

    def get_column(self, nested_keys):
        if hasattr(self.storage, "get_column"):
            column_storage = self.storage.get_column(nested_keys)
            return ReplayBuffer(
                column_storage,
                self.storage_path_relative + "_column_" + ".".join(nested_keys),
                self.count,
                self.offset,
                cache_path=self.cache_path,
                multiprocessing=self._multiprocessing
            )
        else:
            return super().get_column(nested_keys)

    def clear(self):
        with self._lock_scope():
            self.count = 0
            self.offset = 0
            self.storage.clear()

    def close(self) -> None:
        # Only dereference the storage
        # the actual cleanup logic should be implemented in the storage's close method if necessary
        # We do this to avoid issues when we create multiple ReplayBuffer instances that reference the same underlying storage (e.g. when using different RBs for different columns)
        self.storage = None
