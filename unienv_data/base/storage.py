import abc
import os
from typing import Generic, TypeVar, Optional, Any, Dict, Union, Tuple, Sequence, Callable, Type
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space import Space
from unienv_interface.space.space_utils import batch_utils as sbu
from .common import BatchBase, BatchT, IndexableType

class SegmentWriter(abc.ABC):
    """
    Builds one contiguous segment of a storage off to the side.
    Values appended to a writer land at an index range that is only chosen when the writer is committed,
    so several writers of one storage can be filled at the same time (e.g. one per environment of a batched rollout)
    and committed in any order. Obtain one with `SpaceStorage.open_segment_writer()`.
    """

    def __init__(self, storage : "SpaceStorage"):
        self.storage = storage
        self.length = 0
        self.is_finished = False

    def __len__(self) -> int:
        return self.length

    def append(self, value) -> None:
        """Append one structured sample to the segment."""
        if self.is_finished:
            raise RuntimeError("Cannot append to a finished segment writer")
        self._append(value)
        self.length += 1

    def finish(self) -> None:
        """
        Declare the segment complete (no more appends) and release what only appending needs (e.g. encoders).
        Called by `commit()` if needed; finish early when the commit has to wait.
        """
        if not self.is_finished:
            self._finish()
            self.is_finished = True

    @abc.abstractmethod
    def _append(self, value) -> None:
        raise NotImplementedError

    def _finish(self) -> None:
        pass

    @abc.abstractmethod
    def commit(self, start_index : int) -> None:
        """
        Write the segment to the storage indices `[start_index, start_index + len(self))`
        (modulo the capacity of fixed-capacity storages). Storages with unlimited capacity must already be long enough.
        The writer cannot be used afterwards.
        """
        raise NotImplementedError

    def abort(self) -> None:
        """Drop the segment without writing it."""
        pass

    def _commit_indices(self, start_index : int) -> Union[slice, BArrayType]:
        capacity = self.storage.capacity
        if capacity is None:
            return slice(int(start_index), int(start_index) + self.length)
        start_index = int(start_index) % capacity
        if start_index + self.length <= capacity:
            return slice(start_index, start_index + self.length)
        backend = self.storage.backend
        return (backend.arange(self.length, device=self.storage.device) + start_index) % capacity


class BufferedSegmentWriter(SegmentWriter):
    """Keeps the appended samples in memory and writes them with one `set()` on commit."""

    def __init__(self, storage : "SpaceStorage"):
        super().__init__(storage)
        self._values = []

    def _append(self, value) -> None:
        self._values.append(value)

    def commit(self, start_index : int) -> None:
        self.finish()
        if self.length > 0:
            batched_space = sbu.batch_space(self.storage.single_instance_space, 1)
            self.storage.set(self._commit_indices(start_index), sbu.concatenate(batched_space, self._values))
        self._values = []

    def abort(self) -> None:
        self._values = []


class MappedSegmentWriter(SegmentWriter):
    """Forwards to a writer of an inner storage, mapping every appended sample with `transform`."""

    def __init__(self, storage : "SpaceStorage", inner_writer : SegmentWriter, transform : Callable[[Any], Any]):
        super().__init__(storage)
        self.inner_writer = inner_writer
        self.transform = transform

    def _append(self, value) -> None:
        self.inner_writer.append(self.transform(value))

    def _finish(self) -> None:
        self.inner_writer.finish()

    def commit(self, start_index : int) -> None:
        self.finish()
        self.inner_writer.commit(start_index)

    def abort(self) -> None:
        self.inner_writer.abort()


class SpaceStorage(abc.ABC, Generic[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """
    SpaceStorage is an abstract base class for storages that hold instances of a specific space.
    It provides a common interface for creating, loading, and managing the storage of instances of a given space.
    Note that if you want your space storage to support multiprocessing, you need to check / implement `__getstate__` and `__setstate__` methods to ensure that the storage can be pickled and unpickled correctly.
    """
    # ========== Class Creation and Loading Methods ==========
    @classmethod
    def create(
        cls,
        single_instance_space : Space[BatchT, BDeviceType, BDtypeType, BRNGType],
        *args,
        capacity : Optional[int],
        cache_path : Optional[Union[str, os.PathLike]] = None,
        multiprocessing : bool = False,
        **kwargs
    ) -> "SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        """Create a new storage instance for samples from ``single_instance_space``."""
        raise NotImplementedError

    @classmethod
    def load_from(
        cls,
        path: Union[str, os.PathLike],
        single_instance_space: Space[BatchT, BDeviceType, BDtypeType, BRNGType],
        *,
        capacity : Optional[int] = None,
        read_only : bool = True,
        multiprocessing : bool = False,
        **kwargs
    ) -> "SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        """Restore a storage instance previously written to disk."""
        raise NotImplementedError

    # ========== Class Attributes ==========
    
    """
    The file extension (e.g. `.pt`) used for saving a single instance of the space.
    If this is None, it means the storage stores files in a folder
    """
    single_file_ext : Optional[str] = None

    # ======== Instance Attributes ==========
    """
    The total capacity (number of single instances) of the storage.
    If None, the storage has unlimited capacity.
    """
    capacity : Optional[int] = None

    """
    The cache path for the storage.
    If None, the storage will not use caching.
    """
    cache_filename : Optional[Union[str, os.PathLike]] = None

    """
    Can the storage instance be safely used in multiprocessing environments after creation?
    If True, the storage can be used in multiprocessing environments.
    """
    is_multiprocessing_safe : bool = False

    """
    Is the storage mutable? If False, the storage is read-only.
    """
    is_mutable : bool = True

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.single_instance_space.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.single_instance_space.device

    def __init__(
        self,
        single_instance_space : Space[BatchT, BDeviceType, BDtypeType, BRNGType],
    ):
        """Bind the storage to the space of one stored element."""
        self.single_instance_space = single_instance_space
        self._single_sample_batched_space = sbu.batch_space(self.single_instance_space, 1)
        self._active_segment_start_index: Optional[int] = None
        self._active_segment_index: Optional[int] = None

    @property
    def has_open_segment(self) -> bool:
        return self._active_segment_start_index is not None

    @property
    def pending_segment_length(self) -> int:
        return 0

    def get_segments(self):
        return None

    def mark_segment_start(self, start_index: int) -> None:
        assert not self.has_open_segment, "A segment is already open"
        if self.capacity is None:
            self._active_segment_start_index = int(start_index)
            self._active_segment_index = int(start_index)
        else:
            self._active_segment_start_index = int(start_index) % self.capacity
            self._active_segment_index = self._active_segment_start_index

    def append(self, value) -> None:
        """Append one structured sample to the storage.

        Appended values may be immediately visible or only visible after
        ``mark_segment_end()`` depending on the storage implementation.
        """
        if not self.has_open_segment or self._active_segment_index is None:
            raise RuntimeError(f"Cannot append to {type(self).__name__} without an open segment")

        batched_value = sbu.concatenate(self._single_sample_batched_space, [value])
        current_index = self._active_segment_index
        self.set(slice(current_index, current_index + 1), batched_value)
        if self.capacity is None:
            self._active_segment_index = current_index + 1
        else:
            self._active_segment_index = (current_index + 1) % self.capacity

    def mark_segment_end(self) -> None:
        if not self.has_open_segment:
            return
        self._active_segment_start_index = None
        self._active_segment_index = None

    def abort_segment(self) -> None:
        if not self.has_open_segment:
            return
        self._active_segment_start_index = None
        self._active_segment_index = None
    
    def extend_length(self, length : int) -> None:
        """
        This is used by capacity = None storages to extend the length of the storage
        If this is called on a storage with a fixed capacity, we will simply ignore the call.
        """
        pass

    def shrink_length(self, length : int) -> None:
        """
        This is used by capacity = None storages to shrink the length of the storage
        If this is called on a storage with a fixed capacity, we will simply ignore the call.
        """
        pass

    def __len__(self) -> int:
        """
        Returns the number of instances in the storage
        Storages with unlimited capacity should implement this method to return the current length of the storage.
        """
        if self.capacity is None:
            raise NotImplementedError(f"__len__ is not implemented for class {type(self).__name__}")
        return self.capacity

    # We don't define them here, since they are optional and the `ReplayBuffer` checks if they are implemented
    # by using hasattr(self, "get_flattened") and hasattr(self, "set_flattened").
    # def get_flattened(self, index : Union[IndexableType, BArrayType]) -> BArrayType:
    #     raise NotImplementedError
    
    # def set_flattened(self, index : Union[IndexableType, BArrayType], value : BArrayType) -> None:
    #     raise NotImplementedError

    @abc.abstractmethod
    def get(self, index : Union[IndexableType, BArrayType]) -> BatchT:
        """Read one or more samples from storage."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def set(self, index : Union[IndexableType, BArrayType], value : BatchT) -> None:
        """Write one or more samples into storage."""
        raise NotImplementedError
    
    # We don't define them here, since they are optional and the `ReplayBuffer` checks if they are implemented
    # by using hasattr(self, "get_column")
    # def get_column(self, nested_keys : Sequence[str]) -> "SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
    #     raise NotImplementedError

    def open_segment_writer(self) -> SegmentWriter:
        """
        A writer that builds one segment of this storage off to the side; see `SegmentWriter`.
        Unlike `mark_segment_start()` / `append()`, any number of writers can be open at a time.
        The default keeps the samples in memory until the commit; storages override it to stream instead.
        """
        assert self.is_mutable, f"Cannot write to a read-only {type(self).__name__}"
        return BufferedSegmentWriter(self)

    def clear(self) -> None:
        """
        Clear all data inside the storage and set the length to 0 if the storage has unlimited capacity.
        For storages with fixed capacity, this should reset the storage to its initial state.
        """
        self.abort_segment()
        if self.capacity is None:
            self.shrink_length(len(self))

    @abc.abstractmethod
    def dumps(self, path : Union[str, os.PathLike]) -> None:
        """
        Dumps the storage to the specified path.
        This is used for storages that have a single file extension (e.g. `.pt` for PyTorch).
        """
        raise NotImplementedError

    def close(self) -> None:
        pass
    
    def __del__(self):
        self.close()
