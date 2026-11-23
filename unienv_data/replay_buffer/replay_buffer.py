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

from unienv_data.base import BatchBase, BatchT, IndexableType, SpaceStorage, SegmentWriter
import json
import operator
import numpy as np

#: Directory (inside a saved buffer) holding the episode records, and the file mapping storage slots to episodes
_EPISODES_DIR = "episodes"
_STEP_EPISODE_IDS_FILE = "episode_ids.npy"

def _scalar_index_to_int(
    backend : ComputeBackend,
    index : Any,
) -> Any:
    """A 0-d integer index (NumPy integer, 0-d NumPy / torch / JAX array) as a Python ``int``,
    so that it selects one sample as an ``int`` does; any other index is returned unchanged."""
    if isinstance(index, (int, slice)) or index is Ellipsis:
        return index
    if isinstance(index, np.integer):
        return int(index)
    shape = getattr(index, "shape", None)
    # Any array library (an index from another backend than the buffer's included), but not booleans
    if shape is not None and len(shape) == 0 and "bool" not in str(getattr(index, "dtype", "")):
        try:
            return operator.index(index)  # integer dtypes only
        except TypeError:
            pass
    return index

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
    index = _scalar_index_to_int(backend, index)
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
            # One fused reduction: every `backend.all(...)` truth test is a host sync on GPU backends
            assert backend.all(backend.logical_and(index >= -len_transitions, index < len_transitions)), \
                f"Index values {index} are out of bounds for length {len_transitions}"
            nonzero_index = index + len_transitions
        data_index = (nonzero_index + offset) % len_transitions
        return data_index

def _normalize_physical_segment(
    segment: Optional[Sequence[int]],
    *,
    capacity: Optional[int],
) -> Optional[Tuple[int, int]]:
    if segment is None or len(segment) != 2:
        return None
    start = int(segment[0])
    end = int(segment[1])
    if capacity is None:
        if end < start:
            return None
        return (start, end)
    if start < 0 or start >= capacity or end < 0 or end >= capacity:
        return None
    return (start, end)

def _normalize_physical_segments(
    segments: Optional[Sequence[Sequence[int]]],
    *,
    capacity: Optional[int],
) -> List[Tuple[int, int]]:
    if segments is None:
        return []
    normalized_segments: List[Tuple[int, int]] = []
    for segment in segments:
        normalized_segment = _normalize_physical_segment(segment, capacity=capacity)
        if normalized_segment is not None:
            normalized_segments.append(normalized_segment)
    return normalized_segments

def _load_segment_metadata(
    metadata: Dict[str, Any],
    *,
    capacity: Optional[int],
) -> Optional[List[Tuple[int, int]]]:
    if "physical_segments" not in metadata:
        return None
    physical_segments = metadata.get("physical_segments")
    if physical_segments is None:
        return None
    return _normalize_physical_segments(physical_segments, capacity=capacity)

def _expand_physical_segment(
    segment: Tuple[int, int],
    capacity: Optional[int],
) -> List[int]:
    start, end = int(segment[0]), int(segment[1])
    if capacity is None or start <= end:
        return list(range(start, end + 1))
    return list(range(start, capacity)) + list(range(0, end + 1))

def _compress_physical_indices(
    indices: Sequence[int],
    capacity: Optional[int],
) -> List[Tuple[int, int]]:
    if len(indices) == 0:
        return []

    compressed_segments: List[Tuple[int, int]] = []
    run_start = int(indices[0])
    previous = int(indices[0])
    for raw_index in indices[1:]:
        index = int(raw_index)
        contiguous = index == previous + 1 or (
            capacity is not None and previous == capacity - 1 and index == 0
        )
        if contiguous:
            previous = index
            continue
        compressed_segments.append((run_start, previous))
        run_start = previous = index
    compressed_segments.append((run_start, previous))
    return compressed_segments

def _segment_length_from_inclusive(
    segment: Tuple[int, int],
    capacity: Optional[int],
) -> int:
    start, end = int(segment[0]), int(segment[1])
    if capacity is None or start <= end:
        return end - start + 1
    return (capacity - start) + (end + 1)

def _linear_pieces(
    segment: Tuple[int, int],
    capacity: Optional[int],
) -> List[Tuple[int, int]]:
    """Inclusive, non-wrapping pieces of a physical segment, in ring (expansion) order."""
    start, end = int(segment[0]), int(segment[1])
    if capacity is None or start <= end:
        return [(start, end)]
    return [(start, capacity - 1), (0, end)]

def _subtract_physical_range(
    segment: Tuple[int, int],
    victims: Sequence[Tuple[int, int]],
    capacity: Optional[int],
) -> List[Tuple[int, int]]:
    """
    Interval form of
    `_compress_physical_indices([i for i in _expand_physical_segment(segment) if i not in victims], capacity)`
    for inclusive, non-wrapping `victims`: O(#pieces) instead of O(capacity) per call.
    """
    survivors: List[Tuple[int, int]] = []
    for lo, hi in _linear_pieces(segment, capacity):
        current = [(lo, hi)]
        for victim_lo, victim_hi in victims:
            remaining = []
            for x, y in current:
                if victim_hi < x or victim_lo > y:
                    remaining.append((x, y))
                    continue
                if x < victim_lo:
                    remaining.append((x, victim_lo - 1))
                if victim_hi < y:
                    remaining.append((victim_hi + 1, y))
            current = remaining
        survivors.extend(current)
    # Re-join runs that continue each other (also across the end of the ring), as `_compress_physical_indices` does
    merged: List[Tuple[int, int]] = []
    for x, y in survivors:
        if merged:
            prev_x, prev_y = merged[-1]
            if x == prev_y + 1 or (capacity is not None and prev_y == capacity - 1 and x == 0):
                merged[-1] = (prev_x, y)
                continue
        merged.append((x, y))
    return merged

def _physical_segments_to_logical_half_open(
    physical_segments: Sequence[Tuple[int, int]],
    *,
    offset: int,
    count: int,
    capacity: Optional[int],
) -> List[Tuple[int, int]]:
    if capacity is None:
        return sorted(
            ((int(start), int(end) + 1) for start, end in physical_segments),
            key=lambda item: item[0],
        )

    if len(physical_segments) == 0:
        return []
    offset = int(offset) % capacity
    count = int(count)
    logical_segments: List[Tuple[int, int]] = []
    for segment in physical_segments:
        # Rotate every non-wrapping physical piece by `offset` into logical half-open ranges, keep `< count`
        ranges: List[Tuple[int, int]] = []
        for lo, hi in _linear_pieces(segment, capacity):
            # physical [max(lo, offset), hi] -> logical p - offset
            a, b = max(lo, offset), hi
            if a <= b:
                ranges.append((a - offset, b - offset + 1))
            # physical [lo, min(hi, offset - 1)] -> logical p + capacity - offset
            a, b = lo, min(hi, offset - 1)
            if a <= b:
                ranges.append((a + capacity - offset, b + capacity - offset + 1))
        clipped = sorted((a, min(b, count)) for a, b in ranges if a < count)
        if not clipped:
            continue
        run_start, run_end = clipped[0]
        for a, b in clipped[1:]:
            if a != run_end:
                logical_segments.append((run_start, run_end))
                run_start = a
            run_end = max(run_end, b)
        logical_segments.append((run_start, run_end))

    logical_segments.sort(key=lambda item: item[0])
    return logical_segments

class _LocalRingState:
    """``(count, offset)`` of a single-process ReplayBuffer.

    A separate object (rather than two ints on the buffer) so that column views created by
    :meth:`ReplayBuffer.get_column` share the counters of their parent buffer.
    """
    __slots__ = ("count", "offset")

    def __init__(self, count: int, offset: int):
        self.count = int(count)
        self.offset = int(offset)

    def read(self) -> Tuple[int, int]:
        return self.count, self.offset

    def write(self, count: int, offset: int) -> None:
        self.count = int(count)
        self.offset = int(offset)

    def __getstate__(self):
        return (self.count, self.offset)

    def __setstate__(self, state):
        self.count, self.offset = int(state[0]), int(state[1])

class _SharedRingState:
    """``(count, offset)`` of a multiprocessing ReplayBuffer, shared by every process using the buffer.

    Both values are packed into one 64-bit word of shared memory (``offset * (capacity + 1) + count``).
    Writers store it once, while holding the buffer's writer lock; readers load it once, without any lock.
    An aligned 64-bit load / store is a single atomic access on the 64-bit platforms CPython supports, so a
    reader always sees a pair that some writer published - never the new offset with the old count, which
    with two separate counters made reads during wrap-around map indices with a mismatched pair.
    Where packing is not possible (a 32-bit platform, or a capacity too large to pack) readers take the lock.
    """
    #: Largest capacity whose (count, offset) pair fits into one signed 64-bit word.
    MAX_PACKED_CAPACITY = (1 << 31) - 1

    def __init__(self, count: int, offset: int, capacity: Optional[int], lock):
        self._base = None if capacity is None else int(capacity) + 1
        self._lock = lock
        self._lock_free = ctypes.sizeof(ctypes.c_void_p) >= 8 and (
            capacity is None or int(capacity) <= self.MAX_PACKED_CAPACITY
        )
        # `RawValue`: no per-value lock (the buffer's writer lock serializes writers); 8-byte aligned in the mp heap
        self._word = mp.RawValue(ctypes.c_int64, 0)
        self.write(count, offset)

    def _pack(self, count: int, offset: int) -> int:
        if self._base is None:
            assert offset == 0, "Offset must be 0 when capacity is None"
            return count
        return offset * self._base + count

    def _unpack(self, word: int) -> Tuple[int, int]:
        if self._base is None:
            return word, 0
        offset, count = divmod(word, self._base)
        return count, offset

    def read(self) -> Tuple[int, int]:
        if self._lock_free:
            return self._unpack(self._word.value)
        with self._lock:
            return self._unpack(self._word.value)

    @property
    def count(self) -> int:
        if self._lock_free:
            word = self._word.value  # one load: the count alone needs no consistent pair
            return word if self._base is None else word % self._base
        return self.read()[0]

    @property
    def offset(self) -> int:
        return self.read()[1]

    def write(self, count: int, offset: int) -> None:
        # Callers hold the writer lock
        self._word.value = self._pack(int(count), int(offset))

class ReplayBuffer(BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """Ring-buffer style batch storage built on top of a ``SpaceStorage``.

    The buffer exposes the standard ``BatchBase`` interface while handling
    round-robin overwrite semantics, persistence metadata, and optional
    multiprocessing-safe counters.

    With ``multiprocessing=True`` writers serialize on a shared lock and publish ``count`` and ``offset``
    together; readers in other processes take no lock and always map indices with a consistent pair.
    A read that races with a write can still return rows that the write is overwriting (the oldest samples
    of a full buffer), as with any lock-free ring buffer.
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
        maintain_segment_metadata : bool = False,
        episode_space : Optional[Space] = None,
        episode_storage_cls : Optional[Type[SpaceStorage]] = None,
        episode_capacity : Optional[int] = -1,
        episode_storage_kwargs : Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "ReplayBuffer[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        """Create a new replay buffer and its backing storage.

        With ``episode_space``, every segment (episode) of the buffer carries one record of that
        space (task, instruction, success, return, ...), stored once in a second storage of
        ``episode_storage_cls`` (``episode_storage_kwargs``) and passed when the segment closes
        (``extend(..., episode_data=...)``, ``mark_segment_end(episode_data=...)``,
        ``commit_segment(writer, episode_data=...)``): every segment requires one. Records are kept
        for as long as any step of their episode is in the buffer; ``episode_capacity`` (``-1``: the
        buffer's ``capacity``, which always suffices) bounds how many are stored. Segment metadata
        is maintained. See :meth:`get_episode_data` and :meth:`with_episode_data`.
        """
        if episode_space is not None:
            maintain_segment_metadata = True
        if multiprocessing and maintain_segment_metadata:
            raise RuntimeError(
                "ReplayBuffer.create() does not support maintain_segment_metadata=True with multiprocessing=True for mutable buffers; "
                "load a read-only buffer instead."
            )
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
        episode_buffer = None
        if episode_space is not None:
            if episode_storage_cls is None:
                raise ValueError(
                    "episode_space needs an episode_storage_cls for the episode records "
                    "(e.g. HDF5Storage or ParquetStorage; PytorchTensorStorage for torch spaces without text)"
                )
            episode_buffer = ReplayBuffer.create(
                episode_storage_cls,
                episode_space,
                cache_path=None if cache_path is None else os.path.join(cache_path, _EPISODES_DIR),
                capacity=capacity if episode_capacity is not None and episode_capacity < 0 else episode_capacity,
                **(episode_storage_kwargs or {}),
            )
        elif episode_storage_cls is not None or episode_storage_kwargs:
            raise ValueError("episode_storage_cls / episode_storage_kwargs need an episode_space")
        return ReplayBuffer(
            storage,
            storage_path_relative,
            0,
            0,
            cache_path=cache_path,
            multiprocessing=multiprocessing,
            maintain_segment_metadata=maintain_segment_metadata,
            episode_buffer=episode_buffer,
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

        physical_segments = _load_segment_metadata(metadata, capacity=capacity)
        if multiprocessing and not read_only and physical_segments is not None:
            raise RuntimeError(
                "ReplayBuffer.load_from() does not support multiprocessing=True with maintained segment metadata unless read_only=True."
            )

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

        episode_buffer = None
        step_episode_ids = None
        next_episode_id = 0
        episodes = metadata.get("episodes")
        if episodes is not None:
            episode_backend = get_class_from_full_name(episodes["backend"])
            episode_buffer = ReplayBuffer.load_from(
                os.path.join(path, episodes["path_relative"]),
                backend=episode_backend,
                device=device if episode_backend == backend else None,
                read_only=read_only,
            )
            step_episode_ids = np.load(os.path.join(path, episodes["step_episode_ids_file"]))
            next_episode_id = int(episodes["next_episode_id"])

        return ReplayBuffer(
            storage,
            metadata["storage_path_relative"],
            count,
            offset,
            cache_path=path,
            multiprocessing=multiprocessing,
            maintain_segment_metadata=physical_segments is not None,
            physical_segments=physical_segments,
            episode_buffer=episode_buffer,
            step_episode_ids=step_episode_ids,
            next_episode_id=next_episode_id,
        )

    # =========== Instance Attributes and Methods ==========
    def dumps(self, path : Union[str, os.PathLike]):
        """Persist the replay buffer metadata and its backing storage."""
        with self._lock_scope():
            if self.storage.has_open_segment or (
                self._active_segment_start_physical is not None and self._active_segment_length > 0
            ):
                raise RuntimeError(
                    "ReplayBuffer.dumps() cannot be called while a segment is open; "
                    "call mark_segment_end() before dumping"
                )
            os.makedirs(path, exist_ok=True)
            storage_path = os.path.join(path, self.storage_path_relative)
            self.storage.dumps(storage_path)
            count, offset = self._ring_state.read()
            metadata = {
                "type": __class__.__name__,
                "count": count,
                "offset": offset,
                "capacity": self.storage.capacity,
                "storage_cls": get_full_class_name(type(self.storage)),
                "storage_path_relative": self.storage_path_relative,
                "single_instance_space": bsu.space_to_json(self.storage.single_instance_space),
                "physical_segments": None if not self._maintain_segment_metadata else [list(segment) for segment in self._physical_segments],
            }
            if self._episode_buffer is not None:
                self._episode_buffer.dumps(os.path.join(path, _EPISODES_DIR))
                np.save(os.path.join(path, _STEP_EPISODE_IDS_FILE), self._step_episode_ids[:self._step_episode_id_length()])
                metadata["episodes"] = {
                    "path_relative": _EPISODES_DIR,
                    "backend": get_full_class_name(self._episode_buffer.backend),
                    "step_episode_ids_file": _STEP_EPISODE_IDS_FILE,
                    "next_episode_id": int(self._next_episode_id),
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
        *,
        maintain_segment_metadata : bool = False,
        physical_segments : Optional[Sequence[Tuple[int, int]]] = None,
        active_segment_start_physical : Optional[int] = None,
        active_segment_length : int = 0,
        episode_buffer : Optional["ReplayBuffer"] = None,
        step_episode_ids : Optional[np.ndarray] = None,
        next_episode_id : int = 0,
    ):
        """Wrap a storage object with replay-buffer indexing semantics.

        ``episode_buffer`` holds the episode records of the segments (see ``create``);
        ``step_episode_ids`` maps every storage slot to the episode of the step it holds
        (-1: no closed episode) and ``next_episode_id`` is the id of the next record.
        """
        self.storage = storage
        self._column_parent : Optional["ReplayBuffer"] = None
        self._storage_path_relative = storage_path_relative
        self._cache_path = cache_path
        self._multiprocessing = multiprocessing
        self._maintain_segment_metadata = bool(
            maintain_segment_metadata or physical_segments is not None or episode_buffer is not None
        )
        self._episode_buffer = episode_buffer
        self._next_episode_id = int(next_episode_id)
        # Id of the open segment's record, if it was given with its first append()
        self._active_segment_episode_id : Optional[int] = None
        if episode_buffer is None:
            self._step_episode_ids = None
        else:
            # Ids are assigned in the order records are written, so record `id` sits at
            # logical index `id - (next_episode_id - len(episode_buffer))` of the episode buffer
            capacity = storage.capacity
            ids = np.full(capacity if capacity is not None else max(16, int(count)), -1, dtype=np.int64)
            if step_episode_ids is not None:
                step_episode_ids = np.asarray(step_episode_ids, dtype=np.int64)
                ids[:step_episode_ids.shape[0]] = step_episode_ids
            self._step_episode_ids = ids

        if self._multiprocessing and self._maintain_segment_metadata and self.storage.is_mutable:
            raise RuntimeError(
                "ReplayBuffer does not support maintain_segment_metadata=True with multiprocessing=True for mutable buffers; "
                "load the buffer read-only or disable multiprocessing."
            )

        if multiprocessing and storage.is_mutable:
            assert storage.is_multiprocessing_safe, "Storage is not multiprocessing safe"
            # A lock from a fork context cannot be handed to spawn / forkserver processes, while a spawn-context
            # lock works with every start method (fork children simply inherit it)
            self._lock = mp.get_context("spawn").RLock()
            self._ring_state = _SharedRingState(count, offset, storage.capacity, self._lock)
        else:
            self._lock = None
            self._ring_state = _LocalRingState(count, offset)

        self._physical_segments = (
            _normalize_physical_segments(physical_segments, capacity=self.capacity)
            if self._maintain_segment_metadata
            else []
        )
        if self._maintain_segment_metadata and active_segment_start_physical is not None and int(active_segment_length) > 0:
            self._active_segment_start_physical = int(active_segment_start_physical)
            if self.capacity is not None:
                self._active_segment_start_physical %= self.capacity
            self._active_segment_length = int(active_segment_length)
        else:
            self._active_segment_start_physical = None
            self._active_segment_length = 0
        
        super().__init__(
            storage.single_instance_space,
            None
        )

    def _require_not_column_view(self, operation: str) -> None:
        if getattr(self, "_column_parent", None) is not None:
            raise RuntimeError(
                f"Cannot {operation} a column view of a ReplayBuffer; {operation} the parent buffer instead"
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
        return self._ring_state.count

    @count.setter
    def count(self, value: int) -> None:
        with self._lock_scope():
            self._ring_state.write(value, self._ring_state.read()[1])
    
    @property
    def offset(self) -> int:
        return self._ring_state.offset

    @offset.setter
    def offset(self, value: int) -> None:
        with self._lock_scope():
            self._ring_state.write(self._ring_state.read()[0], value)

    def _logical_to_physical_index(self, idx: Union[IndexableType, BArrayType]) -> Union[int, BArrayType]:
        # One snapshot of (count, offset): reading them separately races with a writer in another process
        count, offset = self._ring_state.read()
        return index_with_offset(self.backend, idx, count, offset, self.device)

    @property
    def capacity(self) -> Optional[int]:
        return self.storage.capacity

    @property
    def maintain_segment_metadata(self) -> bool:
        return self._maintain_segment_metadata

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
        idx = _scalar_index_to_int(self.backend, idx)
        if hasattr(self.storage, "get_flattened"):
            data = self.storage.get_flattened(self._logical_to_physical_index(idx))
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
        data_index = self._logical_to_physical_index(_scalar_index_to_int(self.backend, idx))
        data = self.storage.get(data_index)
        return data, None
    
    def set_flattened_at(self, idx: Union[IndexableType, BArrayType], value: BArrayType) -> None:
        idx = _scalar_index_to_int(self.backend, idx)
        if hasattr(self.storage, "set_flattened"):
            self.storage.set_flattened(
                self._logical_to_physical_index(idx), 
                value
            )
            return

        if isinstance(idx, int):
            value = sfu.unflatten_data(self.single_space, value)
        else:
            value = sfu.unflatten_data(self._batched_space, value, start_dim=1)
        self.set_at(idx, value)

    def set_at(self, idx, value):
        self.storage.set(self._logical_to_physical_index(_scalar_index_to_int(self.backend, idx)), value)

    def extend_flattened(
        self,
        value: BArrayType
    ):
        unflattened_data = sfu.unflatten_data(self._batched_space, value, start_dim=1)
        self.extend(unflattened_data)

    def _active_segment_physical_indices(self) -> List[int]:
        if self._active_segment_start_physical is None or self._active_segment_length <= 0:
            return []
        start = int(self._active_segment_start_physical)
        length = int(self._active_segment_length)
        if self.capacity is None:
            return list(range(start, start + length))
        end = (start + length - 1) % self.capacity
        if start <= end:
            return list(range(start, end + 1))
        return list(range(start, self.capacity)) + list(range(0, end + 1))

    def _remove_physical_range_from_metadata(self, start_physical: int, length: int) -> None:
        if not self._maintain_segment_metadata or length <= 0:
            return

        if self.capacity is None:
            victims = [(int(start_physical), int(start_physical) + int(length) - 1)]
        else:
            start_physical = int(start_physical) % self.capacity
            end_physical = (start_physical + int(length) - 1) % self.capacity
            victims = _linear_pieces((start_physical, end_physical), self.capacity)

        updated_segments: List[Tuple[int, int]] = []
        for segment in self._physical_segments:
            seg_start, seg_end = segment
            if seg_start <= seg_end and all(victim_hi < seg_start or victim_lo > seg_end for victim_lo, victim_hi in victims):
                updated_segments.append(segment)  # untouched (the common case: only the oldest segment loses samples)
                continue
            updated_segments.extend(_subtract_physical_range(segment, victims, self.capacity))
        self._physical_segments = updated_segments

        if self._active_segment_start_physical is None or self._active_segment_length <= 0:
            self._active_segment_start_physical = None
            self._active_segment_length = 0
            return
        active_start = int(self._active_segment_start_physical)
        active_end = active_start + int(self._active_segment_length) - 1
        if self.capacity is not None:
            active_end %= self.capacity
        active_runs = _subtract_physical_range((active_start, active_end), victims, self.capacity)
        if len(active_runs) == 0:
            self._active_segment_start_physical = None
            self._active_segment_length = 0
            return

        # The active segment should remain a single tail range. If the removal
        # splits it unexpectedly, keep the most recent surviving run.
        active_run = active_runs[-1]
        self._active_segment_start_physical = int(active_run[0])
        self._active_segment_length = _segment_length_from_inclusive(active_run, self.capacity)

    def _append_to_active_segment(self, physical_index: int) -> None:
        if self._active_segment_start_physical is None or self._active_segment_length <= 0:
            self._active_segment_start_physical = int(physical_index)
            self._active_segment_length = 0
        self._active_segment_length += 1
        
    def extend(self, value, *, episode_data : Optional[Any] = None):
        """Append ``value`` (a batch) at the tail as one segment; ``episode_data`` is its episode
        record (required if the buffer has an ``episode_space``, see ``create``)."""
        self._require_not_column_view("extend")
        with self._lock_scope():
            B = sbu.batch_size_data(value)
            if B == 0:
                return
            self._check_episode_data_argument(episode_data)
            if self.storage.has_open_segment or (
                self._maintain_segment_metadata
                and self._active_segment_start_physical is not None
                and self._active_segment_length > 0
            ):
                raise RuntimeError("Cannot extend while an append-built segment is open; call mark_segment_end() first")

            old_count, old_offset = self._ring_state.read()

            episode_id = self._write_episode_record(episode_data)

            if self.capacity is None:
                assert old_offset == 0, "Offset must be 0 when capacity is None"
                self.storage.extend_length(B)
                self.storage.set(slice(old_count, old_count + B), value)
                self._ring_state.write(old_count + B, 0)
                if self._maintain_segment_metadata:
                    self._physical_segments.append((old_count, old_count + B - 1))
                self._set_step_episode_ids(slice(old_count, old_count + B), episode_id)
                return
            
            # We have a fixed capacity, only keep the last `capacity` elements
            if B >= self.capacity:
                self.storage.set(Ellipsis, sbu.get_at(self._batched_space, value, slice(-self.capacity, None)))
                self._ring_state.write(self.capacity, 0)
                if self._maintain_segment_metadata:
                    self._physical_segments = [(0, self.capacity - 1)]
                self._active_segment_start_physical = None
                self._active_segment_length = 0
                self._set_step_episode_ids(slice(0, self.capacity), episode_id)
                return
            
            # Otherwise, perform round-robin writes
            start_physical = (old_offset + old_count) % self.capacity
            if start_physical + B <= self.capacity:
                # No wrap-around: a slice lets every storage take its contiguous fast path
                # (h5py hyperslab, tensor slice copy, one file for episode storages, ...) instead of fancy indexing
                indexes = slice(start_physical, start_physical + B)
            else:
                indexes = (self.backend.arange(B, device=self.device) + start_physical) % self.capacity
            self.storage.set(indexes, value)
            self._account_round_robin_write(start_physical, B)
            if episode_id is not None:
                self._set_step_episode_ids(self._physical_run(start_physical, B), episode_id)

    def _account_round_robin_write(self, start_physical: int, B: int) -> None:
        """Update count / offset / segment metadata after `B` (<= capacity) samples were written at the tail."""
        old_count, old_offset = self._ring_state.read()
        outflow = max(0, old_count + B - self.capacity)
        if outflow > 0 and self._maintain_segment_metadata:
            self._remove_physical_range_from_metadata(old_offset, outflow)
        # Publish count and offset together (see `_SharedRingState`)
        self._ring_state.write(min(old_count + B, self.capacity), (old_offset + outflow) % self.capacity)
        if self._maintain_segment_metadata:
            self._physical_segments.append((start_physical, (start_physical + B - 1) % self.capacity))

    def open_segment_writer(self) -> SegmentWriter:
        """
        A writer that builds one segment off to the side; `commit_segment()` then appends it at the tail.
        Unlike `append()` / `mark_segment_end()`, any number of writers can be open at a time
        (e.g. one per environment of a batched rollout), and they can be committed in any order.
        Storages that support it (e.g. `VideoStorage`) encode the samples as they are appended.
        """
        return self.storage.open_segment_writer()

    def commit_segment(self, writer: SegmentWriter, *, episode_data : Optional[Any] = None) -> None:
        """Append the segment built by `writer` (from `open_segment_writer()`) at the tail, as `extend()` would.

        ``episode_data`` is the segment's episode record (required if the buffer has an
        ``episode_space``; an empty writer is aborted and needs none)."""
        with self._lock_scope():
            assert writer.storage is self.storage, "The writer was not opened on this replay buffer"
            B = len(writer)
            if B == 0:
                writer.abort()
                return
            self._check_episode_data_argument(episode_data)
            if self.storage.has_open_segment or (
                self._maintain_segment_metadata
                and self._active_segment_start_physical is not None
                and self._active_segment_length > 0
            ):
                raise RuntimeError("Cannot commit a segment while an append-built segment is open; call mark_segment_end() first")

            old_count, old_offset = self._ring_state.read()
            if self.capacity is not None and B > self.capacity:
                writer.abort()
                raise ValueError(f"Cannot commit a {B}-sample segment into a replay buffer of capacity {self.capacity}")
            episode_id = self._write_episode_record(episode_data)
            if self.capacity is None:
                self.storage.extend_length(B)
                try:
                    writer.commit(old_count)
                except Exception:
                    self.storage.shrink_length(B)
                    raise
                self._ring_state.write(old_count + B, 0)
                if self._maintain_segment_metadata:
                    self._physical_segments.append((old_count, old_count + B - 1))
                self._set_step_episode_ids(slice(old_count, old_count + B), episode_id)
                return

            start_physical = (old_offset + old_count) % self.capacity
            writer.commit(start_physical)
            self._account_round_robin_write(start_physical, B)
            if episode_id is not None:
                self._set_step_episode_ids(self._physical_run(start_physical, B), episode_id)

    def append(self, value, *, episode_data : Optional[Any] = None):
        """Append one step to the open segment (opening one if needed).

        ``episode_data`` on the first step of a segment is the segment's episode record, written
        right away (``mark_segment_end()`` then takes none); otherwise the record is given to
        ``mark_segment_end(episode_data=...)``.
        """
        self._require_not_column_view("append to")
        with self._lock_scope():
            if episode_data is not None:
                if self._episode_buffer is None:
                    raise ValueError("this replay buffer has no episode_space: episode_data cannot be stored")
                if (self._active_segment_start_physical is not None and self._active_segment_length > 0) \
                        or self._active_segment_episode_id is not None:
                    raise ValueError(
                        "episode_data goes with the first append() of a segment (or to mark_segment_end()); "
                        "this segment is already open"
                    )
            old_count, old_offset = self._ring_state.read()
            if self.capacity is None:
                physical_index = old_count
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
                self._ring_state.write(old_count + 1, 0)
                if self._maintain_segment_metadata:
                    self._append_to_active_segment(physical_index)
                self._set_step_episode_ids(slice(physical_index, physical_index + 1), self._open_step_episode_id(episode_data))
                return

            physical_index = (old_offset + old_count) % self.capacity
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

            outflow = max(0, old_count + 1 - self.capacity)
            if outflow > 0 and self._maintain_segment_metadata:
                self._remove_physical_range_from_metadata(old_offset, outflow)
            self._ring_state.write(min(old_count + 1, self.capacity), (old_offset + outflow) % self.capacity)
            if self._maintain_segment_metadata:
                self._append_to_active_segment(physical_index)
            self._set_step_episode_ids(slice(physical_index, physical_index + 1), self._open_step_episode_id(episode_data))

    def _open_step_episode_id(self, episode_data : Optional[Any]) -> Optional[int]:
        """Episode id of a step just appended to the open segment, writing the record it came with.

        The record is written after the step, so a failed append leaves no record behind; -1 while
        the open segment has no record yet (``None`` without episode records).
        """
        if self._episode_buffer is None:
            return None
        if episode_data is not None:
            self._active_segment_episode_id = self._write_episode_record(episode_data)
        return -1 if self._active_segment_episode_id is None else self._active_segment_episode_id

    def mark_segment_end(self, *, episode_data : Optional[Any] = None):
        """Close the segment built by ``append()``; ``episode_data`` is its episode record
        (required if the buffer has an ``episode_space`` and the segment holds steps, unless the
        record came with the segment's first ``append()``)."""
        with self._lock_scope():
            open_segment = self._active_segment_start_physical is not None and self._active_segment_length > 0
            if self._active_segment_episode_id is not None:
                if episode_data is not None:
                    raise ValueError("this segment got its episode record with its first append(); mark_segment_end() takes none")
                # Its steps already map to the record
                self._active_segment_episode_id = None
                self.storage.mark_segment_end()
                if self._maintain_segment_metadata and open_segment:
                    active_end_physical = (
                        self._active_segment_start_physical + self._active_segment_length - 1
                        if self.capacity is None
                        else (self._active_segment_start_physical + self._active_segment_length - 1) % self.capacity
                    )
                    self._physical_segments.append((self._active_segment_start_physical, active_end_physical))
                self._active_segment_start_physical = None
                self._active_segment_length = 0
                return
            if self._episode_buffer is not None and open_segment:
                self._check_episode_data_argument(episode_data)
            elif episode_data is not None:
                raise ValueError("episode_data was given but there is no open segment (nothing was appended since the last mark_segment_end())"
                                 if self._episode_buffer is not None else "this replay buffer has no episode_space")
            episode_id = self._write_episode_record(episode_data) if (self._episode_buffer is not None and open_segment) else None
            active_slots = self._active_segment_physical_indices() if episode_id is not None else []
            self.storage.mark_segment_end()
            if self._maintain_segment_metadata and open_segment:
                active_end_physical = (
                    self._active_segment_start_physical + self._active_segment_length - 1
                    if self.capacity is None
                    else (self._active_segment_start_physical + self._active_segment_length - 1) % self.capacity
                )
                self._physical_segments.append((self._active_segment_start_physical, active_end_physical))
            self._active_segment_start_physical = None
            self._active_segment_length = 0
            if episode_id is not None:
                self._set_step_episode_ids(np.asarray(active_slots, dtype=np.int64), episode_id)

    def get_segments(self):
        column_parent = getattr(self, "_column_parent", None)
        if column_parent is not None:
            return column_parent.get_segments()
        with self._lock_scope():
            count, offset = self._ring_state.read()
            if self._maintain_segment_metadata:
                return _physical_segments_to_logical_half_open(
                    self._physical_segments,
                    offset=offset,
                    count=count,
                    capacity=self.capacity,
                )
            storage_segments = self.storage.get_segments()
            if storage_segments is None:
                return None
            return _physical_segments_to_logical_half_open(
                _normalize_physical_segments(storage_segments, capacity=self.capacity),
                offset=offset,
                count=count,
                capacity=self.capacity,
            )

    # =========== Episode records ==========
    def _episode_owner(self) -> "ReplayBuffer":
        """The buffer holding the episode records (the parent of a column view)."""
        parent = getattr(self, "_column_parent", None)
        return parent if parent is not None else self

    @property
    def episode_space(self) -> Optional[Space]:
        """Space of the per-episode records (``None`` if the buffer stores none, see ``create``)."""
        episode_buffer = self._episode_owner()._episode_buffer
        return None if episode_buffer is None else episode_buffer.single_space

    def _check_episode_data_argument(self, episode_data : Optional[Any]) -> None:
        if self._episode_buffer is None:
            if episode_data is not None:
                raise ValueError("this replay buffer has no episode_space: episode_data cannot be stored")
        elif episode_data is None:
            raise ValueError("this replay buffer has an episode_space: every segment needs its episode_data")

    def _write_episode_record(self, episode_data : Optional[Any]) -> Optional[int]:
        """Store the record of a closing segment; returns its episode id (``None`` without episode records)."""
        if self._episode_buffer is None:
            return None
        self._episode_buffer.extend(sbu.concatenate(self._episode_buffer.single_space, [episode_data]))
        episode_id = self._next_episode_id
        self._next_episode_id += 1
        return episode_id

    def _physical_run(self, start_physical : int, length : int) -> Union[slice, np.ndarray]:
        """Storage slots of ``length`` steps written from ``start_physical`` (wrapping around a fixed capacity)."""
        if self.capacity is None or start_physical + length <= self.capacity:
            return slice(start_physical, start_physical + length)
        return (np.arange(length, dtype=np.int64) + start_physical) % self.capacity

    def _step_episode_id_length(self) -> int:
        """Slots of ``_step_episode_ids`` in use: the capacity, or the count of an unbounded buffer."""
        return self.capacity if self.capacity is not None else self.count

    def _set_step_episode_ids(self, slots : Union[slice, np.ndarray], episode_id : Optional[int]) -> None:
        """Map storage slots to ``episode_id`` (-1: a step of the open segment)."""
        if self._step_episode_ids is None or episode_id is None:
            return
        if isinstance(slots, slice):
            needed = int(slots.stop)
        else:
            slots = np.asarray(slots, dtype=np.int64)
            needed = int(slots.max()) + 1 if slots.size else 0
        size = self._step_episode_ids.shape[0]
        if needed > size:  # an unbounded buffer grew
            grown = np.full(max(needed, 2 * size), -1, dtype=np.int64)
            grown[:size] = self._step_episode_ids
            self._step_episode_ids = grown
        self._step_episode_ids[slots] = episode_id

    def get_episode_ids(self, idx : Union[IndexableType, BArrayType]) -> Union[int, np.ndarray]:
        """Episode id of the step(s) at logical index ``idx``: an ``int``, or a NumPy int64 array.

        Steps of the segment that is still open (``append()`` without ``mark_segment_end()``) have
        no episode yet and get ``-1``.
        """
        owner = self._episode_owner()
        if owner._step_episode_ids is None:
            raise RuntimeError("this replay buffer has no episode_space")
        physical = self._logical_to_physical_index(_scalar_index_to_int(self.backend, idx))
        if isinstance(physical, (int, np.integer)):
            return int(owner._step_episode_ids[int(physical)])
        physical = np.asarray(self.backend.to_numpy(physical), dtype=np.int64)
        return owner._step_episode_ids[physical]

    def get_episode_data(self, episode_ids : Union[int, Sequence[int], np.ndarray]) -> Any:
        """The episode record of each id (one record for an ``int``, a batch for an array).

        The episodes available are the ones with a step in the buffer. Raises ``IndexError`` for
        ``-1`` (a step of the open segment), for episodes whose steps were all overwritten (or
        whose record no longer fits ``episode_capacity``) and for unknown ids.
        """
        owner = self._episode_owner()
        episode_buffer = owner._episode_buffer
        if episode_buffer is None:
            raise RuntimeError("this replay buffer has no episode_space")
        with owner._lock_scope():
            next_id = owner._next_episode_id
            first_stored_id = next_id - len(episode_buffer)
            # Segments are stored in id order (only the open segment follows them), so the episodes
            # with a step in the buffer are the ids from the one of the oldest step on
            first_live_id = owner._oldest_step_episode_id()
            first_id = max(first_stored_id, first_live_id)
            ids = np.asarray(episode_ids, dtype=np.int64)
            out_of_range = (ids < first_id) | (ids >= next_id)
            if np.any(out_of_range):
                bad = np.unique(ids[out_of_range])[:5].tolist()
                if -1 in bad:
                    reason = "-1 marks steps of the segment that is still open (close it with mark_segment_end())"
                elif min(bad) < first_live_id:
                    reason = "all their steps were overwritten"
                elif min(bad) < first_stored_id:
                    reason = "their records were evicted while their steps remain: episode_capacity is too small"
                else:
                    reason = "no such episode"
                raise IndexError(f"episode id(s) {bad} not available: {reason}; available ids: [{first_id}, {next_id})")
            # Record `id` sits at logical index `id - first_stored_id` of the episode buffer
            if ids.ndim == 0:
                return episode_buffer.get_at(int(ids) - first_stored_id)
            logical = episode_buffer.backend.from_numpy(
                ids - first_stored_id, dtype=episode_buffer.backend.default_integer_dtype, device=episode_buffer.device,
            )
            return episode_buffer.get_at(logical)

    def _oldest_step_episode_id(self) -> int:
        """Episode id of the oldest step (``next_episode_id`` if there is no closed episode in the buffer)."""
        count, offset = self._ring_state.read()
        if count == 0:
            return self._next_episode_id
        oldest = int(self._step_episode_ids[offset % self.capacity if self.capacity is not None else 0])
        return self._next_episode_id if oldest < 0 else oldest

    def get_episode_segments(self) -> List[Tuple[int, int, int]]:
        """``(start, end, episode_id)`` of every closed segment, as logical half-open ranges in buffer order."""
        segments = self.get_segments() or []
        if not segments:
            return []
        starts = self.backend.from_numpy(
            np.array([start for start, _ in segments], dtype=np.int64),
            dtype=self.backend.default_integer_dtype, device=self.device,
        )
        ids = self.get_episode_ids(starts)
        return [(int(start), int(end), int(episode_id)) for (start, end), episode_id in zip(segments, ids)]

    def with_episode_data(self) -> "ReplayBufferEpisodeView":
        """A read-only view whose samples carry their episode: metadata ``episode_id`` and ``episode``.

        Steps are only joined with their episode records on request, through this view (e.g.
        ``StepSampler(rb.with_episode_data(), ...)``); reading the buffer itself is unchanged.
        """
        return ReplayBufferEpisodeView(self)

    def get_column(self, nested_keys):
        if hasattr(self.storage, "get_column"):
            column_storage = self.storage.get_column(nested_keys)
            count, offset = self._ring_state.read()
            column = ReplayBuffer(
                column_storage,
                self.storage_path_relative + "_column_" + ".".join(nested_keys),
                count,
                offset,
                cache_path=self.cache_path,
                multiprocessing=False,
            )
            # The column is a live view: it shares the ring-buffer counters (and
            # lock) with this buffer and reports this buffer's segments, so it
            # stays consistent after further extends / wrap-around.
            column._ring_state = self._ring_state
            column._lock = self._lock
            column._multiprocessing = self._multiprocessing
            column._maintain_segment_metadata = self._maintain_segment_metadata
            column._column_parent = self
            return column
        else:
            return super().get_column(nested_keys)

    def clear(self):
        self._require_not_column_view("clear")
        with self._lock_scope():
            self.storage.clear()
            self._ring_state.write(0, 0)
            self._physical_segments = []
            self._active_segment_start_physical = None
            self._active_segment_length = 0
            self._active_segment_episode_id = None
            if self._episode_buffer is not None:
                # Ids keep increasing: a record id is never reused
                self._episode_buffer.clear()
                self._step_episode_ids[:] = -1

    def close(self) -> None:
        # Only dereference the storage
        # the actual cleanup logic should be implemented in the storage's close method if necessary
        # We do this to avoid issues when we create multiple ReplayBuffer instances that reference the same underlying storage (e.g. when using different RBs for different columns)
        self.storage = None


class ReplayBufferEpisodeView(BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """The steps of a :class:`ReplayBuffer` with their episodes (see :meth:`ReplayBuffer.with_episode_data`).

    Samples are the buffer's steps; their metadata holds ``episode_id`` and ``episode``, the
    episode's record converted to the buffer's backend and device. Reading a step of the segment
    that is still open raises ``ValueError``: its episode record does not exist yet.
    """
    is_mutable = False

    def __init__(self, buffer : ReplayBuffer[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
        from unienv_interface.space import BoxSpace, DictSpace
        episode_space = buffer.episode_space
        if episode_space is None:
            raise RuntimeError("this replay buffer has no episode_space")
        backend, device = buffer.backend, buffer.device
        self._source_episode_space = episode_space
        self._convert_episodes = episode_space.backend != backend or (device is not None and episode_space.device != device)
        local_episode_space = episode_space.to(backend=backend, device=device) if self._convert_episodes else episode_space
        super().__init__(
            buffer.single_space,
            DictSpace(backend, {
                "episode_id": BoxSpace(backend, 0, np.inf, dtype=backend.default_integer_dtype, shape=(), device=device),
                "episode": local_episode_space,
            }, device=device),
        )
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def get_at(self, idx):
        return self.buffer.get_at(idx)

    def get_flattened_at(self, idx):
        return super().get_flattened_at(_scalar_index_to_int(self.backend, idx))

    def get_flattened_at_with_metadata(self, idx):
        return super().get_flattened_at_with_metadata(_scalar_index_to_int(self.backend, idx))

    def get_at_with_metadata(self, idx):
        idx = _scalar_index_to_int(self.backend, idx)
        data = self.buffer.get_at(idx)
        episode_ids = self.buffer.get_episode_ids(idx)
        if np.any(np.asarray(episode_ids) < 0):
            raise ValueError(
                "some steps belong to the segment that is still open: its episode record does not exist yet "
                "(call mark_segment_end(episode_data=...) first)"
            )
        episodes = self.buffer.get_episode_data(episode_ids)
        if self._convert_episodes:
            episodes = self._source_episode_space.data_to(episodes, backend=self.backend, device=self.device)
        backend = self.backend
        episode_ids = backend.from_numpy(np.asarray(episode_ids, dtype=np.int64), dtype=backend.default_integer_dtype, device=self.device)
        return data, {"episode_id": episode_ids, "episode": episodes}
