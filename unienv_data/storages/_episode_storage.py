from importlib import metadata
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable, Type

from unienv_interface.space import Space
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils.symbol_util import *

from unienv_data.base import SpaceStorage, BatchT, IndexableType

import os
import glob
import shutil
from abc import abstractmethod

class EpisodeStorageBase(SpaceStorage[
    BatchT,
    BArrayType,
    BDeviceType,
    BDtypeType,
    BRNGType,
]):
    """
    Base class for episode storage implementations.
    An episode storage stores episodes of data, where each episode can consist of multiple time steps.
    Each episode is stored as a separate file in the specified cache directory, with a specified file extension.
    The file naming convention is "{start_step_index}_{end_step_index}.{file_ext}", where start_index and end_index define the range of time steps in the episode.
    Note that if the storage has a fixed capacity, the {end_step_index} can be smaller than {start_step_index} due to round-robin overwriting.
    """

    # ========== Instance Implementations ==========
    single_file_ext = None

    def __init__(
        self,
        single_instance_space: Space[Any, BDeviceType, BDtypeType, BRNGType],
        file_ext : str,
        cache_filename : Union[str, os.PathLike],
        mutable : bool = True,
        capacity : Optional[int] = None,
        length : int = 0,
    ):
        assert cache_filename is not None, "EpisodeStorage requires a cache filename"
        super().__init__(single_instance_space)
        self._batched_single_space = sbu.batch_space(self.single_instance_space, 1)
        self.file_ext = file_ext
        self._cache_path = cache_filename
        self.is_mutable = mutable
        self.capacity = capacity
        self.length = length if capacity is None else capacity
        # Cache of file ranges: List[(start_idx, end_idx)]
        self._file_ranges: List[Tuple[int, int]] = []
        self._rebuild_file_range_cache()
    
    def _make_filename(self, start_idx: int, end_idx: int) -> str:
        """Construct a filename from start and end indices."""
        if self.file_ext is not None:
            return os.path.join(self._cache_path, f"{start_idx}_{end_idx}.{self.file_ext}")
        else:
            return os.path.join(self._cache_path, f"{start_idx}_{end_idx}")
    
    def get_start_end_filename_iter(self) -> Iterable[Tuple[int, int, str]]:
        """Iterate over (start_idx, end_idx, filename) tuples, constructing filenames on the fly."""
        for start_idx, end_idx in self._file_ranges:
            yield start_idx, end_idx, self._make_filename(start_idx, end_idx)
    
    def _rebuild_file_range_cache(self):
        """Rebuild the file range cache from disk."""
        all_filenames = glob.glob(os.path.join(self._cache_path, f"*_*.{self.file_ext}" if self.file_ext is not None else "*_*"))
        self._file_ranges = []
        for filename in all_filenames:
            base = os.path.basename(filename)
            name_part = base[:-(len(self.file_ext) + 1)] if self.file_ext is not None else base
            start_str, end_str = name_part.split("_")
            start_idx = int(start_str)
            end_idx = int(end_str)
            self._file_ranges.append((start_idx, end_idx))
        # Sort by start index for consistent iteration
        self._file_ranges.sort(key=lambda x: x[0])
    
    def _add_file_range(self, start_idx: int, end_idx: int):
        """Add a file range to the cache."""
        self._file_ranges.append((start_idx, end_idx))
        self._file_ranges.sort(key=lambda x: x[0])
    
    def _remove_file_range(self, start_idx: int, end_idx: int):
        """Remove a file range from the cache."""
        self._file_ranges = [(s, e) for s, e in self._file_ranges if not (s == start_idx and e == end_idx)]

    def _get_file_length(self, filename: str) -> int:
        """Get the length (number of time steps) stored in a given episode file."""
        base = os.path.basename(filename)
        name_part = base[:-(len(self.file_ext) + 1)] if self.file_ext is not None else base
        start_str, end_str = name_part.split("_")
        start_idx = int(start_str)
        end_idx = int(end_str)
        if start_idx <= end_idx:
            return end_idx - start_idx + 1
        else:
            assert self.capacity is not None, "Wrap-around file length calculation requires fixed capacity"
            return (self.capacity - start_idx) + (end_idx + 1)

    @property
    def cache_filename(self) -> Union[str, os.PathLike]:
        return self._cache_path

    @property
    def is_multiprocessing_safe(self) -> bool:
        return not self.is_mutable
    
    def convert_read_index_to_filenames_and_offsets(
        self,
        index: Union[IndexableType, BArrayType]
    ) -> Tuple[int, List[Tuple[Union[str, os.PathLike], Union[int, BArrayType], Union[int, BArrayType]]]]:
        """
        Convert an index (which can be an integer, slice, list of integers, or backend array) to a list of filenames and offsets.
        Each filename corresponds to an episode file, and the offset indicates the position within that episode.
        """
        def generate_episode_ranges(
            filename : Union[str, os.PathLike],
            start_idx : int,
            end_idx : int,
            index : Union[int, BArrayType]
        ) -> Optional[Tuple[
            Union[str, os.PathLike], # File Path (Absolute)
            Union[int, BArrayType], # Index in File
            Union[int, BArrayType], # Index into Data (Batch)
            Optional[BArrayType]] # Remaining Indexes
        ]:
            if isinstance(index, int):
                if start_idx <= index <= end_idx:
                    offset = index - start_idx
                    return (filename, slice(offset, offset + 1), slice(0, 1), None)
                elif self.capacity is not None and start_idx > end_idx and (
                    index >= start_idx or index <= end_idx
                ):
                    # Handle wrap-around case for fixed-capacity storage
                    if index >= start_idx:
                        offset = index - start_idx
                    else:
                        offset = (self.capacity - start_idx) + index
                    return (filename, slice(offset, offset + 1), slice(0, 1), None)
                else:
                    return None
            else:
                assert self.backend.is_backendarray(index)
                assert self.backend.dtype_is_real_integer(index.dtype)

                if start_idx <= end_idx:
                    mask = self.backend.logical_and(
                        index >= start_idx,
                        index <= end_idx
                    )
                    if self.backend.sum(mask) == 0:
                        return None
                    in_range_indexes = index[mask]
                    index_to_data = self.backend.nonzero(mask)[0]
                    offsets = in_range_indexes - start_idx
                    remaining_indexes = index[~mask]
                elif self.capacity is not None and start_idx > end_idx:
                    mask = self.backend.logical_or(
                        index >= start_idx,
                        index <= end_idx
                    )
                    if self.backend.sum(mask) == 0:
                        return None
                    in_range_indexes = index[mask]
                    index_to_data = self.backend.nonzero(mask)[0]
                    offsets = self.backend.where(
                        in_range_indexes >= start_idx,
                        in_range_indexes - start_idx,
                        (self.capacity - start_idx) + in_range_indexes
                    )
                    remaining_indexes = index[~mask]
                else:
                    return None
                if remaining_indexes.shape[0] == 0:
                    remaining_indexes = None
                return (filename, offsets, index_to_data, remaining_indexes)

        if isinstance(index, slice):
            if self.capacity is not None:
                index = self.backend.arange(*index.indices(self.capacity), device=self.device)
            else:
                index = self.backend.arange(*index.indices(self.length), device=self.device)
        elif index is Ellipsis:
            if self.capacity is not None:
                index = self.backend.arange(0, self.capacity, device=self.device)
            else:
                index = self.backend.arange(0, self.length, device=self.device)
        elif self.backend.is_backendarray(index) and self.backend.dtype_is_boolean(index.dtype):
            index = self.backend.nonzero(index)[0]

        batch_size = index.shape[0] if self.backend.is_backendarray(index) else 1

        remaining_index = index
        all_results = []
        for start_idx, end_idx, filename in self.get_start_end_filename_iter():
            result = generate_episode_ranges(filename, start_idx, end_idx, remaining_index)
            if result is not None:
                filename, offsets, batch_indexes, remaining_index = result
                all_results.append((filename, offsets, batch_indexes))
            if remaining_index is None:
                break
        
        assert remaining_index is None, f"Indexes {remaining_index} were not found in any episode files."
        return (batch_size, all_results)

    def extend_length(self, length):
        assert self.capacity is None, "Cannot extend length of a fixed-capacity storage"
        self.length += length

    def remove_index_range(self, start_index: int, end_index: int):
        """
        Remove data in the index range [start_index, end_index] (inclusive).
        This handles both regular files and wrap-around files.
        If start_index > end_index, this is a wrap-around removal covering [start_index, capacity) and [0, end_index].
        """
        assert self.is_mutable, "Cannot remove index range from a read-only storage"
        if start_index > end_index:
            # Wrap-around removal: remove [start_index, capacity) and [0, end_index]
            assert self.capacity is not None, "Wrap-around removal requires a fixed capacity"
            self.remove_index_range(start_index, self.capacity - 1)
            self.remove_index_range(0, end_index)
            return
        
        to_save = []
        files_to_remove = []  # List of (start_idx, end_idx, filename)
        for file_start_idx, file_end_idx, filename in self.get_start_end_filename_iter():
            
            if file_start_idx <= file_end_idx:
                # Regular (non-wrapping) file
                file_len = file_end_idx - file_start_idx + 1
                if file_start_idx >= start_index and file_end_idx <= end_index:
                    # Entire file is within the range to remove
                    files_to_remove.append((file_start_idx, file_end_idx, filename))
                elif file_start_idx <= end_index and file_end_idx >= start_index:
                    # Partial overlap with the range to remove
                    # Keep data before the removal range
                    if file_start_idx < start_index:
                        data_before = self.get_from_file(filename, slice(0, start_index - file_start_idx), file_len)
                        to_save.append((data_before, file_start_idx, start_index - 1))
                    # Keep data after the removal range
                    if file_end_idx > end_index:
                        data_after = self.get_from_file(filename, slice(end_index - file_start_idx + 1, file_end_idx - file_start_idx + 1), file_len)
                        to_save.append((data_after, end_index + 1, file_end_idx))
                    files_to_remove.append((file_start_idx, file_end_idx, filename))
            else:
                # Wrap-around file (file_start_idx > file_end_idx)
                # File contains indices [file_start_idx, capacity) and [0, file_end_idx]
                file_len = (self.capacity - file_start_idx) + file_end_idx + 1
                
                # Check if removal range overlaps with the high part [file_start_idx, capacity)
                high_overlap_start = max(start_index, file_start_idx)
                high_overlap_end = min(end_index, self.capacity - 1)
                high_overlaps = high_overlap_start <= high_overlap_end
                
                # Check if removal range overlaps with the low part [0, file_end_idx]
                low_overlap_start = max(start_index, 0)
                low_overlap_end = min(end_index, file_end_idx)
                low_overlaps = low_overlap_start <= low_overlap_end
                
                if not high_overlaps and not low_overlaps:
                    # No overlap, keep the file as is
                    continue
                
                # Calculate what to keep from the high part
                high_part_len = self.capacity - file_start_idx
                if high_overlaps:
                    # Keep before the overlap
                    if file_start_idx < high_overlap_start:
                        keep_len = high_overlap_start - file_start_idx
                        data = self.get_from_file(filename, slice(0, keep_len), file_len)
                        to_save.append((data, file_start_idx, high_overlap_start - 1))
                    # Keep after the overlap (still in high part)
                    if high_overlap_end < self.capacity - 1:
                        keep_start_offset = high_overlap_end - file_start_idx + 1
                        keep_end_offset = high_part_len
                        data = self.get_from_file(filename, slice(keep_start_offset, keep_end_offset), file_len)
                        to_save.append((data, high_overlap_end + 1, self.capacity - 1))
                else:
                    # Keep entire high part
                    data = self.get_from_file(filename, slice(0, high_part_len), file_len)
                    to_save.append((data, file_start_idx, self.capacity - 1))
                
                # Calculate what to keep from the low part
                if low_overlaps:
                    # Keep before the overlap
                    if 0 < low_overlap_start:
                        data = self.get_from_file(filename, slice(high_part_len, high_part_len + low_overlap_start), file_len)
                        to_save.append((data, 0, low_overlap_start - 1))
                    # Keep after the overlap
                    if low_overlap_end < file_end_idx:
                        keep_start_offset = high_part_len + low_overlap_end + 1
                        keep_end_offset = file_len
                        data = self.get_from_file(filename, slice(keep_start_offset, keep_end_offset), file_len)
                        to_save.append((data, low_overlap_end + 1, file_end_idx))
                else:
                    # Keep entire low part
                    data = self.get_from_file(filename, slice(high_part_len, file_len), file_len)
                    to_save.append((data, 0, file_end_idx))
                
                files_to_remove.append((file_start_idx, file_end_idx, filename))
        
        # Remove files after reading all necessary data
        for file_start_idx, file_end_idx, filename in files_to_remove:
            os.remove(filename)
            self._remove_file_range(file_start_idx, file_end_idx)
        
        # Save the data that should be kept
        for data, new_start_idx, new_end_idx in to_save:
            new_filename = self._make_filename(new_start_idx, new_end_idx)
            self.set_to_file(new_filename, data)
            self._add_file_range(new_start_idx, new_end_idx)

    def shrink_length(self, length):
        assert self.is_mutable, "Cannot shrink length of a read-only storage"
        assert self.capacity is None, "Cannot shrink length of a fixed-capacity storage"
        from_len = self.length
        to_len = max(from_len - length, 0)
        self.remove_index_range(to_len, from_len - 1)
        self.length = to_len
    
    def __len__(self):
        return self.length if self.capacity is None else self.capacity

    @abstractmethod
    def get_from_file(self, filename : str, index : Union[IndexableType, BArrayType], total_length : int) -> BatchT:
        raise NotImplementedError

    @abstractmethod
    def set_to_file(self, filename : str, batched_value : BatchT):
        raise NotImplementedError

    def get(self, index):
        batch_size, all_filename_offsets = self.convert_read_index_to_filenames_and_offsets(index)
        result_space = sbu.batch_space(self.single_instance_space, batch_size)
        result = result_space.create_empty()
        for filename, file_offset, batch_indexes in all_filename_offsets:
            file_len = self._get_file_length(filename)
            data = self.get_from_file(filename, file_offset, file_len)
            sbu.set_at(
                result_space,
                result,
                batch_indexes,
                data
            )
        if isinstance(index, int):
            result = sbu.get_at(result_space, result, 0)
        return result

    def set(self, index, value):
        assert self.is_mutable, "Storage is not mutable"
        # Make sure the index is continuous
        if isinstance(index, int):
            self.remove_index_range(index, index)
            filename = self._make_filename(index, index)
            self.set_to_file(filename, sbu.concatenate(self._batched_single_space, [value]))
            self._add_file_range(index, index)
            return
        if isinstance(index, slice):
            if self.capacity is not None:
                index = self.backend.arange(*index.indices(self.capacity), device=self.device)
            else:
                index = self.backend.arange(*index.indices(self.length), device=self.device)
        elif index is Ellipsis:
            if self.capacity is not None:
                index = self.backend.arange(0, self.capacity, device=self.device)
            else:
                index = self.backend.arange(0, self.length, device=self.device)
        elif self.backend.is_backendarray(index) and self.backend.dtype_is_boolean(index.dtype):
            index = self.backend.nonzero(index)[0]
        assert self.backend.is_backendarray(index) and self.backend.dtype_is_real_integer(index.dtype) and len(index.shape) == 1, "Index must be a 1D array of integers"
        sorted_indexes_arg = self.backend.argsort(index)
        sorted_indexes = index[sorted_indexes_arg]
        diff_sorted = sorted_indexes[1:] - sorted_indexes[:-1]
        discontinuities = self.backend.nonzero(diff_sorted > 1)[0]
        if discontinuities.shape[0] == 0:
            # Continuous
            start_index = int(sorted_indexes[0])
            end_index = int(sorted_indexes[-1])
            self.remove_index_range(start_index, end_index)
            filename = self._make_filename(start_index, end_index)
            self.set_to_file(filename, sbu.get_at(
                self._batched_single_space,
                value,
                sorted_indexes_arg
            ))
            self._add_file_range(start_index, end_index)
            return
        else:
            assert discontinuities.shape[0] == 1, "Round-robin writes can only handle one discontinuity in the index"
            assert self.capacity is not None, "Round-robin writes require a fixed capacity"
            discontinuity_pos = int(discontinuities[0]) + 1  # Position after the gap in sorted order
            
            # First segment (in sorted order): lower indices [0, ...]
            first_segment_indexes = sorted_indexes[:discontinuity_pos]
            first_start_index = int(first_segment_indexes[0])
            first_end_index = int(first_segment_indexes[-1])
            
            # Second segment (in sorted order): higher indices [..., capacity-1]
            second_segment_indexes = sorted_indexes[discontinuity_pos:]
            second_start_index = int(second_segment_indexes[0])
            second_end_index = int(second_segment_indexes[-1])

            assert first_start_index == 0 and second_end_index == self.capacity - 1, "In round-robin writes, the first segment must start at 0 and the second segment must end at capacity - 1"
            
            # Remove old data using wrap-around removal (start > end)
            self.remove_index_range(second_start_index, first_end_index)
            
            # Reorder the data: high indices first, then low indices (wrap-around order)
            # We want: [high indices data, low indices data]
            wrap_order_arg = self.backend.concat([
                sorted_indexes_arg[discontinuity_pos:],  # high indices first
                sorted_indexes_arg[:discontinuity_pos]   # then low indices
            ], axis=0)
            
            # Write a single wrap-around file: start_index > end_index
            # start_index is the first high index, end_index is the last low index
            filename = self._make_filename(second_start_index, first_end_index)
            self.set_to_file(filename, sbu.get_at(
                self._batched_single_space,
                value,
                wrap_order_arg
            ))
            self._add_file_range(second_start_index, first_end_index)


    def clear(self):
        assert self.is_mutable, "Cannot clear a read-only storage"
        if self.capacity is None:
            self.length = 0
        shutil.rmtree(self._cache_path)
        os.makedirs(self._cache_path, exist_ok=True)
        self._file_ranges = []

    def close(self):
        pass
