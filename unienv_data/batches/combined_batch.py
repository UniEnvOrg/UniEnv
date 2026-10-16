from typing import Sequence, List, Tuple, Union, Dict, Any, Optional, Generic, TypeVar, Iterable, Iterator
from types import EllipsisType
from concurrent.futures import ThreadPoolExecutor
import os
import abc
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.space import Space, DictSpace, BoxSpace
import dataclasses
import copy

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu

from ..base.common import BatchBase, IndexableType, BatchT

__all__ = [
    "convert_single_index_to_batch",
    "convert_index_to_batch",
    "CombinedBatch",
]

def convert_single_index_to_batch(
    backend : ComputeBackend,
    index_starts : BArrayType, # (num_batches, )
    total_length : int, # Total length of all batches
    idx : int,
    device : Optional[BDeviceType] = None,
) -> Tuple[int, int]:
    """
    Convert a single index for this batch to a tuple of:
        - The index of the batch
        - The index to index into the batch
    """
    assert -total_length <= idx < total_length, f"Index {idx} out of bounds for batch of size {total_length}"
    if idx < 0:
        idx += total_length
    batch_index = int(backend.sum(
        idx >= index_starts
    ) - 1)
    return batch_index, idx - int(index_starts[batch_index])

def convert_index_to_batch(
    backend : ComputeBackend,
    index_starts : BArrayType, # (num_batches, )
    total_length : int, # Total length of all batches
    idx : Union[IndexableType, BArrayType],
    device : Optional[BDeviceType] = None,
) -> Tuple[
    int, 
    Union[List[
        Tuple[int, BArrayType, BArrayType]
    ], int]
]:
    """
    Convert an index for this batch to a tuple of:
        - The length of the resulting array
        - List of tuples, each containing:
            - The index of the batch
            - The index to index into the batch
            - The bool mask to index into the resulting array
    """
    if isinstance(idx, slice):
        idx_array = backend.arange(
            *idx.indices(total_length),
            dtype=backend.default_index_dtype,
            device=device
        )
    elif idx is Ellipsis:
        idx_array = backend.arange(
            total_length,
            dtype=backend.default_index_dtype,
            device=device
        )
    elif backend.is_backendarray(idx):
        assert len(idx.shape) == 1, "Index must be 1D"
        assert backend.dtype_is_real_integer(idx.dtype) or backend.dtype_is_boolean(idx.dtype), \
            f"Index must be of integer or boolean type, got {idx.dtype}"
        if backend.dtype_is_boolean(idx.dtype):
            assert idx.shape[0] == total_length, f"Boolean index must have the same length as the batch, got {idx.shape[0]} vs {total_length}"
            idx_array = backend.nonzero(idx)[0]
        else:
            assert backend.all(backend.logical_and(-total_length <= idx, idx < total_length)), \
                f"Index array contains out of bounds indices for batch of size {total_length}"
            idx_array = (idx + total_length) % total_length
    else:
        raise ValueError(f"Invalid index type: {type(idx)}")
    
    assert bool(backend.all(
        backend.logical_and(
            -total_length <= idx_array,
            idx_array < total_length
        )
    )), f"Index {idx} converted to {idx_array} is out of bounds for batch of size {total_length}"
    
    # Convert negative indices to positive indices
    idx_array = backend.at(idx_array)[idx_array < 0].add(total_length)
    idx_array_bigger = idx_array[:, None] >= index_starts[None, :] # (idx_array_shape, len(self.batches))
    idx_array_batch_idx = backend.sum(
        idx_array_bigger,
        axis=-1
    ) - 1 # (idx_array_shape, )
    
    result_batch_list = []
    batch_indexes = backend.unique_values(idx_array_batch_idx)
    for i in range(batch_indexes.shape[0]):
        batch_index = int(batch_indexes[i])
        result_mask = idx_array_batch_idx == batch_index
        index_into_batch = idx_array[result_mask] - index_starts[batch_index]
        result_batch_list.append((batch_index, index_into_batch, result_mask))
    return idx_array.shape[0], result_batch_list

class CombinedBatch(BatchBase[
    BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    def __init__(
        self,
        batches : Sequence[BatchBase[
            BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
        ]],
        device : Optional[BDeviceType] = None,
        use_thread_pool : bool = False,
    ):
        assert len(batches) > 1, "More than one batch must be provided"
        backend = batches[0].backend
        single_space = batches[0].single_space
        single_metadata_space = batches[0].single_metadata_space
        is_mutable = True
        
        for batch in batches:
            assert backend == batch.backend, "All batches must have the same backend"
            assert single_space == batch.single_space, "All batches must have the same single space"
            assert single_metadata_space == batch.single_metadata_space, "All batches must have the same metadata space"
            if device is not None:
                assert device == batch.device, "All batches must have the same device"
            is_mutable = is_mutable and batch.is_mutable
        
        if single_metadata_space is not None:
            new_metadata_space = copy.copy(single_metadata_space)
        else:
            new_metadata_space = DictSpace(
                backend,
                {},
                device=device,
            )
        
        # Add our own metadata space for batch index
        new_metadata_space['batch_index'] = BoxSpace(
            backend,
            low=0,
            high=len(batches),
            dtype=backend.default_integer_dtype,
            device=device,
            shape=(),
        )
        super().__init__(single_space, new_metadata_space)

        self.is_mutable = is_mutable
        self.batches = batches
        self.use_thread_pool = use_thread_pool
        # Thread pool for `use_thread_pool`, owned by the process that created it.
        self._executor : Optional[ThreadPoolExecutor] = None
        self._executor_pid : Optional[int] = None
        # (id(space), batch size) -> (space, batched space)
        self._batch_space_cache : Dict[Tuple[int, int], Tuple[Space, Space]] = {}
        self._build_index_cache()

    def _get_executor(self) -> ThreadPoolExecutor:
        """Thread pool reused across calls (starting one per access costs more than the gathers
        it parallelises for small batches). Re-created after a fork, since threads do not
        survive it: work submitted to the inherited pool would never run."""
        pid = os.getpid()
        if self._executor is None or self._executor_pid != pid:
            self._executor = ThreadPoolExecutor(max_workers=len(self.batches), thread_name_prefix="CombinedBatch")
            self._executor_pid = pid
        return self._executor

    def _cached_batch_space(self, space : Space, batch_size : int) -> Space:
        """``sbu.batch_space(space, batch_size)``, memoised: building batched spaces (bounds arrays
        and their validation) dominates the per-call cost for small batches, and training loops
        reuse the same batch size."""
        cache = self._batch_space_cache
        key = (id(space), int(batch_size))
        entry = cache.get(key)
        if entry is None or entry[0] is not space:
            if len(cache) >= 16:
                cache.clear()
            entry = (space, sbu.batch_space(space, int(batch_size)))
            cache[key] = entry
        return entry[1]

    def get_column(self, nested_keys : Sequence[Any]) -> BatchBase:
        """
        Lazy view over one field of the samples that combines the children's column views, so
        that children able to read a single column cheaply (replay buffers, stacked views, ...)
        do so instead of fetching whole samples. Values, spaces, metadata and the flattened
        layout are those of ``SubItemBatch(self, nested_keys)``, which is returned when the
        children's column views cannot be combined. The view is read-only.
        """
        from .subitem_batch import SubItemBatch
        fallback = lambda: super(CombinedBatch, self).get_column(nested_keys)
        columns = [batch.get_column(nested_keys) for batch in self.batches]
        if all(isinstance(column, SubItemBatch) and column.batch is batch for column, batch in zip(columns, self.batches)):
            return fallback()  # nothing to gain over a view of this batch
        if any(column.single_metadata_space != batch.single_metadata_space for column, batch in zip(columns, self.batches)):
            return fallback()
        try:
            combined = CombinedBatch(columns, use_thread_pool=self.use_thread_pool)
        except AssertionError:
            return fallback()  # the column views disagree on their spaces
        combined.is_mutable = False
        return combined

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_executor"] = None
        state["_executor_pid"] = None
        state["_batch_space_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Views pickled before these attributes existed.
        self.__dict__.setdefault("_executor", None)
        self.__dict__.setdefault("_executor_pid", None)
        self.__dict__.setdefault("_batch_space_cache", {})

    def _build_index_cache(self) -> None:
        """
        Build a cache of start and end indices for each batch
        """
        index_caches = self.backend.zeros(
            (len(self.batches), 3), 
            dtype=self.backend.default_integer_dtype,
            device=self.device
        )
        start_idx = 0
        for i, batch in enumerate(self.batches):
            batch_size = len(batch)
            end_idx = start_idx + batch_size
            index_caches = self.backend.at(index_caches)[i].set(
                self.backend.asarray([start_idx, end_idx, batch_size], dtype=self.backend.default_integer_dtype, device=self.device)
            )
            start_idx = end_idx
        self.index_caches = index_caches # (len(self.batches), 3) where the second dimension is start, end, and size of each batch
    
    def __len__(self) -> int:
        return self.index_caches[-1, 1]

    def _convert_single_index(self, idx : int) -> Tuple[int, int]:
        return convert_single_index_to_batch(
            self.backend,
            self.index_caches[:, 0],
            len(self),
            idx,
            device=self.device,
        )

    def _convert_index(self, idx : Union[IndexableType, BArrayType]) -> Tuple[
        int, 
        List[
            Tuple[int, BArrayType, BArrayType]
        ]
    ]:
        return convert_index_to_batch(
            self.backend,
            self.index_caches[:, 0],
            len(self),
            idx,
            device=self.device,
        )

    def get_flattened_at(self, idx : Union[IndexableType, BaseException]) -> BArrayType:
        if isinstance(idx, int):
            batch_idx, index_into_batch = self._convert_single_index(idx)
            return self.batches[batch_idx].get_flattened_at(index_into_batch)
        else:
            batch_size, batch_list = self._convert_index(idx)
            
            result = None

            if self.use_thread_pool and len(batch_list) > 1:
                data_futures = []
                executor = self._get_executor()
                for batch_index, index_into_batch, mask in batch_list:
                    data_futures.append(executor.submit(self.batches[batch_index].get_flattened_at, index_into_batch))
                for data_future, (batch_index, index_into_batch, mask) in zip(data_futures, batch_list):
                    batch_result = data_future.result()
                    if result is None:
                        result = self.backend.zeros(
                            (batch_size, *batch_result.shape[1:]),
                            dtype=batch_result.dtype,
                            device=self.backend.device(batch_result)
                        )
                    result = self.backend.at(result)[mask].set(batch_result)
            else:
                for batch_index, index_into_batch, mask in batch_list:
                    batch_result = self.batches[batch_index].get_flattened_at(index_into_batch)
                    if result is None:
                        result = self.backend.zeros(
                            (batch_size, *batch_result.shape[1:]),
                            dtype=batch_result.dtype,
                            device=self.backend.device(batch_result)
                        )
                    result = self.backend.at(result)[mask].set(batch_result)
            return result
    
    def get_flattened_at_with_metadata(self, idx : Union[IndexableType, BaseException]) -> Tuple[BArrayType, Dict[str, Any]]:
        if isinstance(idx, int):
            batch_idx, index_into_batch = self._convert_single_index(idx)
            dat, metadata = self.batches[batch_idx].get_flattened_at_with_metadata(index_into_batch)
            if metadata is None:
                metadata = {}
            metadata['batch_index'] = self.backend.full(
                (),
                batch_idx,
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )
            return dat, metadata
        else:
            batch_size, batch_list = self._convert_index(idx)
            
            result = None
            if self.batches[0].single_metadata_space is None:
                metadata_space = None
                metadata = {}
            else:
                metadata_space = self._cached_batch_space(self.batches[0].single_metadata_space, batch_size)
                metadata = metadata_space.create_empty()
            
            batch_index_arr = self.backend.zeros(
                (batch_size,),
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )

            if self.use_thread_pool and len(batch_list) > 1:
                data_futures = []
                executor = self._get_executor()
                for batch_index, index_into_batch, mask in batch_list:
                    data_futures.append(executor.submit(self.batches[batch_index].get_flattened_at_with_metadata, index_into_batch))
                for data_future, (batch_index, index_into_batch, mask) in zip(data_futures, batch_list):
                    batch_result, metadata_result = data_future.result()
                    if result is None:
                        result = self.backend.zeros(
                            (batch_size, *batch_result.shape[1:]),
                            dtype=batch_result.dtype,
                            device=self.device
                        )

                    result = self.backend.at(result)[mask].set(batch_result)
                    if metadata_space is not None:
                        metadata = sbu.set_at(
                            metadata_space,
                            metadata,
                            mask,
                            metadata_result,
                        )
                    batch_index_arr = self.backend.at(batch_index_arr)[mask].set(batch_index)
            else:
                for batch_index, index_into_batch, mask in batch_list:
                    batch_result, metadata_result = self.batches[batch_index].get_flattened_at_with_metadata(index_into_batch)
                    if result is None:
                        result = self.backend.zeros(
                            (batch_size, *batch_result.shape[1:]),
                            dtype=batch_result.dtype,
                            device=self.device
                        )

                    result = self.backend.at(result)[mask].set(batch_result)
                    if metadata_space is not None:
                        metadata = sbu.set_at(
                            metadata_space,
                            metadata,
                            mask,
                            metadata_result,
                        )
                    batch_index_arr = self.backend.at(batch_index_arr)[mask].set(batch_index)
            
            metadata['batch_index'] = batch_index_arr
            return result, metadata
    
    def get_at(self, idx : Union[IndexableType, BArrayType]) -> BatchT:
        if isinstance(idx, int):
            batch_idx, index_into_batch = self._convert_single_index(idx)
            return self.batches[batch_idx].get_at(index_into_batch)
        else:
            batch_size, batch_list = self._convert_index(idx)
            result_space = self._cached_batch_space(self.single_space, batch_size)
            result = result_space.create_empty()
            if self.use_thread_pool and len(batch_list) > 1:
                data_futures = []
                executor = self._get_executor()
                for batch_index, index_into_batch, mask in batch_list:
                    data_futures.append(executor.submit(self.batches[batch_index].get_at, index_into_batch))
                for data_future, (batch_index, index_into_batch, mask) in zip(data_futures, batch_list):
                    result = sbu.set_at(
                        result_space,
                        result,
                        mask,
                        data_future.result(),
                    )
            else:
                for batch_index, index_into_batch, mask in batch_list:
                    result = sbu.set_at(
                        result_space,
                        result,
                        mask,
                        self.batches[batch_index].get_at(index_into_batch),
                    )
            return result

    def get_at_with_metadata(self, idx : Union[IndexableType, BArrayType]) -> Tuple[BatchT, Dict[str, Any]]:
        if isinstance(idx, int):
            batch_idx, index_into_batch = self._convert_single_index(idx)
            dat, metadata = self.batches[batch_idx].get_at_with_metadata(index_into_batch)
            if metadata is None:
                metadata = {}
            metadata['batch_index'] = self.backend.full(
                (),
                batch_idx,
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )
            return dat, metadata
        else:
            batch_size, batch_list = self._convert_index(idx)
            result_space = self._cached_batch_space(self.single_space, batch_size)
            result = result_space.create_empty()
            
            if self.batches[0].single_metadata_space is None:
                metadata_space = None
                metadata = {}
            else:
                metadata_space = self._cached_batch_space(self.batches[0].single_metadata_space, batch_size)
                metadata = metadata_space.create_empty()
            
            batch_index_arr = self.backend.zeros(
                (batch_size,),
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )

            if self.use_thread_pool and len(batch_list) > 1:
                data_futures = []
                executor = self._get_executor()
                for batch_index, index_into_batch, mask in batch_list:
                    data_futures.append(executor.submit(self.batches[batch_index].get_at_with_metadata, index_into_batch))
                for data_future, (batch_index, index_into_batch, mask) in zip(data_futures, batch_list):
                    batch_result, metadata_result = data_future.result()
                    result = sbu.set_at(
                        result_space,
                        result,
                        mask,
                        batch_result,
                    )
                    if metadata_space is not None:
                        metadata = sbu.set_at(
                            metadata_space,
                            metadata,
                            mask,
                            metadata_result,
                        )
                    batch_index_arr = self.backend.at(batch_index_arr)[mask].set(batch_index)
            else:
                for batch_index, index_into_batch, mask in batch_list:
                    batch_result, metadata_result = self.batches[batch_index].get_at_with_metadata(index_into_batch)
                    result = sbu.set_at(
                        result_space,
                        result,
                        mask,
                        batch_result,
                    )
                    if metadata_space is not None:
                        metadata = sbu.set_at(
                            metadata_space,
                            metadata,
                            mask,
                            metadata_result,
                        )
                    batch_index_arr = self.backend.at(batch_index_arr)[mask].set(batch_index)
            
            metadata['batch_index'] = batch_index_arr
            return result, metadata

    def set_flattened_at(self, idx : Union[IndexableType, BArrayType], value : BArrayType) -> None:
        assert self.is_mutable, "Batch is not mutable"
        if isinstance(idx, int):
            batch_idx, index_into_batch = self._convert_single_index(idx)
            self.batches[batch_idx].set_flattened_at(index_into_batch, value)
        else:
            batch_size, batch_list = self._convert_index(idx)
            assert batch_size == value.shape[0], f"Target batch size of {batch_size} does not match value size of {value.shape[0]}"
            for batch_index, index_into_batch, mask in batch_list:
                self.batches[batch_index].set_flattened_at(index_into_batch, value[mask])
    
    def extend_flattened(self, value : BArrayType) -> None:
        assert self.is_mutable, "Batch is not mutable"
        self.batches[-1].extend_flattened(value)
        self._build_index_cache()
    
    def set_at(self, idx : Union[IndexableType, BArrayType], value : BatchT) -> None:
        assert self.is_mutable, "Batch is not mutable"
        if isinstance(idx, int):
            batch_idx, index_into_batch = self._convert_single_index(idx)
            self.batches[batch_idx].set_at(index_into_batch, value)
        else:
            batch_size, batch_list = self._convert_index(idx)
            value_size = sbu.batch_size_data(value)
            assert batch_size == value_size, f"Target batch size of {batch_size} does not match value size of {value_size}"
            for batch_index, index_into_batch, mask in batch_list:
                self.batches[batch_index].set_at(index_into_batch, sbu.get_at(self._batched_space, value, mask))

    def remove_at(self, idx : Union[IndexableType, BArrayType]) -> None:
        assert self.is_mutable, "Batch is not mutable"
        if isinstance(idx, int):
            batch_idx, index_into_batch = self._convert_single_index(idx)
            self.batches[batch_idx].remove_at(index_into_batch)
        else:
            batch_size, batch_list = self._convert_index(idx)
            for batch_index, index_into_batch, mask in batch_list:
                self.batches[batch_index].remove_at(index_into_batch)
        self._build_index_cache()
    
    def extend(self, value : BatchT) -> None:
        assert self.is_mutable, "Batch is not mutable"
        self.batches[-1].extend(value)
        self._build_index_cache()
    
    def close(self) -> None:
        """Release this view. The child batches are not owned by the view and stay open;
        close them through whoever created it (``BatchBase.__del__`` calls ``close()``, so
        closing the children here would tear it down whenever a view is garbage-collected)."""
        executor = getattr(self, "_executor", None)
        if executor is not None and getattr(self, "_executor_pid", None) == os.getpid():
            executor.shutdown(wait=False)
        self._executor = None