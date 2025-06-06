from typing import Sequence, List, Tuple, Union, Dict, Any, Optional, Generic, TypeVar, Iterable, Iterator
from types import EllipsisType
import os
import abc
from xbarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.space import Space, DictSpace, BoxSpace
import dataclasses
import copy

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu

from ..base.common import BatchBase, IndexableType, BatchT

class CombinedBatch(BatchBase[
    BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    def __init__(
        self,
        batches : Sequence[BatchBase[
            BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
        ]],
        device : Optional[BDeviceType] = None,
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
            new_metadata_space = single_metadata_space.spaces.copy()
            self.original_single_metadata_space = single_metadata_space
            self.original_single_metadata_space_flat = sfu.flatten_space(
                single_metadata_space
            )
            self.original_batched_metadata_space = batches[0].batched_metadata_space
        else:
            new_metadata_space = {}
            self.original_single_metadata_space = None
            self.original_single_metadata_space_flat = None
            self.original_batched_metadata_space = None
        new_metadata_space['batch_index'] = BoxSpace(
            backend,
            low=0,
            high=len(batches),
            dtype=backend.default_integer_dtype,
            device=device,
            shape=(),
        )
        new_metadata_space = DictSpace(
            backend,
            new_metadata_space,
            device=device,
        )
        super().__init__(single_space, new_metadata_space)

        self.backend = backend
        self.device = device
        self.is_mutable = is_mutable
        self.batches = batches
        self._build_index_cache()

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
        self.index_caches = index_caches
    
    def __len__(self) -> int:
        return self.index_caches[-1, 1]

    def _convert_single_index(self, idx : int) -> Tuple[int, int]:
        """
        Convert a single index to a tuple containing
         - the batch index
         - the index within the batch
        """
        batch_index = int(self.backend.sum(
            idx >= self.index_caches[:, 0]
        ))
        return batch_index, idx - self.index_caches[batch_index, 0]

    def _convert_index(self, idx : Union[IndexableType, BArrayType]) -> Tuple[
        int, 
        List[
            Tuple[int, BArrayType, BArrayType]
        ]
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
            idx_array = self.backend.arange(
                *idx.indices(len(self)),
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )
        elif idx is Ellipsis:
            idx_array = self.backend.arange(
                len(self),
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )
        elif self.backend.is_backendarray(idx):
            assert len(idx.shape) == 1, "Index must be 1D"
            idx_array = idx
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")
        
        idx_array_positive = idx_array % len(self)
        idx_array_bigger = idx_array_positive[:, None] >= self.index_caches[None, :, 0] # (idx_array_shape, len(self.batches))
        idx_array_batch_idx = self.backend.sum(
            idx_array_bigger,
            axis=-1
        ) - 1 # (idx_array_shape, )
        
        result_batch_list = []
        batch_indexes = self.backend.unique_values(idx_array_batch_idx)
        for i in range(batch_indexes.shape[0]):
            batch_index = int(batch_indexes[i])
            result_mask = idx_array_batch_idx == batch_index
            index_into_batch = idx_array_positive[result_mask] - self.index_caches[batch_index, 0]
            result_batch_list.append((batch_index, index_into_batch, result_mask))
        return idx_array.shape[0], result_batch_list

    def get_flattened_at(self, idx : Union[IndexableType, BaseException]) -> BArrayType:
        if isinstance(idx, int):
            batch_idx, index_into_batch = self._convert_single_index(idx)
            return self.batches[batch_idx].get_flattened_at(index_into_batch)
        else:
            batch_size, batch_list = self._convert_index(idx)
            
            result = None

            for batch_index, index_into_batch, mask in batch_list:
                batch_result = self.batches[batch_index].get_flattened_at(index_into_batch)
                if result is None:
                    result = self.backend.zeros(
                        (batch_size, *batch_result.shape[1:]),
                        dtype=batch_result.dtype,
                        device=self.device
                    )
                result[mask] = batch_result
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
            metadata_arr = None
            if self.original_single_metadata_space_flat is not None:
                metadata_arr = self.backend.zeros(
                    (batch_size, *self.original_single_metadata_space_flat.shape),
                    dtype=self.original_single_metadata_space_flat.dtype,
                    device=self.device
                )
            batch_index_arr = self.backend.zeros(
                (batch_size,),
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )

            for batch_index, index_into_batch, mask in batch_list:
                batch_result, metadata = self.batches[batch_index].get_flattened_at_with_metadata(index_into_batch)
                if result is None:
                    result = self.backend.zeros(
                        (batch_size, *batch_result.shape[1:]),
                        dtype=batch_result.dtype,
                        device=self.device
                    )

                result[mask] = batch_result
                if metadata_arr is not None:
                    metadata_flat = sfu.flatten_data(
                        self.original_batched_metadata_space,
                        metadata,
                        start_dim=1,
                    )
                    metadata_arr[mask] = metadata_flat
                batch_index_arr[mask] = batch_index
            
            if metadata_arr is not None:
                metadata = sfu.unflatten_data(
                    self.original_batched_metadata_space,
                    metadata_arr,
                    start_dim=1,
                )
            else:
                metadata = {}
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
            assert batch_size == len(value), f"Target batch size of {batch_size} does not match value size of {len(value)}"
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
        for batch in self.batches:
            batch.close()
        self.batches = []