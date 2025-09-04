from typing import Sequence, List, Tuple, Union, Dict, Any, Optional, Generic, TypeVar, Iterable, Iterator, Callable
from types import EllipsisType
import os
import abc
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.space import Space, DictSpace, BoxSpace, BinarySpace, TupleSpace
import dataclasses
import copy
import functools

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu

from ..base.common import BatchBase, IndexableType, BatchT

class SliceStackedBatch(BatchBase[
    BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    """
    A batch that stacks frames with given fixed offsets.
    This is a read-only batch, since it is a view of the original batch. If you want to change the data, you should mutate the containing batch instead.
    """

    is_mutable = False

    def __init__(
        self,
        batch: BatchBase[
            BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
        ],
        fixed_offset : BArrayType,
        get_valid_mask_function : Optional[Callable[["SliceStackedBatch", BArrayType, BatchT, Dict[str, Any]], BArrayType]] = None,
        fill_invalid_data : bool = True,
        stack_metadata : bool = False,
    ):
        assert len(fixed_offset.shape) == 1, "Fixed offset must be a 1D tensor"
        assert fixed_offset.shape[0] > 0, "Fixed offset must have a positive length"
        assert batch.backend.any(fixed_offset == 0), "There should be at least one zero in the fixed offset"
        assert batch.backend.all(batch.backend.sort(fixed_offset) == fixed_offset), "Fixed offset must be sorted"
        self.batch = batch

        if batch.single_metadata_space is not None:
            metadata_space = sbu.batch_space(
                batch.single_metadata_space,
                fixed_offset.shape[0],
            ) if stack_metadata else copy.deepcopy(batch.single_metadata_space)
        else:
            metadata_space = DictSpace(
                batch.backend,
                {},
                device=batch.device,
            )
        
        metadata_space['slice_valid_mask'] = BinarySpace(
            batch.backend,
            shape=(fixed_offset.shape[0],),
            device=batch.device,
        )
        super().__init__(
            sbu.batch_space(
                batch.single_space,
                fixed_offset.shape[0],
            ),
            metadata_space,
        )

        # Store fixed offset
        if self.device is not None:
            fixed_offset = self.backend.to_device(fixed_offset, self.device)
        self.fixed_offset = fixed_offset
        self._idx_zero = int(batch.backend.argmax(
            batch.backend.astype(fixed_offset == 0, batch.backend.default_integer_dtype)
        ))

        self.fill_invalid_data = fill_invalid_data
        self.get_valid_mask_function = get_valid_mask_function
        self.stack_metadata = stack_metadata

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.batch.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.batch.device

    def __len__(self) -> int:
        return len(self.batch)

    def expand_index(self, index: BArrayType) -> BArrayType:
        """
        Expand indexes to slice the data
        Args:
            index (BArrayType): A 1D tensor of indices to expand. Can be boolean or integer.
        Returns:
            BArrayType: A 2D tensor of indices, where each row corresponds to the expanded indices for each batch item.
        """
        if isinstance(index, slice):
            index = self.backend.arange(
                *index.indices(len(self.batch)),
                device=self.device,
            )
        elif index is Ellipsis:
            index = self.backend.arange(
                0, len(self.batch),
                device=self.device,
            )
        
        assert self.backend.is_backendarray(index), f"Index must be a backend array, got {type(index)}"
        assert len(index.shape) == 1, f"Index must be a 1D tensor, got shape {index.shape}"
        assert self.backend.dtype_is_real_integer(index.dtype) or self.backend.dtype_is_boolean(index.dtype), \
            f"Index dtype must be an integer or boolean, got {index.dtype}"
        if self.backend.dtype_is_boolean(index.dtype):
            assert index.shape[0] == len(self.batch), f"Index shape {index.shape} does not match batch size {len(self.batch)}"
            index = self.backend.nonzero(index)[0] # (T, )

        # index_shifts = self.backend.arange( # (T, )
        #     -self.prefetch_horizon, self.postfetch_horizon + 1, dtype=index.dtype, device=self.backend.device(index)
        # )
        index = index[:, None] + self.fixed_offset[None, :] # (B, T)
        index = self.backend.clip(index, 0, len(self.batch) - 1)
        return index

    def get_valid_mask_flattened(
        self, 
        expanded_idx : BArrayType, 
        data : BArrayType,
        metadata : Dict[str, Any],
    ) -> BArrayType:
        """
        Get the valid mask for the flattened data.
        Args:
            expanded_idx (B, T): A 2D tensor of indices to slice the data.
            data (B, T, *D): The data to slice.
        Returns:
            Valid Mask (B, T)
        """
        if self.get_valid_mask_function is None:
            return self.backend.ones_like(expanded_idx, dtype=self.backend.default_boolean_dtype)
        else:
            return self.get_valid_mask(
                expanded_idx,
                sfu.unflatten_data(
                    self._batched_space,
                    data,
                    start_dim=2
                ),
                metadata
            )
    
    def get_valid_mask(
        self,
        expanded_idx : BArrayType,
        data : BatchT,
        metadata : Dict[str, Any]
    ) -> BArrayType:
        """
        Get the valid mask for the data.
        Args:
            expanded_idx (B, T): A 2D tensor of indices to slice the data.
            data (BatchT): The data to slice.
        Returns:
            Valid Mask (B, T)
        """
        if self.get_valid_mask_function is None:
            return self.backend.ones_like(expanded_idx, dtype=self.backend.default_boolean_dtype)
        else:
            return self.get_valid_mask_function(
                self,
                expanded_idx,
                data,
                metadata
            )

    def fill_data_with_stack_mask(
        self,
        space : Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]],
        data : Union[BArrayType, BatchT, Any],
        valid_mask : BArrayType
    ) -> Union[BArrayType, BatchT, Any]:
        """
        Fill the data with the mask as if the frames were frame-stacked.
        Args:
            space (Optional[Space]): The space to fill the data with. If None, the data is assumed to be a backend array.
            data (Union[BArrayType, BatchT, Any]): The data to fill sized (B, T, *D)
            valid_mask (BArrayType): The mask to fill the data with, sized (B, T)
        Returns:
            Union[BArrayType, BatchT, Any]: The filled data, sized (B, T, *D)
        """
        data_device = self.backend.device(data) if self.backend.is_backendarray(data) else self.device
        B, T = valid_mask.shape

        assert T == self.fixed_offset.shape[0], \
            f"Valid mask shape {T} does not match target offset size {self.fixed_offset.shape[0]}"

        first_valid_idx = self._idx_zero - self.backend.sum(
            valid_mask[:, :self._idx_zero], axis=1
        )
        last_valid_idx = self._idx_zero + self.backend.sum(
            valid_mask[:, self._idx_zero + 1:], axis=1
        )
        time_idx = self.backend.repeat(self.backend.arange(
            T, device=data_device
        )[None, :], B, axis=0)

        if self._idx_zero > 0:
            time_idx = self.backend.at(time_idx)[:, :self._idx_zero].set(self.backend.where(
                valid_mask[:, :self._idx_zero],
                time_idx[:, :self._idx_zero],
                first_valid_idx[:, None]
            ))
        if T - self._idx_zero - 1 > 0:
            time_idx = self.backend.at(time_idx)[:, self._idx_zero + 1:].set(self.backend.where(
                valid_mask[:, self._idx_zero + 1:],
                time_idx[:, self._idx_zero + 1:],
                last_valid_idx[:, None]
            ))
        
        batch_idx = self.backend.repeat(self.backend.arange(
            B, device=data_device
        )[:, None], T, axis=1)

        if space is None:
            filled_data = data[batch_idx, time_idx]
        else:
            filled_data = sbu.get_at(
                space,
                data,
                (batch_idx, time_idx)
            )
        return filled_data

    def get_flattened_at(self, idx):
        return self.get_flattened_at_with_metadata(idx)[0]

    def get_flattened_at_with_metadata(self, idx):
        if isinstance(idx, int):
            batched_result, batched_metadata = self.get_flattened_at_with_metadata(self.backend.full(
                (1,),
                idx,
                dtype=self.backend.default_integer_dtype,
                device=self.device
            ))
            single_result, single_metadata = batched_result[0], next(
                sbu.iterate(self._batched_metadata_space, batched_metadata)
            )
            return single_result, single_metadata
        
        expanded_idx = self.expand_index(idx)
        expanded_idx_flat = self.backend.reshape(
            expanded_idx, 
            (-1,)
        )
        batched_data_flat, metadata_flat = self.batch.get_flattened_at_with_metadata(expanded_idx_flat)
        batched_data = sbu.reshape_batch_size_in_data(
            self.backend,
            batched_data_flat,
            expanded_idx_flat.shape,
            expanded_idx.shape
        )
        metadata = sbu.reshape_batch_size_in_data(
            self.backend,
            metadata_flat,
            expanded_idx_flat.shape,
            expanded_idx.shape
        ) if metadata_flat is not None else None
        valid_mask = self.get_valid_mask_flattened(
            expanded_idx,
            batched_data,
            metadata
        )
        if self.fill_invalid_data:
            batched_data = self.fill_data_with_stack_mask(
                None,
                batched_data,
                valid_mask
            )
            if self.stack_metadata and metadata is not None:
                metadata = self.fill_data_with_stack_mask(
                    self._batched_metadata_space,
                    metadata,
                    valid_mask
                )
        if not self.stack_metadata and metadata is not None:
            metadata = sbu.get_at(
                self._batched_metadata_space,  # This does not necessarily have to be the space we have with metadata, as it is not temporally stacked
                metadata,
                (slice(None), self.prefetch_horizon)
            )

        metadata['slice_valid_mask'] = valid_mask
        return batched_data, metadata

    def get_at(self, idx : Union[IndexableType, BArrayType]) -> BatchT:
        return self.get_at_with_metadata(idx)[0]

    def get_at_with_metadata(self, idx : Union[IndexableType, BArrayType]) -> Tuple[BatchT, Dict[str, Any]]:
        if isinstance(idx, int):
            batched_result, batched_metadata = self.get_at_with_metadata(self.backend.full(
                (1,),
                idx,
                dtype=self.backend.default_integer_dtype,
                device=self.device
            ))
            single_result, single_metadata = next(zip(
                sbu.iterate(self._batched_space, batched_result),
                sbu.iterate(self._batched_metadata_space, batched_metadata)
            ))
            return single_result, single_metadata
        
        expanded_idx = self.expand_index(idx)
        expanded_idx_flat = self.backend.reshape(
            expanded_idx, 
            (-1,)
        )
        batched_data_flat, metadata_flat = self.batch.get_at_with_metadata(expanded_idx_flat)
        batched_data = sbu.reshape_batch_size_in_data(
            self.backend,
            batched_data_flat,
            expanded_idx_flat.shape,
            expanded_idx.shape
        )
        metadata = sbu.reshape_batch_size_in_data(
            self.backend,
            metadata_flat,
            expanded_idx_flat.shape,
            expanded_idx.shape
        ) if metadata_flat is not None else None
        valid_mask = self.get_valid_mask(
            expanded_idx,
            batched_data,
            metadata
        )
        if self.fill_invalid_data:
            if self.stack_metadata and metadata is not None:
                batched_data, metadata = self.fill_data_with_stack_mask(
                    TupleSpace(
                        self.backend, 
                        (self._batched_space, self._batched_metadata_space),
                        device=self.device
                    ),
                    (batched_data, metadata),
                    valid_mask
                )
            else:
                batched_data = self.fill_data_with_stack_mask(
                    self._batched_space,
                    batched_data,
                    valid_mask
                )
        if not self.stack_metadata and metadata is not None:
            metadata = sbu.get_at(
                self._batched_metadata_space, # This does not necessarily have to be the space we have with metadata, as it is not temporally stacked
                metadata,
                (slice(None), self.prefetch_horizon)
            )
        metadata = {} if metadata is None else metadata
        metadata['slice_valid_mask'] = valid_mask
        return batched_data, metadata

    def close(self) -> None:
        self.batch.close()

    # ========== Valid Mask Utilities ==========
    @staticmethod
    def _valid_mask_function_episodeid_key(
        batch : "SliceStackedBatch",
        expanded_idx : BArrayType,
        data : BatchT,
        metadata : Dict[str, Any],
        *,
        episode_id_key : Union[str, int],
        is_in_metadata : bool = False,
    ) -> BArrayType:
        if is_in_metadata:
            episode_ids = metadata[episode_id_key]
        else:
            episode_ids = data[episode_id_key]
        episode_id_at = episode_ids[:, batch._idx_zero]
        return episode_ids == episode_id_at[:, None]

    @staticmethod
    def get_valid_mask_function_with_episodeid_key(
        episode_id_key : Union[str, int] = "episode_id",
        is_in_metadata : bool = False,
    ) -> Callable[["SliceStackedBatch", BArrayType, BatchT], BArrayType]:
        return functools.partial(
            __class__._valid_mask_function_episodeid_key,
            episode_id_key=episode_id_key,
            is_in_metadata=is_in_metadata
        )

    @staticmethod
    def _valid_mask_function_episode_end_key(
        batch : "SliceStackedBatch",
        expanded_idx : BArrayType,
        data : BatchT,
        metadata : Dict[str, Any],
        *,
        episode_end_key : Union[str, int],
        is_in_metadata : bool = False,
    ) -> BArrayType:
        B, T = expanded_idx.shape

        if is_in_metadata:
            episode_ends = batch.backend.astype(metadata[episode_end_key], batch.backend.default_integer_dtype)
        else:
            episode_ends = batch.backend.astype(data[episode_end_key], batch.backend.default_integer_dtype)
        
        episode_ends_cs = batch.backend.cumulative_sum(episode_ends, axis=1)
        
        # We roll to get the cumulative sum before the current timestep (the number of previous episodes to get the current episode delta id)
        episode_ends_cs_before = batch.backend.roll(episode_ends_cs, shift=1, axis=1)
        episode_ends_cs_before[:, 0] = 0  # First element should be 0

        episode_ends_at = episode_ends_cs_before[:, batch._idx_zero]
        return episode_ends_cs_before == episode_ends_at[:, None]
    
    @staticmethod
    def get_valid_mask_function_with_episode_end_key(
        episode_end_key : Union[str, int] = "episode_end",
        is_in_metadata : bool = False,
    ) -> Callable[["SliceStackedBatch", BArrayType, BatchT], BArrayType]:
        return functools.partial(
            __class__._valid_mask_function_episode_end_key,
            episode_end_key=episode_end_key,
            is_in_metadata=is_in_metadata
        )
