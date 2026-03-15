from typing import List, Sequence, Tuple, Union, Dict, Any, Optional, Generic, TypeVar, Iterable, Iterator, Any
from types import EllipsisType
import os
import abc
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.space import Space, BoxSpace, DictSpace
import dataclasses

from functools import cached_property
from unienv_interface.space.space_utils import batch_utils as space_batch_utils, flatten_utils as space_flatten_utils

__all__ = [
    "BatchT",
    "BatchBase",
    "BatchSampler",
    "IndexableType",
    "convert_index_to_backendarray",
]

IndexableType = Union[int, slice, EllipsisType]

def convert_index_to_backendarray(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    index : IndexableType,
    length : int,
    device : Optional[BDeviceType] = None,
) -> BArrayType:
    """Normalize a Python index into a 1-D backend integer array.

    This helper lets batch implementations accept ``int``, ``slice``, and
    ``...`` while still delegating storage access through a single backend-aware
    indexing path.
    """
    if isinstance(index, int):
        return backend.asarray([index], dtype=backend.default_integer_dtype, device=device)
    elif isinstance(index, slice):
        return backend.arange(*index.indices(length), dtype=backend.default_integer_dtype, device=device)
    elif index is Ellipsis:
        return backend.arange(length, dtype=backend.default_integer_dtype, device=device)
    else:
        raise ValueError("Index must be an integer, slice, or Ellipsis.")

BatchT = TypeVar('BatchT')
class BatchBase(abc.ABC, Generic[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """Base interface for dataset-like collections of structured samples.

    ``BatchBase`` stores the per-sample space definition and provides a uniform
    API for retrieving either structured values or their flattened backend-array
    representation. Concrete subclasses decide where data lives and which
    mutation operations are supported.
    """
    # If the batch is mutable, then the data can be changed (extend_*, set_*, remove_*, etc.)
    is_mutable: bool = True

    def __init__(
        self,
        single_space : Space[BatchT, BDeviceType, BDtypeType, BRNGType],
        single_metadata_space : Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]] = None,
    ):
        """Initialize the batch contract for one logical sample."""
        self.single_space = single_space
        self.single_metadata_space = single_metadata_space

    # For backwards compatibility
    @cached_property
    def _batched_space(self) -> Space[BatchT, BDeviceType, BDtypeType, BRNGType]:
        return space_batch_utils.batch_space(self.single_space, 1)
    
    @cached_property
    def _batched_metadata_space(self) -> Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]:
        if self.single_metadata_space is not None:
            return space_batch_utils.batch_space(self.single_metadata_space, 1)
        else:
            return None

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.single_space.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.single_space.device

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of addressable samples in the batch."""
        raise NotImplementedError
    
    def get_flattened_at(self, idx : Union[IndexableType, BArrayType]) -> BArrayType:
        """Fetch samples as flattened backend arrays."""
        unflattened_data = self.get_at(idx)
        if isinstance(idx, int):
            return space_flatten_utils.flatten_data(self.single_space, unflattened_data)
        else:
            return space_flatten_utils.flatten_data(self._batched_space, unflattened_data, start_dim=1)

    def get_flattened_at_with_metadata(
        self, idx : Union[IndexableType, BArrayType]
    ) -> Tuple[BArrayType, Optional[Dict[str, Any]]]:
        """Fetch flattened samples together with optional per-sample metadata."""
        unflattened_data, metadata = self.get_at_with_metadata(idx)
        if isinstance(idx, int):
            return space_flatten_utils.flatten_data(self.single_space, unflattened_data), metadata
        else:
            return space_flatten_utils.flatten_data(self._batched_space, unflattened_data, start_dim=1), metadata

    def set_flattened_at(self, idx : Union[IndexableType, BArrayType], value : BArrayType) -> None:
        """Overwrite existing samples using flattened data."""
        raise NotImplementedError

    def append_flattened(self, value : BArrayType) -> None:
        """Append one flattened sample to the batch."""
        return self.extend_flattened(value[None])

    def extend_flattened(self, value : BArrayType) -> None:
        """Append a batched block of flattened samples."""
        unflat_data = space_flatten_utils.unflatten_data(self._batched_space, value, start_dim=1)
        self.extend(unflat_data)
    
    def get_at(self, idx : Union[IndexableType, BArrayType]) -> BatchT:
        """Fetch samples in their structured form."""
        flattened_data = self.get_flattened_at(idx)
        if isinstance(idx, int):
            return space_flatten_utils.unflatten_data(self.single_space, flattened_data)
        else:
            return space_flatten_utils.unflatten_data(self._batched_space, flattened_data, start_dim=1)
    
    def get_at_with_metadata(
        self, idx : Union[IndexableType, BArrayType]
    ) -> Tuple[BatchT, Optional[Dict[str, Any]]]:
        """Fetch structured samples together with optional metadata."""
        flattened_data, metadata = self.get_flattened_at_with_metadata(idx)
        if isinstance(idx, int):
            return space_flatten_utils.unflatten_data(self.single_space, flattened_data), metadata
        else:
            return space_flatten_utils.unflatten_data(self._batched_space, flattened_data, start_dim=1), metadata

    def __getitem__(self, idx : Union[IndexableType, BArrayType]) -> BatchT:
        return self.get_at(idx)

    def set_at(self, idx : Union[IndexableType, BArrayType], value : BatchT) -> None:
        """Overwrite existing samples using structured data."""
        if isinstance(idx, int):
            flattened_data = space_flatten_utils.flatten_data(self.single_space, value)
        else:
            flattened_data = space_flatten_utils.flatten_data(self._batched_space, value, start_dim=1)
        self.set_flattened_at(idx, flattened_data)
    
    def __setitem__(self, idx : Union[IndexableType, BArrayType], value : BatchT) -> None:
        self.set_at(idx, value)
    
    def remove_at(self, idx : Union[IndexableType, BArrayType]) -> None:
        """Remove one or more samples from the batch."""
        raise NotImplementedError

    def __delitem__(self, idx : Union[IndexableType, BArrayType]) -> None:
        self.remove_at(idx)

    def append(self, value : BatchT) -> None:
        """Append one structured sample to the batch."""
        batched_data = space_batch_utils.concatenate(self._batched_space, [value])
        self.extend(batched_data)

    def extend(self, value : BatchT) -> None:
        """Append a batched block of structured samples."""
        flattened_data = space_flatten_utils.flatten_data(self._batched_space, value, start_dim=1)
        self.extend_flattened(flattened_data)

    def extend_from(
        self, 
        other : 'BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]',
        chunk_size : int = 8,
        tqdm : bool = False,
    ) -> None:
        """Copy data from another batch in bounded-size chunks."""
        n_total = len(other)
        iterable_start = range(0, n_total, chunk_size)
        if tqdm:
            from tqdm import tqdm
            iterable_start = tqdm(iterable_start, desc="Extending Batch")
        for start_idx in iterable_start:
            end_idx = min(start_idx + chunk_size, n_total)
            data_chunk = other.get_at(slice(start_idx, end_idx))
            self.extend(data_chunk)
    
    def get_slice(self, idx : Union[IndexableType, BArrayType]) -> 'BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]':
        """Create a lazy view over a subset of indices."""
        from unienv_data.batches.subindex_batch import SubIndexedBatch
        return SubIndexedBatch(self, idx)

    def get_column(self, nested_keys : Sequence[str]) -> 'BatchBase[Any, BArrayType, BDeviceType, BDtypeType, BRNGType]':
        """Create a lazy view over a nested field inside each sample."""
        from unienv_data.batches.subitem_batch import SubItemBatch
        return SubItemBatch(self, nested_keys)

    def close(self) -> None:
        pass

    def __del__(self):
        self.close()

SamplerBatchT = TypeVar('SamplerBatchT')
SamplerArrayType = TypeVar('SamplerArrayType')
SamplerDeviceType = TypeVar('SamplerDeviceType')
SamplerDtypeType = TypeVar('SamplerDtypeType')
SamplerRNGType = TypeVar('SamplerRNGType')
class BatchSampler(
    Generic[
        SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType,
    ],
    BatchBase[
        SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType
    ]
):
    """Read-only batch wrapper that samples mini-batches from another batch."""
    data : BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]

    rng : Optional[SamplerRNGType] = None
    data_rng : Optional[BRNGType] = None
    
    is_mutable : bool = False

    def __init__(
        self,
        single_space : Space[BatchT, BDeviceType, BDtypeType, BRNGType],
        single_metadata_space : Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]] = None,
        batch_size : int = 1,
    ) -> None:
        """Configure the sampling batch shape without binding the source data yet."""
        super().__init__(single_space=single_space, single_metadata_space=single_metadata_space)
        self.batch_size = batch_size
    
    def manual_seed(self, seed : int) -> None:
        """Reset sampler RNG state, including the optional data-index RNG."""
        if self.rng is not None:
            self.rng = self.backend.random.random_number_generator(seed, device=self.device)
        if self.data_rng is not None:
            self.data_rng = self.backend.random.random_number_generator(seed, device=self.data.device)

    @cached_property
    def _batched_space(self) -> Space[BatchT, BDeviceType, BDtypeType, BRNGType]:
        return space_batch_utils.batch_space(self.single_space, self.batch_size)
    
    @cached_property
    def _batched_metadata_space(self) -> Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]:
        if self.single_metadata_space is not None:
            return space_batch_utils.batch_space(self.single_metadata_space, self.batch_size)
        else:
            return None
    
    @property
    def sampled_space(self) -> Space[SamplerBatchT, SamplerDeviceType, SamplerDtypeType, SamplerRNGType]:
        return self._batched_space
    
    @property
    def sampled_metadata_space(self) -> Optional[DictSpace[SamplerDeviceType, SamplerDtypeType, SamplerRNGType]]:
        return self._batched_metadata_space

    def __len__(self):
        return len(self.data)

    def sample_index(self) -> SamplerArrayType:
        """Draw random indices for one batch."""
        new_rng, indices = self.backend.random.random_discrete_uniform( # (B, )
            (self.batch_size,),
            0,
            len(self.data),
            rng=self.data_rng if self.data_rng is not None else self.rng,
            device=self.data.device,
        )
        if self.data_rng is not None:
            self.data_rng = new_rng
        else:
            self.rng = new_rng
        return indices

    def sample_flat(self) -> SamplerArrayType:
        """Sample a batch and return it in flattened form."""
        idx = self.sample_index()
        return self.get_flattened_at(idx)
    
    def sample_flat_with_metadata(self) -> Tuple[SamplerArrayType, Optional[Dict[str, Any]]]:
        """Sample a flattened batch together with optional metadata."""
        idx = self.sample_index()
        return self.get_flattened_at_with_metadata(idx)

    def sample(self) -> SamplerBatchT:
        """Sample a batch in its structured form."""
        idx = self.sample_index()
        return self.get_at(idx)
    
    def sample_with_metadata(self) -> Tuple[SamplerBatchT, Optional[Dict[str, Any]]]:
        """Sample a structured batch together with optional metadata."""
        idx = self.sample_index()
        return self.get_at_with_metadata(idx)

    def __iter__(self) -> Iterator[SamplerBatchT]:
        return self.epoch_iter()
    
    def epoch_iter(self) -> Iterator[SamplerBatchT]:
        """Iterate once over a random permutation of the source data."""
        if self.data_rng is not None:
            self.data_rng, idx = self.backend.random.random_permutation(len(self.data), rng=self.data_rng, device=self.data.device)
        else:
            self.rng, idx = self.backend.random.random_permutation(len(self.data), rng=self.rng, device=self.data.device)
        n_batches = len(self.data) // self.batch_size
        num_left = len(self.data) % self.batch_size
        for i in range(n_batches):
            yield self.get_at(idx[i*self.batch_size:(i+1)*self.batch_size])
        if num_left > 0:
            yield self.get_at(idx[-num_left:])
    
    def epoch_iter_with_metadata(self) -> Iterator[Tuple[SamplerBatchT, Optional[Dict[str, Any]]]]:
        """Iterate once over shuffled structured batches plus metadata."""
        if self.data_rng is not None:
            self.data_rng, idx = self.backend.random.random_permutation(len(self.data), rng=self.data_rng, device=self.data.device)
        else:
            self.rng, idx = self.backend.random.random_permutation(len(self.data), rng=self.rng, device=self.data.device)
        n_batches = len(self.data) // self.batch_size
        num_left = len(self.data) % self.batch_size
        for i in range(n_batches):
            yield self.get_at_with_metadata(idx[i*self.batch_size:(i+1)*self.batch_size])
        if num_left > 0:
            yield self.get_at_with_metadata(idx[-num_left:])

    def epoch_flat_iter(self) -> Iterator[SamplerArrayType]:
        """Iterate once over shuffled batches in flattened form."""
        if self.data_rng is not None:
            self.data_rng, idx = self.backend.random.random_permutation(len(self.data), rng=self.data_rng, device=self.data.device)
        else:
            self.rng, idx = self.backend.random.random_permutation(len(self.data), rng=self.rng, device=self.data.device)
        n_batches = len(self.data) // self.batch_size
        num_left = len(self.data) % self.batch_size
        for i in range(n_batches):
            yield self.get_flattened_at(idx[i*self.batch_size:(i+1)*self.batch_size])
        if num_left > 0:
            yield self.get_flattened_at(idx[-num_left:])
    
    def epoch_flat_iter_with_metadata(self) -> Iterator[Tuple[SamplerArrayType, Optional[Dict[str, Any]]]]:
        """Iterate once over shuffled flattened batches plus metadata."""
        if self.data_rng is not None:
            self.data_rng, idx = self.backend.random.random_permutation(len(self.data), rng=self.data_rng, device=self.data.device)
        else:
            self.rng, idx = self.backend.random.random_permutation(len(self.data), rng=self.rng, device=self.data.device)
        n_batches = len(self.data) // self.batch_size
        num_left = len(self.data) % self.batch_size
        for i in range(n_batches):
            yield self.get_flattened_at_with_metadata(idx[i*self.batch_size:(i+1)*self.batch_size])
        if num_left > 0:
            yield self.get_flattened_at_with_metadata(idx[-num_left:])

    def close(self) -> None:
        pass
