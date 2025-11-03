from typing import List, Tuple, Union, Dict, Any, Optional, Generic, TypeVar, Iterable, Iterator
from types import EllipsisType
import os
import abc
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.space import Space, BoxSpace, DictSpace
import dataclasses

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
    # If the batch is mutable, then the data can be changed (extend_*, set_*, remove_*, etc.)
    is_mutable: bool = True

    def __init__(
        self,
        single_space : Space[BatchT, BDeviceType, BDtypeType, BRNGType],
        single_metadata_space : Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]] = None,
    ):
        self.single_space = single_space
        self.single_metadata_space = single_metadata_space
        self._batched_space : Space[BatchT, BDeviceType, BDtypeType, BRNGType] = space_batch_utils.batch_space(single_space, 1)
        if single_metadata_space is not None:
            self._batched_metadata_space : DictSpace[
                BDeviceType, BDtypeType, BRNGType
            ] = space_batch_utils.batch_space(single_metadata_space, 1)
        else:
            self._batched_metadata_space = None

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.single_space.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.single_space.device

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
    
    def get_flattened_at(self, idx : Union[IndexableType, BArrayType]) -> BArrayType:
        unflattened_data = self.get_at(idx)
        if isinstance(idx, int):
            return space_flatten_utils.flatten_data(self.single_space, unflattened_data)
        else:
            return space_flatten_utils.flatten_data(self._batched_space, unflattened_data, start_dim=1)

    def get_flattened_at_with_metadata(
        self, idx : Union[IndexableType, BArrayType]
    ) -> Tuple[BArrayType, Optional[Dict[str, Any]]]:
        unflattened_data, metadata = self.get_at_with_metadata(idx)
        if isinstance(idx, int):
            return space_flatten_utils.flatten_data(self.single_space, unflattened_data), metadata
        else:
            return space_flatten_utils.flatten_data(self._batched_space, unflattened_data, start_dim=1), metadata

    def set_flattened_at(self, idx : Union[IndexableType, BArrayType], value : BArrayType) -> None:
        raise NotImplementedError

    def append_flattened(self, value : BArrayType) -> None:
        return self.extend_flattened(value[None])

    def extend_flattened(self, value : BArrayType) -> None:
        unflat_data = space_flatten_utils.unflatten_data(self._batched_space, value, start_dim=1)
        self.extend(unflat_data)
    
    def get_at(self, idx : Union[IndexableType, BArrayType]) -> BatchT:
        flattened_data = self.get_flattened_at(idx)
        if isinstance(idx, int):
            return space_flatten_utils.unflatten_data(self.single_space, flattened_data)
        else:
            return space_flatten_utils.unflatten_data(self._batched_space, flattened_data, start_dim=1)
    
    def get_at_with_metadata(
        self, idx : Union[IndexableType, BArrayType]
    ) -> Tuple[BatchT, Optional[Dict[str, Any]]]:
        flattened_data, metadata = self.get_flattened_at_with_metadata(idx)
        if isinstance(idx, int):
            return space_flatten_utils.unflatten_data(self.single_space, flattened_data), metadata
        else:
            return space_flatten_utils.unflatten_data(self._batched_space, flattened_data, start_dim=1), metadata

    def __getitem__(self, idx : Union[IndexableType, BArrayType]) -> BatchT:
        return self.get_at(idx)

    def set_at(self, idx : Union[IndexableType, BArrayType], value : BatchT) -> None:
        if isinstance(idx, int):
            flattened_data = space_flatten_utils.flatten_data(self.single_space, value)
        else:
            flattened_data = space_flatten_utils.flatten_data(self._batched_space, value, start_dim=1)
        self.set_flattened_at(idx, flattened_data)
    
    def __setitem__(self, idx : Union[IndexableType, BArrayType], value : BatchT) -> None:
        self.set_at(idx, value)
    
    def remove_at(self, idx : Union[IndexableType, BArrayType]) -> None:
        raise NotImplementedError

    def __delitem__(self, idx : Union[IndexableType, BArrayType]) -> None:
        self.remove_at(idx)

    def append(self, value : BatchT) -> None:
        batched_data = space_batch_utils.concatenate(self._batched_space, [value])
        self.extend(batched_data)

    def extend(self, value : BatchT) -> None:
        flattened_data = space_flatten_utils.flatten_data(self._batched_space, value, start_dim=1)
        self.extend_flattened(flattened_data)

    def close(self) -> None:
        pass

    def __del__(self) -> None:
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
        super().__init__(single_space=single_space, single_metadata_space=single_metadata_space)
        self.batch_size = batch_size
        self._batched_space : Space[SamplerBatchT, SamplerDeviceType, SamplerDtypeType, SamplerRNGType] = space_batch_utils.batch_space(self.single_space, batch_size)
        self._batched_metadata_space : Optional[DictSpace[SamplerDeviceType, SamplerDtypeType, SamplerRNGType]] = space_batch_utils.batch_space(self.single_metadata_space, batch_size) if self.single_metadata_space is not None else None

    def manual_seed(self, seed : int) -> None:
        if self.rng is not None:
            self.rng = self.backend.random.random_number_generator(seed, device=self.device)
        if self.data_rng is not None:
            self.data_rng = self.backend.random.random_number_generator(seed, device=self.data.device)

    @property
    def sampled_space(self) -> Space[SamplerBatchT, SamplerDeviceType, SamplerDtypeType, SamplerRNGType]:
        return self._batched_space
    
    @property
    def sampled_metadata_space(self) -> Optional[DictSpace[SamplerDeviceType, SamplerDtypeType, SamplerRNGType]]:
        return self._batched_metadata_space

    def __len__(self):
        return len(self.data)

    def sample_index(self) -> SamplerArrayType:
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
        idx = self.sample_index()
        return self.get_flattened_at(idx)
    
    def sample_flat_with_metadata(self) -> Tuple[SamplerArrayType, Optional[Dict[str, Any]]]:
        idx = self.sample_index()
        return self.get_flattened_at_with_metadata(idx)

    def sample(self) -> SamplerBatchT:
        idx = self.sample_index()
        return self.get_at(idx)
    
    def sample_with_metadata(self) -> Tuple[SamplerBatchT, Optional[Dict[str, Any]]]:
        idx = self.sample_index()
        return self.get_at_with_metadata(idx)

    def __iter__(self) -> Iterator[SamplerBatchT]:
        return self.epoch_iter()
    
    def epoch_iter(self) -> Iterator[SamplerBatchT]:
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

    def __del__(self) -> None:
        self.close()