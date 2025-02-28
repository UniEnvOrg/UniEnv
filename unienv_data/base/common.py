from typing import List, Tuple, Union, Dict, Any, Optional, Generic, TypeVar, Iterable, Iterator
from types import EllipsisType
import os
import abc
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.space import Space, batch_utils as space_batch_utils, Dict as DictSpace, flatten_utils as space_flatten_utils
import dataclasses

IndexableType = Union[int, slice, EllipsisType]

BatchT = TypeVar('BatchT')
class BatchBase(abc.ABC, Generic[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device: Optional[BDeviceType] = None

    # If the batch is mutable, then the data can be changed (extend_*, set_*, remove_*, etc.)
    is_mutable: bool = True

    def __init__(
        self,
        single_space : Space[Any, Any, BDeviceType, BDtypeType, BRNGType],
    ):
        self.single_space = single_space
        self._batched_space : Space[
            BatchT, Any, BDeviceType, BDtypeType, BRNGType
        ] = space_batch_utils.batch_space(single_space, 1)

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_flattened_at(self, idx : Union[IndexableType, BArrayType]) -> BArrayType:
        raise NotImplementedError

    def set_flattened_at(self, idx : Union[IndexableType, BArrayType], value : BArrayType) -> None:
        raise NotImplementedError

    def extend_flattened(self, value : BArrayType) -> None:
        raise NotImplementedError
    
    def get_at(self, idx : Union[IndexableType, BArrayType]) -> BatchT:
        flattened_data = self.get_flattened_at(idx)
        if isinstance(idx, int):
            return space_flatten_utils.unflatten_data(self.single_space, flattened_data)
        else:
            return space_flatten_utils.batch_unflatten_data(self._batched_space, flattened_data)
    
    def __getitem__(self, idx : Union[IndexableType, BArrayType]) -> BatchT:
        return self.get_at(idx)

    def set_at(self, idx : Union[IndexableType, BArrayType], value : BatchT) -> None:
        if isinstance(idx, int):
            flattened_data = space_flatten_utils.flatten_data(self.single_space, value)
        else:
            flattened_data = space_flatten_utils.batch_flatten_data(self._batched_space, value)
        self.set_flattened_at(idx, flattened_data)
    
    def __setitem__(self, idx : Union[IndexableType, BArrayType], value : BatchT) -> None:
        self.set_at(idx, value)
    
    def remove_at(self, idx : Union[IndexableType, BArrayType]) -> None:
        raise NotImplementedError

    def __delitem__(self, idx : Union[IndexableType, BArrayType]) -> None:
        self.remove_at(idx)

    def extend(self, value : BatchT) -> None:
        flattened_data = space_flatten_utils.batch_flatten_data(self._batched_space, value)
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
class BatchSampler(abc.ABC, Generic[
    SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType,
    BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType,
]):
    batch_size : int
    sampled_space : Space[SamplerBatchT, Any, SamplerDeviceType, SamplerDtypeType, SamplerRNGType]
    sampled_space_flat : Space[SamplerArrayType, Any, SamplerDeviceType, SamplerDtypeType, SamplerRNGType]

    backend : ComputeBackend[SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType]
    device : Optional[SamplerDeviceType] = None

    data : BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]

    rng : Optional[SamplerRNGType] = None
    data_rng : Optional[BRNGType] = None

    @abc.abstractmethod
    def get_flat_at(self, idx : SamplerArrayType) -> SamplerArrayType:
        raise NotImplementedError
    
    def get_at(self, idx : SamplerArrayType) -> SamplerBatchT:
        return space_flatten_utils.unflatten_data(self.sampled_space, self.get_flat_at(idx), start_dim=1)

    def sample_index(self) -> SamplerArrayType:
        new_rng, indices = self.backend.random_discrete_uniform( # (B, )
            self.data_rng if self.data_rng is not None else self.rng,
            (self.batch_size,),
            0,
            len(self.data),
            device=self.data.device,
        )
        if self.data_rng is not None:
            self.data_rng = new_rng
        else:
            self.rng = new_rng
        return indices

    def sample_flat(self) -> SamplerArrayType:
        idx = self.sample_index()
        return self.get_flat_at(idx)

    def sample(self) -> SamplerBatchT:
        idx = self.sample_index()
        return self.get_at(idx)
    
    def __iter__(self) -> Iterator[SamplerBatchT]:
        return self.epoch_iter()
    
    def epoch_iter(self) -> Iterator[SamplerBatchT]:
        if self.data_rng is not None:
            self.data_rng, idx = self.backend.random_permutation(self.data_rng, len(self.data), device=self.data.device)
        else:
            self.rng, idx = self.backend.random_permutation(self.rng, len(self.data), device=self.data.device)
        n_batches = len(self.data) // self.batch_size
        num_left = len(self.data) % self.batch_size
        for i in range(n_batches):
            yield self.get_at(idx[i*self.batch_size:(i+1)*self.batch_size])
        if num_left > 0:
            yield self.get_at(idx[-num_left:])
    
    def epoch_flat_iter(self) -> Iterator[SamplerArrayType]:
        if self.data_rng is not None:
            self.data_rng, idx = self.backend.random_permutation(self.data_rng, len(self.data), device=self.data.device)
        else:
            self.rng, idx = self.backend.random_permutation(self.rng, len(self.data), device=self.data.device)
        n_batches = len(self.data) // self.batch_size
        num_left = len(self.data) % self.batch_size
        for i in range(n_batches):
            yield self.get_flat_at(idx[i*self.batch_size:(i+1)*self.batch_size])
        if num_left > 0:
            yield self.get_flat_at(idx[-num_left:])

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()