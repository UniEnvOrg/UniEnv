from typing import List, Tuple, Union, Dict, Any, Optional, Generic, TypeVar
import os
import abc
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.space import Space, batch_utils as space_batch_utils, Dict as DictSpace, flatten_utils as space_flatten_utils
import dataclasses

BatchT = TypeVar('BatchT')
class BatchBase(abc.ABC, Generic[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device: Optional[BDeviceType] = None

    # If the batch is mutable, then the data can be changed (extend_*, set_*, etc.)
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
    def get_flattened_at(self, idx : Optional[Union[int, slice, BArrayType]] = None) -> BArrayType:
        raise NotImplementedError

    def set_flattened_at(self, idx : Optional[Union[int, slice, BArrayType]], value : BArrayType) -> None:
        raise NotImplementedError
    
    def extend_flattened(self, value : BArrayType) -> None:
        raise NotImplementedError
    
    def get_at(self, idx : Optional[Union[int, slice, BArrayType]] = None) -> BatchT:
        flattened_data = self.get_flattened_at(idx)
        if isinstance(idx, int):
            return space_flatten_utils.unflatten_data(self.single_space, flattened_data)
        else:
            return space_flatten_utils.batch_unflatten_data(self._batched_space, flattened_data)
    
    def __getitem__(self, idx : Union[int, slice, BArrayType]) -> BatchT:
        return self.get_at(idx)

    def set_at(self, idx : Optional[Union[int, slice, BArrayType]], value : BatchT) -> None:
        if isinstance(idx, int):
            flattened_data = space_flatten_utils.flatten_data(self.single_space, value)
        else:
            flattened_data = space_flatten_utils.batch_flatten_data(self._batched_space, value)
        self.set_flattened_at(idx, flattened_data)
    
    def __setitem__(self, idx : Union[int, slice, BArrayType], value : BatchT) -> None:
        self.set_at(idx, value)

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

    def sample_flat(self) -> SamplerArrayType:
        return space_flatten_utils.flatten_data(self.sampled_space, self.sample(), start_dim=1)

    @abc.abstractmethod
    def sample(self) -> SamplerBatchT:
        """
        Should sample a batch of transitions from the data.
        """
        raise NotImplementedError
    
    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()