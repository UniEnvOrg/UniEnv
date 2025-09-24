from typing import Dict, Any, Optional, Tuple, Union, Generic, TypeVar
import numpy as np
import copy
import dataclasses

from unienv_interface.space import Space
from unienv_interface.space import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

DataT = TypeVar('DataT')

@dataclasses.dataclass(frozen=True)
class SpaceDataQueueState(Generic[DataT]):
    data: DataT # (L, B, ...) or (L, ...)

    def replace(self, **changes: Any) -> 'SpaceDataQueueState':
        return dataclasses.replace(self, **changes)

class FuncSpaceDataQueue(
    Generic[DataT, BArrayType, BDeviceType, BDtypeType, BRNGType]
):
    def __init__(
        self,
        space : Space[DataT, BDeviceType, BDtypeType, BRNGType],
        batch_size : Optional[int],
        maxlen: int,
    ) -> None:
        assert maxlen > 0, "Max length must be greater than 0"
        assert batch_size is None or batch_size > 0, "Batch size must be greater than 0 if provided"
        assert batch_size is None or sbu.batch_size(space) == batch_size, "Batch size must match the space's batch size if provided"
        self.single_space = space
        self.stacked_space = sbu.batch_space(space, maxlen) # (L, ...) or (L, B, ...)
        self.output_space = sbu.swap_batch_dims(
            self.stacked_space, 0, 1
        ) if batch_size is not None else self.stacked_space # (B, L, ...) or (L, ...)
        self._maxlen = maxlen
        self._batch_size = batch_size

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def backend(self) -> ComputeBackend:
        return self.single_space.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.single_space.device
    
    def init(
        self,
        initial_data : DataT,
    ) -> SpaceDataQueueState:
        return self.reset(
            SpaceDataQueueState(self.stacked_space.create_empty()),
            initial_data
        )

    def reset(
        self, 
        state : SpaceDataQueueState,
        initial_data : DataT,
        mask : Optional[BArrayType] = None,
    ) -> SpaceDataQueueState:
        assert self.batch_size is None or mask is None, \
            "Mask should not be provided if batch size is empty"
        index = (
            slice(None), mask
        ) if mask is not None else slice(None)
        
        expanded_data = sbu.get_at( # Add a singleton horizon dimension to the data
            self.single_space,
            initial_data,
            None
        )
        return state.replace(
            data=sbu.set_at(
                self.stacked_space,
                state.data,
                index,
                expanded_data
            )
        )

    def add(self, state : SpaceDataQueueState, data : DataT) -> SpaceDataQueueState:
        new_data = self.backend.map_fn_over_arrays(
            state.data,
            lambda x: self.backend.roll(x, shift=-1, axis=0),
        )
        new_data = sbu.set_at(
            self.stacked_space,
            new_data,
            -1,
            data
        )
        return state.replace(data=new_data)
    
    def get_output_data(self, state : SpaceDataQueueState) -> DataT:
        if self.batch_size is None:
            return state.data
        else:
            return sbu.swap_batch_dims_in_data(
                self.backend,
                state.data,
                0, 1
            ) # (L, B, ...) -> (B, L, ...)

class SpaceDataQueue(
    Generic[DataT, BArrayType, BDeviceType, BDtypeType, BRNGType]
):
    def __init__(
        self,
        space : Space[DataT, BDeviceType, BDtypeType, BRNGType],
        batch_size : Optional[int],
        maxlen: int,
    ) -> None:
        self.func_queue = FuncSpaceDataQueue(
            space,
            batch_size,
            maxlen
        )
        self.state = None

    @property
    def single_space(self) -> Space[DataT, BDeviceType, BDtypeType, BRNGType]:
        return self.func_queue.single_space

    @property
    def stacked_space(self) -> Space[DataT, BDeviceType, BDtypeType, BRNGType]:
        return self.func_queue.stacked_space

    @property
    def output_space(self) -> Space[DataT, BDeviceType, BDtypeType, BRNGType]:
        return self.func_queue.output_space
    
    @property
    def maxlen(self) -> int:
        return self.func_queue.maxlen

    @property
    def batch_size(self) -> Optional[int]:
        return self.func_queue.batch_size
    

    @property
    def backend(self) -> ComputeBackend:
        return self.func_queue.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.func_queue.device

    def reset(
        self,
        initial_data : DataT,
        mask : Optional[BArrayType] = None,
    ) -> None:
        if self.state is None:
            assert mask is None, "Mask should not be provided on the first reset"
            self.state = self.func_queue.init(initial_data)
        else:
            self.state = self.func_queue.reset(
                self.state,
                initial_data,
                mask
            )
        
    def add(self, data : DataT) -> None:
        assert self.state is not None, "Data queue must be reset before adding data"
        self.state = self.func_queue.add(
            self.state,
            data
        )

    def get_output_data(self) -> DataT:
        assert self.state is not None, "Data queue must be reset before getting output data"
        return self.func_queue.get_output_data(
            self.state
        )