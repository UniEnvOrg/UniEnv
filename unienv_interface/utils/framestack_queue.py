from typing import Dict, Any, Optional, Tuple, Union, Generic, TypeVar
import numpy as np
import copy
import dataclasses

from unienv_interface.space import Space, BoxSpace, BinarySpace, DictSpace, TupleSpace
from unienv_interface.space import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

DataT = TypeVar('DataT')

# Backends whose arrays are immutable, so ``sbu.set_at`` already returns new buffers.
_IMMUTABLE_BACKENDS = ("jax",)
# Backends whose arrays support ``empty_like`` + in-place slice assignment (fast path).
_FAST_PATH_BACKENDS = ("numpy", "pytorch")

_LEAF = "leaf"

def _fast_plan(space : Space) -> Optional[Any]:
    """
    Nested plan (``_LEAF`` / dict / tuple) mirroring ``space`` when every leaf is a plain
    Box / Binary space using the generic array ``set_at``; ``None`` otherwise.
    """
    if type(space) is DictSpace:
        plans = {key: _fast_plan(subspace) for key, subspace in space.spaces.items()}
        return None if any(plan is None for plan in plans.values()) else plans
    if type(space) is TupleSpace:
        plans = tuple(_fast_plan(subspace) for subspace in space.spaces)
        return None if any(plan is None for plan in plans) else plans
    if isinstance(space, (BoxSpace, BinarySpace)) and sbu.set_at.dispatch(type(space)) is sbu.set_at.dispatch(BoxSpace):
        return _LEAF
    return None

def _shift_append(plan : Any, stacked : Any, value : Any) -> Any:
    """``roll(stacked, -1, axis=0)`` then ``[-1] = value``, written into one fresh buffer."""
    if plan == _LEAF: # compare by value: the plan may come back from a pickle
        out = np.empty_like(stacked) if isinstance(stacked, np.ndarray) else stacked.new_empty(stacked.shape)
        out[:-1] = stacked[1:]
        out[-1] = value
        return out
    if isinstance(plan, dict):
        return {key: _shift_append(subplan, stacked[key], value[key]) for key, subplan in plan.items()}
    return tuple(_shift_append(subplan, stacked[i], value[i]) for i, subplan in enumerate(plan))

def _swap_first_two_dims(plan : Any, data : Any) -> Any:
    if plan == _LEAF: # compare by value: the plan may come back from a pickle
        return data.swapaxes(0, 1) if isinstance(data, np.ndarray) else data.transpose(0, 1)
    if isinstance(plan, dict):
        return {key: _swap_first_two_dims(subplan, data[key]) for key, subplan in plan.items()}
    return tuple(_swap_first_two_dims(subplan, data[i]) for i, subplan in enumerate(plan))

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
        # numpy / torch with plain Box / Binary leaves: ``add`` and ``get_output_data``
        # skip the generic roll + ``at[].set`` and the space-dispatched transposes.
        self._fast_plan = _fast_plan(space) if space.backend.simplified_name in _FAST_PATH_BACKENDS else None

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
        assert self.batch_size is not None or mask is None, \
            "Mask should not be provided if batch size is empty"
        index = (
            slice(None), mask
        ) if mask is not None else slice(None)
        
        expanded_data = sbu.get_at( # Add a singleton horizon dimension to the data
            self.single_space,
            initial_data,
            None
        )
        # ``sbu.set_at`` writes in place for numpy / torch, and ``state.data`` is
        # shared with observations already handed out by ``get_output_data`` (and
        # with the caller's state in the functional API). Write into a fresh buffer.
        if mask is None:
            data = self.stacked_space.create_empty()
        elif self.backend.simplified_name in _IMMUTABLE_BACKENDS:
            data = state.data
        else:
            data = self.backend.map_fn_over_arrays(
                state.data,
                lambda x: self.backend.asarray(x, copy=True),
            )
        return state.replace(
            data=sbu.set_at(
                self.stacked_space,
                data,
                index,
                expanded_data
            )
        )

    def add(self, state : SpaceDataQueueState, data : DataT) -> SpaceDataQueueState:
        if self._fast_plan is not None:
            return state.replace(data=_shift_append(self._fast_plan, state.data, data))
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
        elif self._fast_plan is not None:
            return _swap_first_two_dims(self._fast_plan, state.data) # (L, B, ...) -> (B, L, ...)
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