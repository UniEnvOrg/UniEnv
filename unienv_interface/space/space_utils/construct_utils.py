import typing
from copy import deepcopy
from functools import singledispatch
from typing import Optional, Any, Iterable, Iterator, Sequence, Tuple, Literal, Mapping
import numpy as np
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space.spaces import *

__all__ = [
    'construct_space_from_data_stream',
    'construct_space_from_data',
]

def construct_space_from_data_stream(
    data : Iterable[Any],
    backend : ComputeBackend,
    add_bounds : bool = True
) -> Space:
    """Construct a space from a stream of data.

    Args:
        data (Iterable[Any]): An iterable stream of data samples.
        backend (ComputeBackend): The compute backend to use for array operations.

    Returns:
        Space: The constructed space.
    """
    candidate = None
    for d in data:
        candidate = construct_space_from_data(d, backend, candidate, add_bounds=add_bounds)
    return candidate

def construct_space_from_data(
    data : Any,
    backend : ComputeBackend,
    candidate_space : Optional[Space] = None,
    add_bounds : bool = False
) -> Space:
    if backend.is_backendarray(data) and backend.dtype_is_boolean(data.dtype):
        assert candidate_space is None or (
            isinstance(candidate_space, BinarySpace) and
            candidate_space.shape == data.shape and
            (candidate_space.dtype is None or candidate_space.dtype == data.dtype) and
            (candidate_space.device is None or candidate_space.device == backend.device(data))
        )

        return BinarySpace(
            backend,
            data.shape,
            dtype=data.dtype,
            device=backend.device(data)
        )
    elif backend.is_backendarray(data):
        assert candidate_space is None or (
            isinstance(candidate_space, (BoxSpace, DynamicBoxSpace)) and
            candidate_space.dtype == data.dtype and
            (candidate_space.device is None or candidate_space.device == backend.device(data))
        )
        if candidate_space is None:
            return BoxSpace(
                backend,
                low=data if add_bounds else -backend.inf,
                high=data if add_bounds else backend.inf,
                shape=data.shape,
                dtype=data.dtype,
                device=backend.device(data)
            )
        elif isinstance(candidate_space, BoxSpace):
            if candidate_space.shape != data.shape:
                assert len(candidate_space.shape) == len(data.shape)
                new_low_shape, new_high_shape = _dynamic_box_find_shape(
                    candidate_space.shape,
                    candidate_space.shape,
                    data.shape
                )
                broadcast_shape = _dynamic_box_get_broadcast_shape(new_low_shape, new_high_shape)

                return DynamicBoxSpace(
                    backend,
                    *(_dynamic_box_update_bounds(
                        backend,
                        broadcast_shape,
                        candidate_space._low,
                        candidate_space._high,
                        data
                    ) if add_bounds else (candidate_space._low, candidate_space._high)),
                    shape_low=new_low_shape,
                    shape_high=new_high_shape,
                    dtype=data.dtype,
                    device=backend.device(data)
                )
        else: # elif isinstance(candidate_space, DynamicBoxSpace):
            new_low_shape, new_high_shape = _dynamic_box_find_shape(
                candidate_space.shape_low,
                candidate_space.shape_high,
                data.shape
            )
            broadcast_shape = _dynamic_box_get_broadcast_shape(new_low_shape, new_high_shape)

            return DynamicBoxSpace(
                backend,
                *(_dynamic_box_update_bounds(
                    backend,
                    broadcast_shape,
                    candidate_space._low,
                    candidate_space._high,
                    data
                ) if add_bounds else (candidate_space._low, candidate_space._high)),
                shape_low=new_low_shape,
                shape_high=new_high_shape,
                dtype=data.dtype,
                device=backend.device(data)
            )
    elif isinstance(data, Mapping):
        assert candidate_space is None or (
            isinstance(candidate_space, DictSpace)
            and set(candidate_space.spaces.keys()) == set(data.keys())
        )
        if candidate_space is None:
            spaces = {k: construct_space_from_data(v, backend) for k, v in data.items()}
        else:
            spaces = {
                k: construct_space_from_data(v, backend, candidate_space.spaces[k])
                for k, v in data.items()
            }
        return DictSpace(backend, spaces)
    elif isinstance(data, str):
        assert candidate_space is None or isinstance(candidate_space, TextSpace)
        max_length = len(data)
        if candidate_space is not None:
            max_length = max(max_length, candidate_space.max_length)
        if not add_bounds:
            max_length = max(max_length, 4096) # Arbitrary large length if not adding bounds
        return TextSpace(
            backend,
            max_length=max_length
        )
    elif isinstance(data, Sequence):
        assert candidate_space is None or (
            isinstance(candidate_space, TupleSpace)
            and len(candidate_space.spaces) == len(data)
        )
        if candidate_space is None:
            spaces = [construct_space_from_data(d, backend) for d in data]
        else:
            spaces = [
                construct_space_from_data(d, backend, candidate_space.spaces[i])
                for i, d in enumerate(data)
            ]
        return TupleSpace(backend, spaces)
    else:
        raise ValueError(f"Unsupported data type for space construction: {type(data)}")

def _dynamic_box_find_shape(
    shape_low : Sequence[int],
    shape_high : Sequence[int],
    data_shape : Sequence[int]
) -> Tuple[Sequence[int], Sequence[int]]:
    assert len(shape_low) == len(shape_high) == len(data_shape)
    new_shape_low = list(shape_low)
    new_shape_high = list(shape_high)
    for i in range(len(data_shape)):
        new_shape_low[i] = min(shape_low[i], data_shape[i])
        new_shape_high[i] = max(shape_high[i], data_shape[i])
    return new_shape_low, new_shape_high

def _dynamic_box_get_broadcast_shape(
    shape_low : Sequence[int],
    shape_high : Sequence[int],
) -> Sequence[int]:
    assert len(shape_low) == len(shape_high)
    broadcast_shape = []
    for low, high in zip(shape_low, shape_high):
        if low == high:
            broadcast_shape.append(low)
        else:
            broadcast_shape.append(1)  # Use 1 to indicate dynamic dimension
    return tuple(broadcast_shape)

def reshape_to_broadcastable(
    backend : ComputeBackend,
    target_shape : Sequence[int],
    array : BArrayType,
    method : Literal['min', 'max'] = 'min'
) -> BArrayType:
    if array.shape == target_shape:
        return array
    else:
        assert len(array.shape) == len(target_shape)
        for i, (t_dim, a_dim) in enumerate(zip(target_shape, array.shape)):
            assert t_dim == a_dim or t_dim == 1 or a_dim == 1
            if t_dim == a_dim or a_dim == 1:
                continue
            else:
                if method == 'min':
                    array = backend.min(array, axis=i, keepdims=True)
                else:
                    array = backend.max(array, axis=i, keepdims=True)
        return array

def _dynamic_box_update_bounds(
    backend : ComputeBackend,
    target_shape : Sequence[int],
    current_low : BArrayType,
    current_high : BArrayType,
    new_data : BArrayType
) -> Tuple[BArrayType, BArrayType]:
    new_low = backend.minimum(
        reshape_to_broadcastable(backend, target_shape, current_low, method='min'),
        reshape_to_broadcastable(backend, target_shape, new_data, method='min')
    )
    new_high = backend.maximum(
        reshape_to_broadcastable(backend, target_shape, current_high, method='max'),
        reshape_to_broadcastable(backend, target_shape, new_data, method='max')
    )
    return new_low, new_high