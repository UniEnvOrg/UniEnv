import typing
from copy import deepcopy
from functools import singledispatch
from typing import Optional, Any, Iterable, Iterator
import numpy as np
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space.spaces import *

__all__ = [
    "is_flattenable",
    "flat_dim",
    "flatten_space",
    "flatten_data",
    "unflatten_data",
]

def is_flattenable(space: Space, start_dim : int = 0) -> bool:
    return flat_dim(space, start_dim) is not None

@singledispatch
def flat_dim(space : Space, start_dim : int = 0) -> Optional[int]:
    return None

@flat_dim.register(BoxSpace)
@flat_dim.register(BinarySpace)
def _flat_dim_common(space: typing.Union[BoxSpace, BinarySpace], start_dim : int = 0) -> int:
    if start_dim < -len(space.shape) or start_dim > len(space.shape):
        return None
    return int(np.prod(space.shape[start_dim:]))

@flat_dim.register(DynamicBoxSpace)
def _flat_dim_dynamic_box(space: DynamicBoxSpace, start_dim : int = 0) -> Optional[int]:
    if start_dim < -len(space.shape_high) or start_dim > len(space.shape_high):
        return None
    return int(np.prod(space.shape_high[start_dim:]))

@flat_dim.register(DictSpace)
def _flat_dim_dict(space: DictSpace, start_dim : int = 0) -> Optional[int]:
    dims = 0
    for key, subspace in space.spaces.items():
        dim = flat_dim(subspace, start_dim)
        if dim is None:
            return None
        dims += dim
    return dims

@flat_dim.register(TupleSpace)
def _flat_dim_tuple(space: TupleSpace, start_dim : int = 0) -> Optional[int]:
    dims = 0
    for subspace in space.spaces:
        dim = flat_dim(subspace, start_dim)
        if dim is None:
            return None
        dims += dim
    return dims

@flat_dim.register(TextSpace)
def _flat_dim_text(space: TextSpace, start_dim : int = 0) -> int:
    if start_dim != 0 or space.charset is None:
        return None
    return space.max_length

@flat_dim.register(UnionSpace)
def _flat_dim_oneof(space: UnionSpace, start_dim : int = 0) -> Optional[int]:
    if start_dim != 0:
        return None
    max_dim = 0
    for subspace in space.spaces:
        dim = flat_dim(subspace, start_dim)
        if dim is None:
            return None
        max_dim = max(max_dim, dim)
    return max_dim + 1

@singledispatch
def flatten_space(space: Space, start_dim : int = 0) -> BoxSpace:
    raise NotImplementedError(f"Unknown space: `{space}`")

@flatten_space.register(BoxSpace)
def _flatten_space_box(space: BoxSpace, start_dim : int = 0) -> BoxSpace:
    assert -len(space.shape) <= start_dim <= len(space.shape)
    low = space.low
    high = space.high
    return BoxSpace(
        space.backend,
        low=space.backend.reshape(
            low, low.shape[:start_dim] + (-1,)
        ), 
        high=space.backend.reshape(
            high, high.shape[:start_dim] + (-1,)
        ), 
        dtype=space.dtype,
        device=space.device
    )

@flatten_space.register(DynamicBoxSpace)
def _flatten_space_dynamic_box(space: DynamicBoxSpace, start_dim : int = 0) -> BoxSpace:
    assert -len(space.shape_high) <= start_dim <= len(space.shape_high)
    low = space.get_low(space.shape_high)
    high = space.get_high(space.shape_high)
    return BoxSpace(
        space.backend,
        low=space.backend.reshape(
            low, low.shape[:start_dim] + (-1,)
        ),
        high=space.backend.reshape(
            high, high.shape[:start_dim] + (-1,)
        ),
        dtype=space.dtype,
        device=space.device
    )

@flatten_space.register(BinarySpace)
def _flatten_space_binary(space: BinarySpace, start_dim : int = 0) -> BoxSpace:
    assert -len(space.shape) <= start_dim <= len(space.shape)
    return BoxSpace(
        space.backend,
        low=0, high=1, 
        shape=space.shape[:start_dim] + (flat_dim(space, start_dim),), 
        dtype=space.backend.default_integer_dtype,
        device=space.device
    )

@flatten_space.register(TupleSpace)
@flatten_space.register(DictSpace)
def _flatten_space_composition(space: typing.Union[TupleSpace, DictSpace], start_dim : int = 0) -> BoxSpace:
    space_list = [flatten_space(s, start_dim) for s in space.spaces] if isinstance(space, TupleSpace) else [flatten_space(s, start_dim) for s in space.spaces.values()]
    flat_low = space.backend.concat([s.low for s in space_list], axis=start_dim)
    flat_high = space.backend.concat([s.high for s in space_list], axis=start_dim)

    return BoxSpace(
        space.backend,
        low=flat_low,
        high=flat_high,
        dtype=space.backend.result_type(flat_low, flat_high),
        device=space.device
    )

@flatten_space.register(TextSpace)
def _flatten_space_text(space: TextSpace, start_dim : int = 0) -> BoxSpace:
    if start_dim != 0 or space.charset is None:
        raise ValueError("Text space can only be flattened at start_dim 0 with a defined character set.")
    
    return BoxSpace(
        space.backend,
        low=0, high=len(space.charset), 
        shape=(space.max_length,), 
        dtype=space.backend.default_integer_dtype,
        device=space.device
    )

@flatten_space.register(UnionSpace)
def _flatten_space_oneof(space: UnionSpace, start_dim : int = 0) -> BoxSpace:
    if start_dim != 0:
        return None
    
    num_subspaces = len(space.spaces)
    max_flatdim = max(flat_dim(s, start_dim) for s in space.spaces) + 1

    space_list = [flatten_space(s, start_dim) for s in space.spaces]
    lows = space.backend.asarray([space.backend.min(s.low) for s in space_list])
    highs = space.backend.asarray([space.backend.max(s.high) for s in space_list])

    overall_low = space.backend.min(lows)
    overall_high = space.backend.max(highs)

    low = space.backend.concat([
        space.backend.zeros(1), 
        space.backend.full(max_flatdim - 1, overall_low)
    ])
    high = space.backend.concat([
        space.backend.asarray([num_subspaces - 1]), 
        space.backend.full(max_flatdim - 1, overall_high)
    ])

    return BoxSpace(
        space.backend,
        low=low, high=high, 
        dtype=space.backend.default_floating_dtype,
        device=space.device
    )

@singledispatch
def flatten_data(space : Space, data : Any, start_dim : int = 0) -> BArrayType:
    raise NotImplementedError(f"Flattening not supported for space {space}")

@singledispatch
def unflatten_data(space : Space, data : BArrayType, start_dim : int = 0) -> Any:
    raise NotImplementedError(f"Unflattening not supported for space {space}")

@flatten_data.register(BoxSpace)
@flatten_data.register(BinarySpace)
def _flatten_data_common(space: typing.Union[BoxSpace, BinarySpace], data: BArrayType, start_dim : int = 0) -> BArrayType:
    assert -len(space.shape) <= start_dim <= len(space.shape)
    return space.backend.reshape(data, data.shape[:start_dim] + (-1,))

@unflatten_data.register(BoxSpace)
@unflatten_data.register(BinarySpace)
def _unflatten_data_common(space: typing.Union[BoxSpace, BinarySpace], data: Any, start_dim : int = 0) -> BArrayType:
    assert -len(space.shape) <= start_dim <= len(space.shape)
    unflat_dat = space.backend.reshape(data, data.shape[:start_dim] + space.shape[start_dim:])
    unflat_dat = space.backend.astype(unflat_dat, space.dtype)
    return unflat_dat

@flatten_data.register(DynamicBoxSpace)
def _flatten_data_dynamic_box(space: DynamicBoxSpace, data: BArrayType, start_dim : int = 0) -> BArrayType:
    assert -len(space.shape_high) <= start_dim <= len(space.shape_high)
    padded_data = space.pad_data(data, start_axis=start_dim)
    reshaped_data = space.backend.reshape(padded_data, padded_data.shape[:start_dim] + (-1,))
    return reshaped_data

@unflatten_data.register(DynamicBoxSpace)
def _unflatten_data_dynamic_box(space: DynamicBoxSpace, data: BArrayType, start_dim : int = 0) -> BArrayType:
    assert -len(space.shape_high) <= start_dim <= len(space.shape_high)
    padded_data = space.backend.reshape(data, data.shape[:start_dim] + space.shape_high[start_dim:])
    data = space.unpad_data(padded_data, start_axis=start_dim)
    return data

@flatten_data.register(DictSpace)
def _flatten_data_dict(space: DictSpace, data: typing.Dict[str, typing.Any], start_dim : int = 0) -> BArrayType:
    to_concat = []
    for key, subspace in space.spaces.items():
        flat_sample = flatten_data(subspace, data[key], start_dim)
        flat_dim_data = flat_dim(subspace, start_dim)
        assert flat_sample.shape[-1] == flat_dim_data, f"Flattened data dimension mismatch for key `{key}`: {flat_sample.shape[-1]} != {flat_dim_data}, original data {data[key].shape}, original space {subspace}"
        if flat_sample is not None:
            to_concat.append(flat_sample)
    return space.backend.concat(to_concat, axis=start_dim)

@unflatten_data.register(DictSpace)
def _unflatten_data_dict(space: DictSpace, data: Any, start_dim : int = 0) -> Any:
    result = {}
    start = 0
    for key, subspace in space.spaces.items():
        end = start + flat_dim(subspace, start_dim)
        part_idx = space.backend.arange(start, end, dtype=space.backend.default_integer_dtype, device=space.backend.device(data))
        part_data = space.backend.take(data, part_idx, axis=start_dim)
        result[key] = unflatten_data(subspace, part_data, start_dim)
        start = end
    return result

@flatten_data.register(TextSpace)
def _flatten_data_text(space: TextSpace, data: str, start_dim : int = 0) -> BArrayType:
    assert start_dim == 0 and space.charset is not None, "Text space can only be flattened at start_dim 0 with a defined character set."

    pad_size = space.max_length - len(data)
    data = space.backend.asarray(
        [space.character_index(c) for c in data], 
        dtype=space.backend.default_integer_dtype
    )
    if pad_size > 0:
        padding = space.backend.full(
            pad_size, 
            len(space.charset), 
            dtype=space.backend.default_integer_dtype
        )
        data = space.backend.concat(
            [data, padding]
        )
    return data

@unflatten_data.register(TextSpace)
def _unflatten_data_text(space: TextSpace, data: BArrayType, start_dim : int = 0) -> str:
    assert start_dim == 0 and space.charset is not None, "Text space can only be unflattened at start_dim 0 with a defined character set."
    return "".join([
        space.charset_list[int(data[i])] for i in data.shape[0] if data[i] < len(space.charset_list)
    ])

@flatten_data.register(TupleSpace)
def _flatten_data_tuple(space: TupleSpace, data: typing.Tuple, start_dim : int = 0) -> BArrayType:
    return space.backend.concat([
        flatten_data(subspace, data[i], start_dim) for i, subspace in enumerate(space.spaces)
    ], axis=start_dim)

@unflatten_data.register(TupleSpace)
def _unflatten_data_tuple(space: TupleSpace, data: BArrayType, start_dim : int = 0) -> typing.Tuple:
    result = []
    start = 0
    for subspace in space.spaces:
        end = start + flat_dim(subspace, start_dim)
        part_idx = space.backend.arange(start, end, dtype=space.backend.default_integer_dtype, device=space.backend.device(data))
        part_data = space.backend.take(data, part_idx, axis=start_dim)
        result.append(unflatten_data(subspace, part_data, start_dim))
        start = end
    return tuple(result)

@flatten_data.register(UnionSpace)
def _flatten_data_oneof(space: UnionSpace, data: typing.Tuple[int, Any], start_dim : int = 0) -> BArrayType:
    assert start_dim == 0
    space_idx, space_data = data
    flat_sample = flatten_data(space.spaces[space_idx], space_data, start_dim)
    padding_size = flat_dim(space, start_dim=start_dim) - len(flat_sample)
    if padding_size > 0:
        padding = space.backend.zeros(padding_size, dtype=space.dtype)
        flat_sample = space.backend.concat((flat_sample, padding))
    index_array = space.backend.full(1, space_idx)
    return space.backend.concat((index_array, flat_sample))

@unflatten_data.register(UnionSpace)
def _unflatten_data_oneof(space: UnionSpace, data: BArrayType, start_dim : int = 0) -> typing.Tuple[int, Any]:
    assert start_dim == 0
    space_idx = data[0]
    subspace = space.spaces[space_idx]
    subspace_data = data[1:flat_dim(subspace)+1]
    return (space_idx, unflatten_data(subspace, subspace_data))

