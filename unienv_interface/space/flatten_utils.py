import typing
from copy import deepcopy
from functools import singledispatch
from typing import Optional, Any, Iterable, Iterator
from unienv_interface.space import Space, Box
import numpy as np

from unienv_interface.space import (
    Box,
    Dict,
    Discrete,
    Graph,
    MultiDiscrete,
    MultiBinary,
    Sequence,
    Space,
    Text,
    Tuple,
    Union
)

__ALL__ = [
    "is_flattenable",
    "flat_dim",
    "flatten_space"
    "flatten_data",
    "unflatten_data",
    "is_batch_flattenable",
    "batch_flat_dim",
    "batch_flatten_space",
    "batch_flatten_data",
    "batch_unflatten_data"
]

def is_flattenable(space: Space, start_dim : int = 0) -> bool:
    return flat_dim(space, start_dim) is not None

@singledispatch
def flat_dim(space : Space, start_dim : int = 0) -> Optional[int]:
    return None

@flat_dim.register(Box)
@flat_dim.register(MultiBinary)
@flat_dim.register(MultiDiscrete)
def _flat_dim_common(space: typing.Union[
    Box, MultiBinary, MultiDiscrete
], start_dim : int = 0) -> int:
    assert start_dim >= 0 and start_dim <= len(space.shape)
    return int(np.prod(space.shape[start_dim:]))

@flat_dim.register(Discrete)
def _flat_dim_discrete(space: Discrete, start_dim : int = 0) -> int:
    assert start_dim == 0
    return 1

@flat_dim.register(Dict)
def _flat_dim_dict(space: Dict, start_dim : int = 0) -> Optional[int]:
    dims = 0
    for key, subspace in space.spaces.items():
        dim = flat_dim(subspace, start_dim)
        if dim is None:
            return None
        dims += dim
    return dims

@flat_dim.register(Text)
def _flat_dim_text(space: Text, start_dim : int = 0) -> int:
    assert start_dim == 0
    return space.max_length

@flat_dim.register(Tuple)
def _flat_dim_tuple(space: Tuple, start_dim : int = 0) -> Optional[int]:
    dims = 0
    for subspace in space.spaces:
        dim = flat_dim(subspace, start_dim)
        if dim is None:
            return None
        dims += dim
    return dims

@flat_dim.register(Union)
def _flat_dim_oneof(space: Union, start_dim : int = 0) -> Optional[int]:
    assert start_dim == 0
    max_dim = 0
    for subspace in space.spaces:
        dim = flat_dim(subspace, start_dim)
        if dim is None:
            return None
        max_dim = max(max_dim, dim)
    return max_dim + 1

@singledispatch
def flatten_space(space: Space, start_dim : int = 0) -> Box:
    raise NotImplementedError(f"Unknown space: `{space}`")

@flatten_space.register(Box)
def _flatten_space_box(space: Box, start_dim : int = 0) -> Box:
    assert start_dim >= 0 and start_dim <= len(space.shape)
    return Box(
        space.backend,
        low=space.backend.array_api_namespace.reshape(
            space.low, space.low.shape[:start_dim] + (-1,)
        ), 
        high=space.backend.array_api_namespace.reshape(
            space.high, space.high.shape[:start_dim] + (-1,)
        ), 
        dtype=space.dtype,
        device=space.device
    )

@flatten_space.register(Discrete)
@flatten_space.register(MultiBinary)
@flatten_space.register(MultiDiscrete)
def _flatten_space_binary(space: Discrete | MultiBinary | MultiDiscrete, start_dim : int = 0) -> Box:
    return Box(
        space.backend,
        low=0, high=1, 
        shape=space.shape[:start_dim] + (flat_dim(space, start_dim),), 
        dtype=space.dtype or space.backend.default_integer_dtype,
        device=space.device
    )

@flatten_space.register(Tuple)
@flatten_space.register(Dict)
def _flatten_space_tuple(space: Tuple | Dict, start_dim : int = 0) -> Box:
    assert space.is_flattenable
    space_list = [flatten_space(s, start_dim) for s in space.spaces] if isinstance(space, Tuple) else [flatten_space(s, start_dim) for s in space.spaces.values()]

    return Box(
        space.backend,
        low=space.backend.array_api_namespace.concat([s.low for s in space_list], axis=start_dim),
        high=space.backend.array_api_namespace.concat([s.high for s in space_list], axis=start_dim),
        dtype=space.backend.default_floating_dtype,
        device=space.device
    )

@flatten_space.register(Text)
def _flatten_space_text(space: Text, start_dim : int = 0) -> Box:
    assert start_dim == 0
    return Box(
        space.backend,
        low=0, high=len(space.character_set), 
        shape=(space.max_length,), 
        dtype=space.backend.default_integer_dtype,
        device=space.device
    )

@flatten_space.register(Union)
def _flatten_space_oneof(space: Union, start_dim : int = 0) -> Box:
    assert start_dim == 0
    num_subspaces = len(space.spaces)
    max_flatdim = max(flat_dim(s) for s in space.spaces) + 1

    lows = space.backend.array_api_namespace.asarray([space.backend.array_api_namespace.min(flatten_space(s).low) for s in space.spaces])
    highs = space.backend.array_api_namespace.asarray([space.backend.array_api_namespace.max(flatten_space(s).high) for s in space.spaces])

    overall_low = space.backend.array_api_namespace.min(lows)
    overall_high = space.backend.array_api_namespace.max(highs)

    low = space.backend.array_api_namespace.concat([
        space.backend.array_api_namespace.zeros(1), 
        space.backend.array_api_namespace.full(max_flatdim - 1, overall_low)
    ])
    high = space.backend.array_api_namespace.concat([
        space.backend.array_api_namespace.asarray([num_subspaces - 1]), 
        space.backend.array_api_namespace.full(max_flatdim - 1, overall_high)
    ])

    return Box(
        backend=space.backend,
        low=low, high=high, 
        dtype=space.backend.default_floating_dtype,
        device=space.device
    )

@singledispatch
def flatten_data(space : Space, data : Any, start_dim : int = 0) -> Any:
    raise NotImplementedError(f"Flattening not supported for space {space}")

@singledispatch
def unflatten_data(space : Space, data : Any, start_dim : int = 0) -> Any:
    raise NotImplementedError(f"Unflattening not supported for space {space}")

@flatten_data.register(Box)
@flatten_data.register(Discrete)
@flatten_data.register(MultiBinary)
@flatten_data.register(MultiDiscrete)
def _flatten_data_common(space: typing.Union[
    Box, Discrete, MultiBinary, MultiDiscrete
], data: Any, start_dim : int = 0) -> Any:
    assert start_dim >= 0 and start_dim <= len(space.shape)
    
    return space.backend.array_api_namespace.reshape(data, data.shape[:start_dim] + (-1,))

@unflatten_data.register(Box)
@unflatten_data.register(Discrete)
@unflatten_data.register(MultiBinary)
@unflatten_data.register(MultiDiscrete)
def _unflatten_data_common(space: typing.Union[
    Box, Discrete, MultiBinary, MultiDiscrete
], data: Any, start_dim : int = 0) -> Any:
    assert start_dim >= 0 and start_dim <= len(space.shape)
    unflat_dat = space.backend.array_api_namespace.reshape(data, data.shape[:start_dim] + space.shape[start_dim:])
    unflat_dat = space.backend.array_api_namespace.astype(unflat_dat, space.dtype)
    return unflat_dat

@flatten_data.register(Dict)
def _flatten_data_dict(space: Dict, data: typing.Dict[str, typing.Any], start_dim : int = 0) -> Any:
    return space.backend.array_api_namespace.concat([
        flatten_data(subspace, data[key], start_dim) for key, subspace in space.spaces.items()
    ], axis=0)

@unflatten_data.register(Dict)
def _unflatten_data_dict(space: Dict, data: Any, start_dim : int = 0) -> Any:
    result = {}
    start = 0
    for key, subspace in space.spaces.items():
        end = start + flat_dim(subspace, start_dim)
        part_idx = space.backend.array_api_namespace.arange(start, end, dtype=space.backend.default_integer_dtype, device=space.backend.get_device(data))
        part_data = space.backend.array_api_namespace.take(data, part_idx, axis=start_dim)
        result[key] = unflatten_data(subspace, part_data, start_dim)
        start = end
    return result

@flatten_data.register(Text)
def _flatten_data_text(space: Text, data: str, start_dim : int = 0) -> Any:
    assert start_dim == 0
    pad_size = space.max_length - len(data)
    data = space.backend.array_api_namespace.asarray(
        [space.character_index(c) for c in data], 
        dtype=space.backend.default_integer_dtype
    )
    if pad_size > 0:
        padding = space.backend.array_api_namespace.full(
            pad_size, 
            len(space.character_set), 
            dtype=space.backend.default_integer_dtype
        )
        data = space.backend.array_api_namespace.concat(
            [data, padding]
        )
    return data

@unflatten_data.register(Text)
def _unflatten_data_text(space: Text, data: Any, start_dim : int = 0) -> str:
    assert start_dim == 0
    return "".join([
        space.character_set[int(i)] for i in data if i < len(space.character_set)
    ])

@flatten_data.register(Tuple)
def _flatten_data_tuple(space: Tuple, data: typing.Tuple, start_dim : int = 0) -> Any:
    return space.backend.array_api_namespace.concat([
        flatten_data(subspace, data[i], start_dim) for i, subspace in enumerate(space.spaces)
    ], axis=start_dim)

@unflatten_data.register(Tuple)
def _unflatten_data_tuple(space: Tuple, data: Any, start_dim : int = 0) -> Any:
    result = []
    start = 0
    for subspace in space.spaces:
        end = start + flat_dim(subspace, start_dim)
        part_idx = space.backend.array_api_namespace.arange(start, end, dtype=space.backend.default_integer_dtype, device=space.backend.get_device(data))
        part_data = space.backend.array_api_namespace.take(data, part_idx, axis=start_dim)
        print(part_data)
        result.append(unflatten_data(subspace, part_data, start_dim))
        start = end
    return tuple(result)

@flatten_data.register(Union)
def _flatten_data_oneof(space: Union, data: typing.Tuple[int, Any], start_dim : int = 0) -> Any:
    assert start_dim == 0
    space_idx, space_data = data
    flat_sample = flatten_data(space.spaces[space_idx], space_data)
    padding_size = space.flat_dim - len(flat_sample)
    if padding_size > 0:
        padding = space.backend.array_api_namespace.zeros(padding_size, dtype=space.dtype)
        flat_sample = space.backend.array_api_namespace.concat((flat_sample, padding))
    index_array = space.backend.array_api_namespace.full(1, space_idx)
    return space.backend.array_api_namespace.concat((index_array, flat_sample))

@unflatten_data.register(Union)
def _unflatten_data_oneof(space: Union, data: Any, start_dim : int = 0) -> Any:
    assert start_dim == 0
    space_idx = data[0]
    subspace = space.spaces[space_idx]
    subspace_data = data[1:flat_dim(subspace)+1]
    return (space_idx, unflatten_data(subspace, subspace_data))

def is_batch_flattenable(space: Space) -> bool:
    return batch_flat_dim(space) is not None

@singledispatch
def batch_flat_dim(space: Space) -> Optional[int]:
    return None

@batch_flat_dim.register(Box)
@batch_flat_dim.register(MultiBinary)
@batch_flat_dim.register(MultiDiscrete)
def _batch_flat_dim_common(space: Box) -> Optional[int]:
    if len(space.shape) <= 0:
        return None
    return int(np.prod(space.shape[1:]))

@batch_flat_dim.register(Dict)
def _batch_flat_dim_dict(space: Dict) -> Optional[int]:
    dims = 0
    for key, subspace in space.spaces.items():
        dim = batch_flat_dim(subspace)
        if dim is None:
            return None
        dims += dim
    return dims

@batch_flat_dim.register(Tuple)
def _batch_flat_dim_tuple(space: Tuple) -> Optional[int]:
    if len(space.spaces) == 0:
        return 0

    if all(type(subspace) in batch_flat_dim.registry for subspace in space.spaces):
        dims = 0
        for subspace in space.spaces:
            dim = batch_flat_dim(subspace)
            if dim is None:
                return None
            dims += dim
        return dims
    else:
        flat_dims = [flat_dim(subspace) for subspace in space.spaces]
        if any((dim is None or dim != flat_dims[0]) for dim in flat_dims):
            return None
        return flat_dims[0]

@singledispatch
def batch_flatten_space(space: Space) -> Box:
    raise NotImplementedError(f"Space {space} is not batch-flattenable")

@batch_flatten_space.register(Box)
def _batch_flatten_space_box(space: Box) -> Box:
    return Box(
        space.backend,
        low=space.backend.array_api_namespace.reshape(
            space.low, (space.shape[0], -1)
        ), 
        high=space.backend.array_api_namespace.reshape(
            space.high, (space.shape[0], -1)
        ), 
        dtype=space.dtype,
        device=space.device
    )

@batch_flatten_space.register(MultiBinary)
@batch_flatten_space.register(MultiDiscrete)
def _batch_flatten_space_multi_binary_discrete(space: typing.Union[MultiBinary, MultiDiscrete]) -> Box:
    return Box(
        space.backend,
        low=0, high=1, 
        shape=(space.shape[0], batch_flat_dim(space)), 
        dtype=space.dtype or space.backend.default_integer_dtype,
        device=space.device
    )

@batch_flatten_space.register(Dict)
def _batch_flatten_space_dict(space: Dict) -> Box:
    space_list = [batch_flatten_space(s) for s in space.spaces.values()]
    return Box(
        space.backend,
        low=space.backend.array_api_namespace.concat([s.low for s in space_list], axis=1),
        high=space.backend.array_api_namespace.concat([s.high for s in space_list], axis=1),
        dtype=space.backend.default_floating_dtype,
        device=space.device
    )

@batch_flatten_space.register(Tuple)
def _batch_flatten_space_tuple(space: Tuple) -> Box:
    if all(type(subspace) in batch_flat_dim.registry for subspace in space.spaces):
        space_list = [batch_flatten_space(s) for s in space.spaces]
        return Box(
            space.backend,
            low=space.backend.array_api_namespace.concat([s.low for s in space_list], axis=1),
            high=space.backend.array_api_namespace.concat([s.high for s in space_list], axis=1),
            dtype=space.backend.default_floating_dtype,
            device=space.device
        )
    else:
        space_list = [flatten_space(s) for s in space.spaces]
        return Box(
            space.backend,
            low=space.backend.array_api_namespace.concat([s.low for s in space_list], axis=0),
            high=space.backend.array_api_namespace.concat([s.high for s in space_list], axis=0),
            dtype=space.backend.default_floating_dtype,
            device=space.device
        )
    
@singledispatch
def batch_flatten_data(space: Space, data: Any) -> Any:
    raise NotImplementedError(f"Space {space} is not batch-flattenable")

@singledispatch
def batch_unflatten_data(space: Space, data: Any) -> Any:
    raise NotImplementedError(f"Space {space} is not batch-flattenable")

@batch_flatten_data.register(Box)
@batch_flatten_data.register(MultiBinary)
@batch_flatten_data.register(MultiDiscrete)
def _batch_flatten_data_common(space: typing.Union[
    Box, MultiBinary, MultiDiscrete
], data: Any) -> Any:
    return space.backend.array_api_namespace.reshape(
        data, 
        (data.shape[0], -1)
    )

@batch_unflatten_data.register(Box)
@batch_unflatten_data.register(MultiBinary)
@batch_unflatten_data.register(MultiDiscrete)
def _batch_unflatten_data_common(space: typing.Union[
    Box, MultiBinary, MultiDiscrete
], data: Any) -> Any:
    return space.backend.array_api_namespace.reshape(
        data, 
        (data.shape[0], *space.shape[1:])
    )

@batch_flatten_data.register(Dict)
def _batch_flatten_data_dict(space: Dict, data: typing.Dict[str, typing.Any]) -> Any:
    return space.backend.array_api_namespace.concat([
        batch_flatten_data(subspace, data[key]) for key, subspace in space.spaces.items()
    ], axis=1)

@batch_unflatten_data.register(Dict)
def _batch_unflatten_data_dict(space: Dict, data: Any) -> Any:
    result = {}
    start = 0
    for key, subspace in space.spaces.items():
        end = start + batch_flat_dim(subspace)
        result[key] = batch_unflatten_data(subspace, data[:, start:end])
        start = end
    return result

@batch_flatten_data.register(Tuple)
def _batch_flatten_data_tuple(space: Tuple, data: typing.Tuple) -> Any:
    if all(type(subspace) in batch_flat_dim.registry for subspace in space.spaces):
        return space.backend.array_api_namespace.concat([
            batch_flatten_data(subspace, data[i]) for i, subspace in enumerate(space.spaces)
        ], axis=1)
    else:
        return space.backend.array_api_namespace.stack([
            flatten_data(subspace, data[i]) for i, subspace in enumerate(space.spaces)
        ], axis=0)
    
@batch_unflatten_data.register(Tuple)
def _batch_unflatten_data_tuple(space: Tuple, data: Any) -> Any:
    if all(type(subspace) in batch_flat_dim.registry for subspace in space.spaces):
        result = []
        start = 0
        for subspace in space.spaces:
            end = start + batch_flat_dim(subspace)
            result.append(batch_unflatten_data(subspace, data[:, start:end]))
            start = end
        return tuple(result)
    else:
        result = []
        for i, subspace in enumerate(space.spaces):
            result.append(unflatten_data(subspace, data[i]))
        return tuple(result)