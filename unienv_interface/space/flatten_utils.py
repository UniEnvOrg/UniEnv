import typing
from copy import deepcopy
from functools import singledispatch
from typing import Optional, Any, Iterable, Iterator
from unienv_interface.space import Space, Box

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
    "flatten_data",
    "unflatten_data",
    "flatten_space"
]

def is_flattenable(space: Space) -> bool:
    return space.is_flattenable

def flat_dim(space : Space) -> Optional[int]:
    return space.flat_dim

def flatten_data(space : Space, data : Any) -> Any:
    return space.flatten(data)

def unflatten_data(space : Space, data : Any) -> Any:
    return space.unflatten(data)

@singledispatch
def flatten_space(space: Space) -> Box:
    raise NotImplementedError(f"Unknown space: `{space}`")

@flatten_space.register(Box)
def _flatten_space_box(space: Box) -> Box:
    return Box(
        space.backend,
        low=space.backend.array_api_namespace.reshape(
            space.low, (-1,)
        ), 
        high=space.backend.array_api_namespace.reshape(
            space.high, (-1,)
        ), 
        dtype=space.dtype,
        device=space.device
    )

@flatten_space.register(Discrete)
@flatten_space.register(MultiBinary)
@flatten_space.register(MultiDiscrete)
def _flatten_space_binary(space: Discrete | MultiBinary | MultiDiscrete) -> Box:
    return Box(
        space.backend,
        low=0, high=1, 
        shape=(flat_dim(space),), 
        dtype=space.dtype or space.backend.default_integer_dtype,
        device=space.device
    )

@flatten_space.register(Tuple)
@flatten_space.register(Dict)
def _flatten_space_tuple(space: Tuple | Dict) -> Box:
    assert space.is_flattenable
    space_list = [flatten_space(s) for s in space.spaces] if isinstance(space, Tuple) else [flatten_space(s) for s in space.spaces.values()]

    return Box(
        space.backend,
        low=space.backend.array_api_namespace.concat([s.low for s in space_list], axis=0),
        high=space.backend.array_api_namespace.concat([s.high for s in space_list], axis=0),
        dtype=space.backend.default_floating_dtype,
        device=space.device
    )

@flatten_space.register(Text)
def _flatten_space_text(space: Text) -> Box:
    return Box(
        space.backend,
        low=0, high=len(space.character_set), 
        shape=(space.flat_dim,), 
        dtype=space.backend.default_integer_dtype,
        device=space.device
    )

@flatten_space.register(Union)
def _flatten_space_oneof(space: Union) -> Box:
    num_subspaces = len(space.spaces)
    max_flatdim = max(flat_dim(s) for s in space.spaces) + 1

    lows = space.backend.array_api_namespace.asarray([space.backend.array_api_namespace.min(flatten_space(s).low) for s in space.spaces])
    highs = space.backend.array_api_namespace.asarray([space.backend.array_api_namespace.max(flatten_space(s).high) for s in space.spaces])

    overall_low = space.backend.array_api_namespace.min(lows)
    overall_high = space.backend.array_api_namespace.max(highs)

    low = space.backend.array_api_namespace.concatenate([
        space.backend.array_api_namespace.asarray([0]), 
        space.backend.array_api_namespace.full(max_flatdim - 1, overall_low)
    ])
    high = space.backend.array_api_namespace.concatenate([
        space.backend.array_api_namespace.asarray([num_subspaces - 1]), 
        space.backend.array_api_namespace.full(max_flatdim - 1, overall_high)
    ])

    return Box(
        backend=space.backend,
        low=low, high=high, 
        shape=(max_flatdim,), 
        dtype=space.backend.default_floating_dtype,
        device=space.device
    )