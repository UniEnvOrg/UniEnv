from __future__ import annotations

import typing
from copy import deepcopy
from functools import singledispatch
from typing import Optional, Any, Iterable, Iterator
import gymnasium as gym

import numpy as np
from unienv_interface.backends import ComputeBackend
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

__all__ = [
    "to_gym_space",
    "from_gym_space",
    "to_gym_data",
    "from_gym_data"
]

def to_gym_space(space: Space) -> gym.Space:
    return space.to_gym_space()

@singledispatch
def from_gym_space(
    gym_space : gym.Space,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> Space:
    raise TypeError(
        f"The space provided to `from_gym_space` is not supported, type: {type(gym_space)}, {gym_space}"
    )

def to_gym_data(space : Space, data: Any) -> Any:
    return space.to_gym_data(data)

def from_gym_data(space : Space, data: Any) -> Any:
    return space.from_gym_data(data)

@from_gym_space.register(gym.spaces.Box)
def _from_gym_space_box(
    gym_space : gym.spaces.Box,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> Box:
    assert isinstance(gym_space, gym.spaces.Box), f"Expects gym_space to be of type gym.spaces.Box, actual type: {type(gym_space)}"
    return Box(
        backend=backend,
        low=backend.from_numpy(gym_space.low, dtype=dtype, device=device),
        high=backend.from_numpy(gym_space.high, dtype=dtype, device=device),
        device=device,
        dtype=dtype,
        shape=gym_space.shape,
    )

@from_gym_space.register(gym.spaces.Dict)
def _from_gym_space_dict(
    gym_space : gym.spaces.Dict,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> Dict:
    return Dict(
        backend=backend,
        spaces={key: from_gym_space(space, backend, dtype, device) for key, space in gym_space.spaces.items()},
        device=device,
    )

@from_gym_space.register(gym.spaces.Discrete)
def _from_gym_space_discrete(
    gym_space : gym.spaces.Discrete,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> Discrete:
    return Discrete(
        backend=backend,
        n=int(gym_space.n),
        start=int(gym_space.start),
        device=device,
        dtype=dtype,
    )

@from_gym_space.register(gym.spaces.Graph)
def _from_gym_space_graph(
    gym_space : gym.spaces.Graph,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> Graph:
    return Graph(
        backend=backend,
        node_space=from_gym_space(
            gym_space.node_space,
            backend=backend,
            dtype=dtype,
            device=device,
        ),
        edge_space=from_gym_space(
            gym_space.edge_space,
            backend=backend,
            dtype=dtype,
            device=device,
        ) if gym_space.edge_space is not None else None,
        device=device,
    )

@from_gym_space.register(gym.spaces.MultiBinary)
def _from_gym_space_multi_binary(
    gym_space : gym.spaces.MultiBinary,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> MultiBinary:
    return MultiBinary(
        backend=backend,
        shape=gym_space.shape,
        dtype=dtype,
        device=device,
    )

@from_gym_space.register(gym.spaces.MultiDiscrete)
def _from_gym_space_multi_discrete(
    gym_space : gym.spaces.MultiDiscrete,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> MultiDiscrete:
    nvec = backend.from_numpy(gym_space.nvec, dtype=dtype, device=device)
    start = backend.from_numpy(gym_space.start, dtype=dtype, device=device)
    return MultiDiscrete(
        backend=backend,
        nvec=nvec,
        start=start,
        dtype=dtype,
        device=device,
    )

@from_gym_space.register(gym.spaces.Sequence)
def _from_gym_space_sequence(
    gym_space : gym.spaces.Sequence,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> Sequence:
    assert not gym_space.stack, "Stacking is not supported"
    return Sequence(
        backend=backend,
        space=from_gym_space(
            gym_space.feature_space,
            backend=backend,
            dtype=dtype,
            device=device
        ),
        stack=gym_space.stack,
    )

@from_gym_space.register(gym.spaces.Text)
def _from_gym_space_text(
    gym_space : gym.spaces.Text,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> Text:
    
    return Text(
        backend=backend,
        max_length=gym_space.max_length,
        min_length=gym_space.min_length,
        charset=gym_space.character_set,
        device=device,
    )

@from_gym_space.register(gym.spaces.Tuple)
def _from_gym_space_tuple(
    gym_space : gym.spaces.Tuple,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> Tuple:
    return Tuple(
        backend=backend,
        spaces=[from_gym_space(subspace, backend=backend, dtype=dtype, device=device) for subspace in gym_space.spaces],
        device=device
    )

def _from_gym_space_union(
    gym_space : "gym.spaces.OneOf",
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
    seed : Optional[int] = None
) -> Union:
    return Union(
        backend=backend,
        spaces=[from_gym_space(space, backend=backend, dtype=dtype, device=device) for space in gym_space.spaces],
        device=device,
        seed=seed
    )

if "OneOf" in gym.spaces.__all__:
    from_gym_space.register(gym.spaces.OneOf, _from_gym_space_union)
