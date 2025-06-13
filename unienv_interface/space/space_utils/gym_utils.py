from __future__ import annotations

import typing
from copy import deepcopy
from functools import singledispatch
from typing import Optional, Any, Iterable, Iterator
try:
    import gymnasium as gym
except ImportError:
    raise ImportError(
        "The 'gymnasium' package is required for this module. "
        "Please install it using 'pip install gymnasium'."
    )

import numpy as np
from unienv_interface.backends import ComputeBackend
from unienv_interface.space.spaces import *

__all__ = [
    "to_gym_space",
    "from_gym_space",
    "to_gym_data",
    "from_gym_data"
]

@singledispatch
def to_gym_space(space: Space) -> gym.Space:
    raise TypeError(
        f"The space provided to `to_gym_space` is not supported, type: {type(space)}, {space}"
    )

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

@singledispatch
def to_gym_data(space : Space, data: Any) -> Any:
    raise TypeError(
        f"The space provided to `to_gym_data` is not supported, type: {type(space)}, {space}"
    )

@singledispatch
def from_gym_data(space : Space, data: Any) -> Any:
    raise TypeError(
        f"The space provided to `from_gym_data` is not supported, type: {type(space)}, {space}"
    )

# ========== BoxSpace Implementations ==========

@to_gym_space.register(BoxSpace)
def _to_gym_space_box(
    space: BoxSpace,
) -> gym.spaces.Box:
    new_low = space.backend.to_numpy(space.low)
    new_high = space.backend.to_numpy(space.high)
    return gym.spaces.Box(
        low=new_low,
        high=new_high,
        shape=space.shape,
        dtype=new_low.dtype
    )

@from_gym_space.register(gym.spaces.Box)
def _from_gym_space_box(
    gym_space : gym.spaces.Box,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> BoxSpace:
    new_low = backend.from_numpy(gym_space.low, dtype=dtype, device=device)
    new_high = backend.from_numpy(gym_space.high, dtype=dtype, device=device)
    dtype = dtype or new_low.dtype or backend.default_floating_dtype
    
    if not (backend.dtype_is_real_floating(dtype) or backend.dtype_is_real_integer(dtype)):
        dtype = backend.default_floating_dtype

    return BoxSpace(
        backend,
        low=new_low,
        high=new_high,
        device=device,
        dtype=dtype,
        shape=gym_space.shape,
    )

# Additional BoxSpace from_methods for compatibility with redundant gym spaces
@from_gym_space.register(gym.spaces.Discrete)
def _from_gym_space_discrete(
    gym_space : gym.spaces.Discrete,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> BoxSpace:
    assert dtype is None or backend.dtype_is_real_integer(dtype), f"Unsupported dtype for BoxSpace: {dtype}"
    return BoxSpace(
        backend,
        low=gym_space.start,
        high=gym_space.start + gym_space.n, # No need to -1 as BoxSpace will include only [low, high)
        dtype=dtype or backend.default_integer_dtype,
        device=device,
        shape=(),
    )

@from_gym_space.register(gym.spaces.MultiDiscrete)
def _from_gym_space_multi_discrete(
    gym_space : gym.spaces.MultiDiscrete,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> BoxSpace:
    assert dtype is None or backend.dtype_is_real_integer(dtype), f"Unsupported dtype for BoxSpace: {dtype}"
    nvec = backend.from_numpy(gym_space.nvec, dtype=dtype, device=device)
    start = backend.from_numpy(gym_space.start, dtype=dtype, device=device)
    return BoxSpace(
        backend,
        low=start,
        high=start + nvec,  # No need to -1 as BoxSpace will include only [low, high)
        dtype=dtype or backend.default_integer_dtype,
        device=device,
        shape=gym_space.shape
    )

@from_gym_data.register(BoxSpace)
def _from_gym_data_box(
    space: BoxSpace,
    data: Any,
) -> Any:
    if isinstance(data, np.ndarray):
        return space.backend.from_numpy(data, dtype=space.dtype, device=space.device)
    elif isinstance(data, int):
        return space.backend.from_numpy(np.array(data), dtype=space.dtype, device=space.device)
    else:
        raise TypeError(f"Unsupported data type for BoxSpace: {type(data)}")

@to_gym_data.register(BoxSpace)
def _to_gym_data_box(
    space: BoxSpace,
    data: Any,
) -> Any:
    assert space.backend.is_backendarray(data), f"Data must be a backend array, got: {type(data)}"
    return space.backend.to_numpy(data)

# ========== DiscreteSpace Implementations ==========
@to_gym_space.register(BinarySpace)
def _to_gym_space_binary(
    space: BinarySpace,
) -> gym.spaces.MultiBinary:
    return gym.spaces.MultiBinary(
        shape=space.shape,
    )

@from_gym_space.register(gym.spaces.MultiBinary)
def _from_gym_space_multi_binary(
    gym_space : gym.spaces.MultiBinary,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> BinarySpace:
    assert dtype is None or backend.dtype_is_boolean(dtype), f"Unsupported dtype for BinarySpace: {dtype}"
    return BinarySpace(
        backend,
        shape=gym_space.shape,
        dtype=dtype,
        device=device,
    )

@from_gym_data.register(BinarySpace)
def _from_gym_data_binary(
    space: BinarySpace,
    data: Any,
) -> Any:
    if isinstance(data, np.ndarray):
        return space.backend.from_numpy(data, dtype=space.dtype, device=space.device)
    elif isinstance(data, bool):
        return space.backend.from_numpy(np.array(data), dtype=space.dtype, device=space.device)
    else:
        raise TypeError(f"Unsupported data type for BinarySpace: {type(data)}")

@to_gym_data.register(BinarySpace)
def _to_gym_data_binary(
    space: BinarySpace,
    data: Any,
) -> Any:
    assert space.backend.is_backendarray(data), f"Data must be a backend array, got: {type(data)}"
    return space.backend.to_numpy(data).astype(bool)

# ========== GraphSpace Implementations ==========
@to_gym_space.register(GraphSpace)
def _to_gym_space_graph(
    space: GraphSpace,
) -> gym.spaces.Graph:
    assert space.node_feature_space is not None, "GraphSpace must have a node feature space defined"

    node_space = to_gym_space(space.node_feature_space)
    edge_space = to_gym_space(space.edge_feature_space) if space.edge_feature_space is not None else None
    return gym.spaces.Graph(
        node_space=node_space,
        edge_space=edge_space,
    )

@from_gym_space.register(gym.spaces.Graph)
def _from_gym_space_graph(
    gym_space : gym.spaces.Graph,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
    *,
    is_edge: bool = False,
    min_nodes: int = 1, 
    max_nodes: typing.Optional[int] = None, 
    min_edges: int = 1, 
    max_edges: typing.Optional[int] = None, 
) -> GraphSpace:
    node_space = from_gym_space(
        gym_space.node_space,
        backend=backend,
        dtype=dtype,
        device=device
    )
    edge_space = from_gym_space(
        gym_space.edge_space,
        backend=backend,
        dtype=dtype,
        device=device
    ) if gym_space.edge_space is not None else None

    return GraphSpace(
        backend,
        node_feature_space=node_space,
        edge_feature_space=edge_space,
        is_edge=is_edge,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        min_edges=min_edges,
        max_edges=max_edges,
        batch_shape=(),
        device=device
    )

@from_gym_data.register(GraphSpace)
def _from_gym_data_graph(
    space: GraphSpace,
    data: Any,
) -> Any:
    if isinstance(data, gym.spaces.GraphInstance):
        node_number = data.nodes.shape[0]
        edge_number = data.edges.shape[0] if data.edges is not None else None
        node_features = from_gym_data(space.node_feature_space, data.nodes)
        edge_features = from_gym_data(space.edge_feature_space, data.edges) if data.edges is not None else None
        edges = space.backend.from_numpy(data.edge_links, dtype=space.backend.default_integer_dtype, device=space.device) if data.edge_links is not None else None
        return GraphInstance(
            n_nodes=space.backend.full((), node_number, dtype=space.backend.default_integer_dtype, device=space.device),
            n_edges=space.backend.full((), edge_number, dtype=space.backend.default_integer_dtype, device=space.device) if edge_number is not None else None,
            nodes_features=node_features,
            edges_features=edge_features,
            edges=edges
        )
    else:
        raise TypeError(f"Unsupported data type for GraphSpace: {type(data)}")

@to_gym_data.register(GraphSpace)
def _to_gym_data_graph(
    space: GraphSpace,
    data: Any,
) -> Any:
    assert isinstance(data, GraphInstance), f"Data must be a GraphInstance, got: {type(data)}"
    assert data.n_nodes.shape == (), f"n_nodes must be a scalar, got shape: {data.n_nodes.shape}"
    assert data.nodes_features is not None, "GraphInstance must have node features"

    node_data = to_gym_data(space.node_feature_space, data.nodes_features)
    edge_data = to_gym_data(space.edge_feature_space, data.edges_features) if data.edges_features is not None else None
    edge_links = space.backend.to_numpy(data.edges) if data.edges is not None else None

    return gym.spaces.GraphInstance(
        nodes=node_data,
        edges=edge_data,
        edge_links=edge_links
    )

# ========== TextSpace Implementations ==========
@to_gym_space.register(TextSpace)
def _to_gym_space_text(
    space: TextSpace,
) -> gym.spaces.Text:
    assert space.charset is not None, "TextSpace must have a charset defined"
    return gym.spaces.Text(
        max_length=space.max_length,
        min_length=space.min_length,
        charset=space.charset
    )

@from_gym_space.register(gym.spaces.Text)
def _from_gym_space_text(
    gym_space : gym.spaces.Text,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> TextSpace:
    return TextSpace(
        backend,
        max_length=gym_space.max_length,
        min_length=gym_space.min_length,
        charset=gym_space.character_set,
        device=device,
    )

@from_gym_data.register(TextSpace)
def _from_gym_data_text(
    space: TextSpace,
    data: str,
) -> Any:
    assert isinstance(data, str), f"Data must be a string, got: {type(data)}"
    return data

@to_gym_data.register(TextSpace)
def _to_gym_data_text(
    space: TextSpace,
    data: str,
) -> Any:
    assert isinstance(data, str), f"Data must be a string, got: {type(data)}"
    return data

# ========== DictSpace Implementations ==========
@to_gym_space.register(DictSpace)
def _to_gym_space_dict(
    space: DictSpace,
) -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {key: to_gym_space(subspace) for key, subspace in space.spaces.items()}
    )

@from_gym_space.register(gym.spaces.Dict)
def _from_gym_space_dict(
    gym_space : gym.spaces.Dict,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> DictSpace:
    return DictSpace(
        backend=backend,
        spaces={key: from_gym_space(space, backend, dtype, device) for key, space in gym_space.spaces.items()},
        device=device,
    )

@from_gym_data.register(DictSpace)
def _from_gym_data_dict(
    space: DictSpace,
    data: typing.Mapping[str, Any],
) -> typing.Dict[str, Any]:
    assert isinstance(data, typing.Mapping), f"Data must be a mapping, got: {type(data)}"
    return {
        key: from_gym_data(subspace, data[key]) for key, subspace in space.spaces.items()
    }

@to_gym_data.register(DictSpace)
def _to_gym_data_dict(
    space: DictSpace,
    data: typing.Mapping[str, Any],
) -> typing.Dict[str, Any]:
    assert isinstance(data, typing.Mapping), f"Data must be a mapping, got: {type(data)}"
    return {
        key: to_gym_data(subspace, data[key]) for key, subspace in space.spaces.items()
    }

# ========== TupleSpace Implementations ==========
@to_gym_space.register(TupleSpace)
def _to_gym_space_tuple(
    space: TupleSpace,
) -> gym.spaces.Tuple:
    return gym.spaces.Tuple(
        [to_gym_space(subspace) for subspace in space.spaces]
    )

@from_gym_space.register(gym.spaces.Tuple)
def _from_gym_space_tuple(
    gym_space : gym.spaces.Tuple,
    backend : ComputeBackend,
    dtype : Optional[Any] = None,
    device : Optional[Any] = None,
) -> TupleSpace:
    return TupleSpace(
        backend=backend,
        spaces=[from_gym_space(subspace, backend=backend, dtype=dtype, device=device) for subspace in gym_space.spaces],
        device=device
    )

@from_gym_data.register(TupleSpace)
def _from_gym_data_tuple(
    space: TupleSpace,
    data: typing.Tuple[Any, ...],
) -> typing.Tuple[Any, ...]:
    assert isinstance(data, tuple), f"Data must be a tuple, got: {type(data)}"
    return tuple(from_gym_data(subspace, value) for subspace, value in zip(space.spaces, data))

@to_gym_data.register(TupleSpace)
def _to_gym_data_tuple(
    space: TupleSpace,
    data: typing.Tuple[Any, ...],
) -> typing.Tuple[Any, ...]:
    assert isinstance(data, tuple), f"Data must be a tuple, got: {type(data)}"
    return tuple(to_gym_data(subspace, value) for subspace, value in zip(space.spaces, data))

# ========== UnionSpace Implementations ==========
if "OneOf" in dir(gym.spaces):
    @to_gym_space.register(UnionSpace)
    def _to_gym_space_union(
        space: UnionSpace,
    ) -> gym.spaces.OneOf:
        return gym.spaces.OneOf(
            [to_gym_space(subspace) for subspace in space.spaces]
        )
    
    @from_gym_space.register(gym.spaces.OneOf)
    def _from_gym_space_union(
        gym_space : "gym.spaces.OneOf",
        backend : ComputeBackend,
        dtype : Optional[Any] = None,
        device : Optional[Any] = None,
        seed : Optional[int] = None
    ) -> UnionSpace:
        return UnionSpace(
            backend,
            spaces=[from_gym_space(space, backend=backend, dtype=dtype, device=device) for space in gym_space.spaces],
            device=device
        )

    @from_gym_data.register(UnionSpace)
    def _from_gym_data_union(
        space: UnionSpace,
        data: typing.Tuple[int, Any],
    ) -> Any:
        data_idx, data_value = data
        assert 0 <= data_idx < len(space.spaces), f"Data index {data_idx} out of bounds for UnionSpace with {len(space.spaces)} spaces"
        return from_gym_data(space.spaces[data_idx], data_value)
    
    @to_gym_data.register(UnionSpace)
    def _to_gym_data_union(
        space: UnionSpace,
        data: typing.Tuple[int, Any],
    ) -> typing.Tuple[int, Any]:
        data_idx, data_value = data
        assert 0 <= data_idx < len(space.spaces), f"Data index {data_idx} out of bounds for UnionSpace with {len(space.spaces)} spaces"
        return (data_idx, to_gym_data(space.spaces[data_idx], data_value))