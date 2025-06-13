from __future__ import annotations

import typing
import copy
from functools import singledispatch
from typing import Optional, Any, Iterable, Iterator
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType, ArrayAPIGetIndex, ArrayAPISetIndex

from ..spaces import *

__all__ = [
    "batch_size",
    "batch_space",
    "batch_differing_spaces",
    "unbatch_spaces",
    "swap_batch_dims",
    "swap_batch_dims_in_data",
    "iterate",
    "get_at",
    "set_at",
    "concatenate",
]

def _tensor_transpose(backend : ComputeBackend, tensor : BArrayType, dim1 : int, dim2 : int) -> BArrayType:
    dims = list(range(tensor.ndim))
    dims[dim1] = dim2
    dims[dim2] = dim1
    return backend.permute_dims(tensor, axes=tuple(dims))

def _shape_transpose(shape : tuple[int, ...], dim1 : int, dim2 : int) -> tuple[int, ...]:
    shape = list(shape)
    shape[dim1], shape[dim2] = shape[dim2], shape[dim1]
    return tuple(shape)

@singledispatch
def swap_batch_dims(space: Space, dim1: int, dim2: int) -> Space:
    raise TypeError(
        f"The space provided to `swap_batch_dims` is not a Space instance, type: {type(space)}, {space}"
    )

@swap_batch_dims.register(BoxSpace)
def _swap_batch_dims_box(space: BoxSpace, dim1: int, dim2: int):
    return BoxSpace(
        backend=space.backend,
        low=_tensor_transpose(space.backend, space.low, dim1, dim2),
        high=_tensor_transpose(space.backend, space.high, dim1, dim2),
        dtype=space.dtype,
        device=space.device,
    )

@swap_batch_dims.register(BinarySpace)
def _swap_batch_dims_binary(space: BinarySpace, dim1: int, dim2: int):
    return BinarySpace(
        backend=space.backend,
        shape=_shape_transpose(space.shape, dim1, dim2),
        dtype=space.dtype,
        device=space.device,
    )

@swap_batch_dims.register(DictSpace)
def _swap_batch_dims_dict(space: DictSpace, dim1: int, dim2: int):
    return DictSpace(
        backend=space.backend,
        spaces={key: swap_batch_dims(subspace, dim1, dim2) for key, subspace in space.spaces.items()},
        device=space.device,
    )

@swap_batch_dims.register(TupleSpace)
def _swap_batch_dims_tuple(space: TupleSpace, dim1: int, dim2: int):
    assert all(type(subspace) in swap_batch_dims.registry for subspace in space.spaces), "Expected all subspaces in TupleSpace to be swappable"
    return TupleSpace(
        backend=space.backend,
        spaces=[swap_batch_dims(subspace, dim1, dim2) for subspace in space.spaces],
        device=space.device,
    )

@singledispatch
def swap_batch_dims_in_data(
    space: Space, data: Any, dim1: int, dim2: int
) -> Any:
    raise TypeError(
        f"The space provided to `swap_batch_dims_in_data` is not a Space instance, type: {type(space)}, {space}"
    )

@swap_batch_dims_in_data.register(BoxSpace)
@swap_batch_dims_in_data.register(BinarySpace)
def _swap_batch_dims_in_data_common(space: typing.Union[BoxSpace, BinarySpace], data: BArrayType, dim1: int, dim2: int):
    return _tensor_transpose(space.backend, data, dim1, dim2)

@swap_batch_dims_in_data.register(DictSpace)
def _swap_batch_dims_in_data_dict(space: DictSpace, data: dict[str, Any], dim1: int, dim2: int):
    return {
        key: swap_batch_dims_in_data(subspace, data[key], dim1, dim2)
        for key, subspace in space.spaces.items()
    }

@swap_batch_dims_in_data.register(TupleSpace)
def _swap_batch_dims_in_data_tuple(space: TupleSpace, data: tuple[Any, ...], dim1: int, dim2: int):
    assert all(type(subspace) in swap_batch_dims_in_data.registry for subspace in space.spaces), "Expected all subspaces in TupleSpace to be swappable"
    return tuple(
        swap_batch_dims_in_data(subspace, data[i], dim1, dim2)
        for i, subspace in enumerate(space.spaces)
    )

@singledispatch
def batch_size(space: Space) -> Optional[int]:
    """Space does not support batching."""
    return None

@batch_size.register(BoxSpace)
@batch_size.register(BinarySpace)
def _batch_size_box(space: typing.Union[BoxSpace, BinarySpace]):
    return space.shape[0] if len(space.shape) > 0 else None

@batch_size.register(GraphSpace)
def _batch_size_graph(space: GraphSpace):
    return space.batch_shape[0] if len(space.batch_shape) > 0 else None

@batch_size.register(DictSpace)
def _batch_size_dict(space: DictSpace):
    for subspace in space.spaces.values():
        try:
            return batch_size(subspace)
        except:
            continue

@batch_size.register(TupleSpace)
def _batch_size_tuple(space: TupleSpace):
    if all(type(subspace) in batch_size.registry for subspace in space.spaces):
        for subspace in space.spaces:
            try:
                return batch_size(subspace)
            except:
                continue

    return len(space.spaces)

def batch_size_data(data: Any) -> Optional[int]:
    if hasattr(data, "shape"):
        return data.shape[0] if len(data.shape) > 0 else None
    elif isinstance(data, typing.Mapping):
        for value in data.values():
            return batch_size_data(value)
    elif isinstance(data, GraphInstance):
        return data.n_nodes.shape[0] if len(data.n_nodes.shape) > 0 else None
    elif isinstance(data, typing.Sequence):
        for value in data:
            return batch_size_data(value)
    else:
        raise TypeError(f"Unable to determine batch size of data, type: {type(data)}")

@singledispatch
def batch_space(space: Space, n: int = 1) -> Space:
    raise TypeError(
        f"The space provided to `batch_space` does not support batching, type: {type(space)}, {space}"
    )

@batch_space.register(BoxSpace)
def _batch_space_box(space: BoxSpace, n: int = 1):
    return BoxSpace(
        backend=space.backend,
        low=space._low[None],
        high=space._high[None],
        dtype=space.dtype,
        device=space.device,
        shape=(n,) + space.shape,
    )

@batch_space.register(BinarySpace)
def _batch_space_binary(space: BinarySpace, n: int = 1):
    return BinarySpace(
        backend=space.backend,
        shape=(n,) + space.shape,
        device=space.device,
        dtype=space.dtype,
    )

@batch_space.register(GraphSpace)
def _batch_space_graph(space: GraphSpace, n: int = 1):
    return GraphSpace(
        backend=space.backend,
        node_feature_space=space.node_feature_space,
        edge_feature_space=space.edge_feature_space,
        is_edge=space.is_edge,
        min_nodes=space.min_nodes,
        max_nodes=space.max_nodes,
        min_edges=space.min_edges,
        max_edges=space.max_edges,
        batch_shape=(n,) + space.batch_shape,
        device=space.device
    )

@batch_space.register(DictSpace)
def _batch_space_dict(space: DictSpace, n: int = 1):
    return DictSpace(
        backend=space.backend,
        spaces={key: batch_space(subspace, n=n) for key, subspace in space.spaces.items()},
        device=space.device,
    )

@batch_space.register(TupleSpace)
def _batch_space_tuple(space: TupleSpace, n: int = 1):
    return TupleSpace(
        backend=space.backend,
        spaces=[batch_space(subspace, n=n) for subspace in space.spaces],
        device=space.device,
    )

@singledispatch
def batch_differing_spaces(spaces: typing.Sequence[Space], device : Optional[Any] = None) -> Space:
    assert len(spaces) > 0, "Expects a non-empty list of spaces"
    assert all(
        isinstance(space, type(spaces[0])) for space in spaces
    ), f"Expects all spaces to be the same type, actual types: {[type(space) for space in spaces]}"
    assert all(
        spaces[0].backend == space.backend for space in spaces
    ), f"Expects all spaces to have the same backend, actual backends: {[space.backend for space in spaces]}"
    assert (
        type(spaces[0]) in batch_differing_spaces.registry
    ), f"Requires the Space type to have a registered `batch_differing_space`, current list: {batch_differing_spaces.registry}"

    return batch_differing_spaces.dispatch(type(spaces[0]))(spaces, device)

@batch_differing_spaces.register(BoxSpace)
def _batch_differing_spaces_box(spaces: typing.Sequence[BoxSpace], device : Optional[Any] = None):
    assert all(
        spaces[0].dtype == space.dtype for space in spaces
    ), f"Expected all dtypes to be equal, actually {[space.dtype for space in spaces]}"
    assert all(
        spaces[0].shape == space.shape for space in spaces
    ), f"Expected all BoxSpace.low shape to be equal, actually {[space.low.shape for space in spaces]}"
    assert all(
        spaces[0].shape == space.shape for space in spaces
    ), f"Expected all BoxSpace.high shape to be equal, actually {[space.high.shape for space in spaces]}"

    backend = spaces[0].backend
    target_low = backend.stack(backend.broadcast_arrays(*[space._low for space in spaces]), axis=0)
    target_high = backend.stack(backend.broadcast_arrays(*[space._high for space in spaces]), axis=0)

    return BoxSpace(
        backend=backend,
        low=target_low,
        high=target_high,
        dtype=spaces[0].dtype,
        device=device if device is not None else spaces[0].device,
    )

@batch_differing_spaces.register(BinarySpace)
def _batch_differing_spaces_binary(spaces: typing.Sequence[BinarySpace], device : Optional[Any] = None):
    assert all(spaces[0].shape == space.shape for space in spaces)
    
    backend=spaces[0].backend
    return BinarySpace(
        backend=backend,
        shape=(len(spaces),) + spaces[0].shape,
        dtype=spaces[0].dtype,
        device=device if device is not None else spaces[0].device,
    )

@batch_differing_spaces.register(DictSpace)
def _batch_differing_spaces_dict(spaces: typing.Sequence[DictSpace], device : Optional[Any] = None):
    assert all(spaces[0].keys() == space.keys() for space in spaces)

    return DictSpace(
        backend=spaces[0].backend,
        spaces={
            key: batch_differing_spaces([space[key] for space in spaces], device)
            for key in spaces[0].keys()
        },
        device=device if device is not None else spaces[0].device,
    )

@batch_differing_spaces.register(TupleSpace)
def _batch_differing_spaces_tuple(spaces: typing.Sequence[TupleSpace], device : Optional[Any] = None):
    return TupleSpace(
        backend=spaces[0].backend,
        spaces=[
            batch_differing_spaces(subspaces, device)
            for subspaces in zip(*[space.spaces for space in spaces])
        ],
        device=device if device is not None else spaces[0].device,
    )

@singledispatch
def unbatch_spaces(space: Space) -> Iterable[Space]:
    raise TypeError(
        f"The space provided to `unbatch_spaces` is not a batched Space instance, type: {type(space)}, {space}"
    )

@unbatch_spaces.register(BoxSpace)
def _unbatch_spaces_box(space: BoxSpace):
    assert len(space.shape) > 0, "Expected BoxSpace to be batched, but it is not."
    low = space._low
    high = space._high
    for i in range(space.shape[0]):
        yield BoxSpace(
            backend=space.backend,
            low=low[i] if low.shape[0] > i else low[0],
            high=high[i] if high.shape[0] > i else high[0],
            dtype=space.dtype,
            device=space.device,
        )

@unbatch_spaces.register(BinarySpace)
def _unbatch_spaces_binary(space: BinarySpace):
    for i in range(space.shape[0]):
        yield BinarySpace(
            backend=space.backend,
            shape=space.shape[1:],
            device=space.device,
            dtype=space.dtype,
        )

@unbatch_spaces.register(GraphSpace)
def _unbatch_spaces_graph(space: GraphSpace):
    assert len(space.batch_shape) > 0, "Expected GraphSpace to be batched, but it is not."
    for i in range(space.batch_shape[0]):
        yield GraphSpace(
            backend=space.backend,
            node_feature_space=copy.deepcopy(space.node_feature_space),
            edge_feature_space=copy.deepcopy(space.edge_feature_space),
            is_edge=space.is_edge,
            min_nodes=space.min_nodes,
            max_nodes=space.max_nodes,
            min_edges=space.min_edges,
            max_edges=space.max_edges,
            batch_shape=space.batch_shape[1:],
            device=space.device,
        )

@unbatch_spaces.register(DictSpace)
def _unbatch_spaces_dict(space: DictSpace):
    subspace_iterators = {}
    for key, subspace in space.spaces.items():
        subspace_iterators[key] = unbatch_spaces(subspace)
    for items in zip(*subspace_iterators.values()):
        yield DictSpace(
            backend=space.backend,
            spaces={key: value for key, value in zip(subspace_iterators.keys(), items)},
            device=space.device,
        )

@unbatch_spaces.register(TupleSpace)
def _unbatch_spaces_tuple(space: TupleSpace):
    for unbatched_subspaces_i in zip(*[unbatch_spaces(subspace) for subspace in space.spaces]):
        yield TupleSpace(
            backend=space.backend,
            spaces=unbatched_subspaces_i,
            device=space.device,
        )

def iterate(space: Space, items: Any) -> Iterator:
    for i in range(batch_size_data(items)):
        yield get_at(space, items, i)

@singledispatch
def get_at(space: Space, items: Any, index: ArrayAPIGetIndex) -> Any:
    raise TypeError(
        f"The space provided to `get_at` is not a batched space instance, type: {type(space)}, {space}"
    )

@get_at.register(BoxSpace)
@get_at.register(BinarySpace)
def _get_at_common(space: typing.Union[BoxSpace, BinarySpace], items: BArrayType, index: ArrayAPIGetIndex):
    return items[index]

@get_at.register(GraphSpace)
def _get_at_graph(space: GraphSpace, items: GraphInstance, index: ArrayAPIGetIndex):
    return GraphInstance(
        n_nodes=items.n_nodes[index],
        n_edges=items.n_edges[index] if items.n_edges is not None else None,
        nodes_features=items.nodes_features[index] if items.nodes_features is not None else None,
        edges_features=items.edges_features[index] if items.edges_features is not None else None,
        edges=items.edges[index] if items.edges is not None else None,
    )

@get_at.register(DictSpace)
def _get_at_dict(space: DictSpace, items: typing.Mapping[str, Any], index : ArrayAPIGetIndex):
    ret = {key: get_at(subspace, items[key], index) for key, subspace in space.spaces.items()}
    return ret

@get_at.register(TupleSpace)
def _get_at_tuple(space: TupleSpace, items: typing.Tuple[Any, ...], index : ArrayAPIGetIndex):    
    return tuple(get_at(subspace, item, index) for (subspace, item) in zip(space.spaces, items))

@singledispatch
def set_at(
    space: Space, items: Any, index: ArrayAPISetIndex, value: Any
) -> Any:
    raise TypeError(
        f"The space provided to `set_at` is not a batched space instance, type: {type(space)}, {space}"
    )


@set_at.register(BoxSpace)
@set_at.register(BinarySpace)
def _set_at_common(
    space: typing.Union[BoxSpace, BinarySpace],
    items: BArrayType,
    index: ArrayAPISetIndex,
    value: BArrayType,
) -> BArrayType:
    return space.backend.at(items)[index].set(value)

@set_at.register(GraphSpace)
def _set_at_graph(
    space: GraphSpace,
    items: GraphInstance,
    index: ArrayAPISetIndex,
    value: GraphInstance,
) -> GraphInstance:
    return GraphInstance(
        n_nodes=space.backend.at(items.n_nodes)[index].set(value.n_nodes),
        n_edges=(
            space.backend.at(items.n_edges)[index].set(value.n_edges)
            if items.n_edges is not None and value.n_edges is not None
            else None
        ),
        nodes_features=(
            space.backend.at(items.nodes_features)[index].set(value.nodes_features)
            if items.nodes_features is not None and value.nodes_features is not None
            else None
        ),
        edges_features=(
            space.backend.at(items.edges_features)[index].set(value.edges_features)
            if items.edges_features is not None and value.edges_features is not None
            else None
        ),
        edges=(
            space.backend.at(items.edges)[index].set(value.edges)
            if items.edges is not None and value.edges is not None
            else None
        ),
    )

@set_at.register(DictSpace)
def _set_at_dict(
    space: DictSpace,
    items: typing.Mapping[str, Any],
    index: ArrayAPISetIndex,
    value: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: set_at(subspace, items[key], index, value[key])
        for key, subspace in space.spaces.items()
    }

@set_at.register(TupleSpace)
def _set_at_tuple(
    space: TupleSpace,
    items: typing.Tuple[Any, ...],
    index: ArrayAPISetIndex,
    value: tuple[Any, ...],
) -> tuple[Any, ...]:
    return tuple(
        set_at(subspace, items[i], index, value[i])
        for i, subspace in enumerate(space.spaces)
    )

@singledispatch
def concatenate(
    space: Space, items: Iterable[Any]
) -> Any:
    raise TypeError(
        f"The space provided to `concatenate` is not a Space instance, type: {type(space)}, {space}"
    )

@concatenate.register(BoxSpace)
@concatenate.register(BinarySpace)
def _concatenate_base(
    space: typing.Union[BoxSpace, BinarySpace],
    items: Iterable,
) -> Any:
    return space.backend.stack(items, axis=0)

@concatenate.register(GraphSpace)
def _concatenate_graph(
    space: GraphSpace, items: Iterable[GraphInstance]
) -> GraphInstance:
    n_nodes = space.backend.stack([item.n_nodes for item in items], axis=0)
    n_edges = (
        space.backend.stack([item.n_edges for item in items], axis=0)
        if all(item.n_edges is not None for item in items)
        else None
    )
    nodes_features = (
        space.backend.stack([item.nodes_features for item in items], axis=0)
        if all(item.nodes_features is not None for item in items)
        else None
    )
    edges_features = (
        space.backend.stack([item.edges_features for item in items], axis=0)
        if all(item.edges_features is not None for item in items)
        else None
    )
    edges = (
        space.backend.stack([item.edges for item in items], axis=0)
        if all(item.edges is not None for item in items)
        else None
    )
    
    return GraphInstance(
        n_nodes=n_nodes,
        n_edges=n_edges,
        nodes_features=nodes_features,
        edges_features=edges_features,
        edges=edges,
    )

@concatenate.register(DictSpace)
def _concatenate_dict(
    space: DictSpace, items: Iterable
) -> dict[str, Any]:
    return {
        key: concatenate(subspace, [item[key] for item in items])
        for key, subspace in space.spaces.items()
    }

@concatenate.register(TupleSpace)
def _concatenate_tuple(
    space: TupleSpace, items: Iterable
) -> tuple[Any, ...]:
    if all(type(subspace) in concatenate.registry for subspace in space.spaces):
        return tuple(
            concatenate(subspace, [item[i] for item in items])
            for (i, subspace) in enumerate(space.spaces)
        )
    
    return tuple(items)
