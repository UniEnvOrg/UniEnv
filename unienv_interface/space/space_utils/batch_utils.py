from __future__ import annotations

import typing
import copy
from functools import singledispatch
from typing import Optional, Any, Iterable, Iterator
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType, ArrayAPIGetIndex, ArrayAPISetIndex
from unienv_interface.backends.numpy import NumpyComputeBackend
import numpy as np

from ..spaces import *

__all__ = [
    "batch_size",
    "batch_space",
    "batch_differing_spaces",
    "unbatch_spaces",
    "reshape_batch_size",
    "reshape_batch_size_in_data",
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
def reshape_batch_size(
    space : Space,
    old_batch_shape : typing.Tuple[int], 
    new_batch_shape : typing.Tuple[int]
) -> Space:
    raise TypeError(
        f"The space provided to `reshape_batch_size` is not supported, type: {type(space)}, {space}"
    )

@reshape_batch_size.register(BoxSpace)
def _reshape_batch_size_box(
    space: BoxSpace, 
    old_batch_shape: typing.Tuple[int],
    new_batch_shape: typing.Tuple[int]
) -> BoxSpace:
    assert np.prod(old_batch_shape) == np.prod(new_batch_shape), \
        f"Expected the product of old and new batch shapes to be equal, but got old {np.prod(old_batch_shape)} != new {np.prod(new_batch_shape)}"
    assert space.shape[:len(old_batch_shape)] == old_batch_shape, \
        f"Expected the beginning of the shape to match the old batch shape, but got {space.shape[:len(old_batch_shape)]} != {old_batch_shape}"
    new_shape = new_batch_shape + space.shape[len(old_batch_shape):]
    new_low = space.backend.reshape(space.low, new_shape)
    new_high = space.backend.reshape(space.high, new_shape)
    return BoxSpace(
        backend=space.backend,
        low=new_low,
        high=new_high,
        dtype=space.dtype,
        device=space.device,
        shape=new_shape,
    )

@reshape_batch_size.register(DynamicBoxSpace)
def _reshape_batch_size_dynamic_box(
    space: DynamicBoxSpace,
    old_batch_shape: typing.Tuple[int],
    new_batch_shape: typing.Tuple[int]
) -> DynamicBoxSpace:
    assert np.prod(old_batch_shape) == np.prod(new_batch_shape), \
        f"Expected the product of old and new batch shapes to be equal, but got old {np.prod(old_batch_shape)} != new {np.prod(new_batch_shape)}"
    assert space.shape_low[:len(old_batch_shape)] == old_batch_shape, \
        f"Expected the beginning of the shape_low to match the old batch shape, but got {space.shape_low[:len(old_batch_shape)]} != {old_batch_shape}"
    assert space.shape_high[:len(old_batch_shape)] == old_batch_shape, \
        f"Expected the beginning of the shape_high to match the old batch shape, but got {space.shape_high[:len(old_batch_shape)]} != {old_batch_shape}"
    
    new_shape_low = new_batch_shape + space.shape_low[len(old_batch_shape):]
    new_shape_high = new_batch_shape + space.shape_high[len(old_batch_shape):]
    
    value_resize_origin = tuple([
        space.shape_low[i] if space.shape_low[i] == space.shape_high[i] else 1
        for i in range(len(space.shape_low))
    ])
    value_resize_target = tuple([
        new_shape_low[i] if new_shape_low[i] == new_shape_high[i] else 1
        for i in range(len(new_shape_low))
    ])
    new_low = space.backend.reshape(space.get_low(value_resize_origin), value_resize_target)
    new_high = space.backend.reshape(space.get_high(value_resize_origin), value_resize_target)
    return DynamicBoxSpace(
        backend=space.backend,
        low=new_low,
        high=new_high,
        shape_low=new_shape_low,
        shape_high=new_shape_high,
        dtype=space.dtype,
        device=space.device,
        fill_value=space.fill_value,
    )

@reshape_batch_size.register(BinarySpace)
def _reshape_batch_size_binary(
    space: BinarySpace,
    old_batch_shape: typing.Tuple[int],
    new_batch_shape: typing.Tuple[int]
) -> BinarySpace:
    assert np.prod(old_batch_shape) == np.prod(new_batch_shape), \
        f"Expected the product of old and new batch shapes to be equal, but got old {np.prod(old_batch_shape)} != new {np.prod(new_batch_shape)}"
    assert space.shape[:len(old_batch_shape)] == old_batch_shape, \
        f"Expected the beginning of the shape to match the old batch shape, but got {space.shape[:len(old_batch_shape)]} != {old_batch_shape}"
    new_shape = new_batch_shape + space.shape[len(old_batch_shape):]
    return BinarySpace(
        backend=space.backend,
        shape=new_shape,
        dtype=space.dtype,
        device=space.device,
    )

@reshape_batch_size.register(GraphSpace)
def _reshape_batch_size_graph(
    space: GraphSpace,
    old_batch_shape: typing.Tuple[int],
    new_batch_shape: typing.Tuple[int]
) -> GraphSpace:
    assert np.prod(old_batch_shape) == np.prod(new_batch_shape), \
        f"Expected the product of old and new batch shapes to be equal, but got old {np.prod(old_batch_shape)} != new {np.prod(new_batch_shape)}"
    assert space.batch_shape[:len(old_batch_shape)] == old_batch_shape, \
        f"Expected the beginning of the batch shape to match the old batch shape, but got {space.batch_shape[:len(old_batch_shape)]} != {old_batch_shape}"
    new_shape = new_batch_shape + space.batch_shape[len(old_batch_shape):]
    return GraphSpace(
        backend=space.backend,
        node_feature_space=space.node_feature_space,
        edge_feature_space=space.edge_feature_space,
        is_edge=space.is_edge,
        min_nodes=space.min_nodes,
        max_nodes=space.max_nodes,
        min_edges=space.min_edges,
        max_edges=space.max_edges,
        batch_shape=new_shape,
        device=space.device,
    )

@reshape_batch_size.register(DictSpace)
def _reshape_batch_size_dict(
    space: DictSpace,
    old_batch_shape: typing.Tuple[int],
    new_batch_shape: typing.Tuple[int]
) -> DictSpace:
    new_spaces = {
        key: reshape_batch_size(subspace, old_batch_shape, new_batch_shape)
        for key, subspace in space.spaces.items()
    }
    return DictSpace(
        backend=space.backend,
        spaces=new_spaces,
        device=space.device,
    )

@reshape_batch_size.register(TupleSpace)
def _reshape_batch_size_tuple(
    space: TupleSpace,
    old_batch_shape: typing.Tuple[int],
    new_batch_shape: typing.Tuple[int]
) -> TupleSpace:
    new_spaces = [
        reshape_batch_size(subspace, old_batch_shape, new_batch_shape)
        for subspace in space.spaces
    ]
    return TupleSpace(
        backend=space.backend,
        spaces=new_spaces,
        device=space.device,
    )

@reshape_batch_size.register(BatchedSpace)
def _reshape_batch_size_batched(
    space: BatchedSpace,
    old_batch_shape: typing.Tuple[int],
    new_batch_shape: typing.Tuple[int]
) -> BatchedSpace:
    assert len(old_batch_shape) <= len(space.batch_shape) and old_batch_shape == space.batch_shape[:len(old_batch_shape)], \
        f"Expected the old batch shape to be a prefix of the current batch shape, but got old {old_batch_shape} != current {space.batch_shape}"
    return BatchedSpace(
        single_space=space.single_space,
        batch_shape=new_batch_shape + space.batch_shape[len(old_batch_shape):]
    )

def reshape_batch_size_in_data(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    data : Any, 
    old_batch_shape: typing.Tuple[int], 
    new_batch_shape: typing.Tuple[int]
) -> Any:
    assert np.prod(old_batch_shape) == np.prod(new_batch_shape), \
        f"Expected the product of old and new batch shapes to be equal, but got old {np.prod(old_batch_shape)} != new {np.prod(new_batch_shape)}"
    
    if data is None:
        return None
    
    if backend.is_backendarray(data):
        assert data.shape[:len(old_batch_shape)] == old_batch_shape, \
            f"Expected the beginning of the shape to match the old batch shape, but got {data.shape[:len(old_batch_shape)]} != {old_batch_shape}"
        data = data.reshape(new_batch_shape + data.shape[len(old_batch_shape):])
    elif isinstance(data, np.ndarray) and data.dtype == object:
        assert data.shape[:len(old_batch_shape)] == old_batch_shape, \
            f"Expected the beginning of the shape to match the old batch shape, but got {data.shape[:len(old_batch_shape)]} != {old_batch_shape}"
        data = data.reshape(new_batch_shape + data.shape[len(old_batch_shape):])
    elif isinstance(data, GraphInstance):
        assert data.n_nodes.shape[:len(old_batch_shape)] == old_batch_shape, \
            f"Expected the beginning of the n_nodes shape to match the old batch shape, but got {data.n_nodes.shape[:len(old_batch_shape)]} != {old_batch_shape}"
        data = GraphInstance(
            n_nodes=backend.reshape(data.n_nodes, new_batch_shape + data.n_nodes.shape[len(old_batch_shape):]),
            n_edges=None if data.n_edges is None else backend.reshape(data.n_edges, new_batch_shape + data.n_edges.shape[len(old_batch_shape):]),
            nodes_features=None if data.nodes_features is None else backend.reshape(data.nodes_features, new_batch_shape + data.nodes_features.shape[len(old_batch_shape):]),
            edges_features=None if data.edges_features is None else backend.reshape(data.edges_features, new_batch_shape + data.edges_features.shape[len(old_batch_shape):]),
            edges=None if data.edges is None else backend.reshape(data.edges, new_batch_shape + data.edges.shape[len(old_batch_shape):]),
        )
    elif isinstance(data, typing.Mapping):
        data = {
            key: reshape_batch_size_in_data(
                backend,
                value,
                old_batch_shape,
                new_batch_shape
            )
            for key, value in data.items()
        }
    elif isinstance(data, typing.Sequence):
        return tuple([
            reshape_batch_size_in_data(
                backend,
                item,
                old_batch_shape,
                new_batch_shape
            ) for item in data
        ])
    else:
        raise TypeError(
            f"Unable to reshape batch size of data, type: {type(data)}, {data}"
        )
    
    return data

@singledispatch
def swap_batch_dims(space: Space, dim1: int, dim2: int) -> Space:
    raise TypeError(
        f"The space provided to `swap_batch_dims` is not supported, type: {type(space)}, {space}"
    )

@swap_batch_dims.register(BoxSpace)
def _swap_batch_dims_box(space: BoxSpace, dim1: int, dim2: int):
    return BoxSpace(
        backend=space.backend,
        low=_tensor_transpose(space.backend, space.low, dim1, dim2),
        high=_tensor_transpose(space.backend, space.high, dim1, dim2),
        dtype=space.dtype,
        device=space.device,
        shape=_shape_transpose(space.shape, dim1, dim2),
    )

@swap_batch_dims.register(DynamicBoxSpace)
def _swap_batch_dims_dynamic_box(space: DynamicBoxSpace, dim1: int, dim2: int):
    return DynamicBoxSpace(
        backend=space.backend,
        low=_tensor_transpose(space.backend, space.low, dim1, dim2),
        high=_tensor_transpose(space.backend, space.high, dim1, dim2),
        shape_low=_shape_transpose(space.shape_low, dim1, dim2),
        shape_high=_shape_transpose(space.shape_high, dim1, dim2),
        dtype=space.dtype,
        device=space.device,
        fill_value=space.fill_value,
    )

@swap_batch_dims.register(BinarySpace)
def _swap_batch_dims_binary(space: BinarySpace, dim1: int, dim2: int):
    return BinarySpace(
        backend=space.backend,
        shape=_shape_transpose(space.shape, dim1, dim2),
        dtype=space.dtype,
        device=space.device,
    )

@swap_batch_dims.register(GraphSpace)
def _swap_batch_dims_graph(space: GraphSpace, dim1: int, dim2: int):
    return GraphSpace(
        backend=space.backend,
        node_feature_space=space.node_feature_space,
        edge_feature_space=space.edge_feature_space,
        is_edge=space.is_edge,
        min_nodes=space.min_nodes,
        max_nodes=space.max_nodes,
        min_edges=space.min_edges,
        max_edges=space.max_edges,
        batch_shape=_shape_transpose(space.batch_shape, dim1, dim2),
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

@swap_batch_dims.register(BatchedSpace)
def _swap_batch_dims_batched(space: BatchedSpace, dim1: int, dim2: int):
    return BatchedSpace(
        single_space=space.single_space,
        batch_shape=_shape_transpose(space.batch_shape, dim1, dim2)
    )

def swap_batch_dims_in_data(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType], 
    data: Any, 
    dim1: int, 
    dim2: int
) -> Any:
    if backend.is_backendarray(data):
        return _tensor_transpose(backend, data, dim1, dim2)
    elif isinstance(data, np.ndarray) and data.dtype != object:
        return _tensor_transpose(NumpyComputeBackend, data, dim1, dim2)
    elif isinstance(data, GraphInstance):
        return GraphInstance(
            n_nodes=_tensor_transpose(backend, data.n_nodes, dim1, dim2),
            n_edges=None if data.n_edges is None else _tensor_transpose(backend, data.n_edges, dim1, dim2),
            nodes_features=None if data.nodes_features is None else _tensor_transpose(backend, data.nodes_features, dim1, dim2),
            edges_features=None if data.edges_features is None else _tensor_transpose(backend, data.edges_features, dim1, dim2),
            edges=None if data.edges is None else _tensor_transpose(backend, data.edges, dim1, dim2),
        )
    elif isinstance(data, typing.Mapping):
        return {
            key: swap_batch_dims_in_data(backend, value, dim1, dim2)
            for key, value in data.items()
        }
    elif isinstance(data, typing.Sequence):
        return tuple([
            swap_batch_dims_in_data(backend, item, dim1, dim2) for item in data
        ])
    else:
        raise TypeError(f"Unable to determine batch size of data, type: {type(data)}")

@singledispatch
def batch_size(space: Space) -> Optional[int]:
    """Space does not support batching."""
    return None

@batch_size.register(BoxSpace)
@batch_size.register(BinarySpace)
def _batch_size_box(space: typing.Union[BoxSpace, BinarySpace]):
    return space.shape[0] if len(space.shape) > 0 else None

@batch_size.register(DynamicBoxSpace)
def _batch_size_dynamic_box(space: DynamicBoxSpace):
    return space.shape_low[0] if len(space.shape_low) > 0 and space.shape_high[0] == space.shape_low[0] else None

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

@batch_size.register(BatchedSpace)
def _batch_size_batched(space: BatchedSpace):
    return space.batch_shape[0] if len(space.batch_shape) > 0 else None

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

@batch_space.register(DynamicBoxSpace)
def _batch_space_dynamic_box(space: DynamicBoxSpace, n: int = 1):
    return DynamicBoxSpace(
        backend=space.backend,
        low=space._low[None],
        high=space._high[None],
        shape_low=(n,) + space.shape_low,
        shape_high=(n,) + space.shape_high,
        dtype=space.dtype,
        device=space.device,
        fill_value=space.fill_value,
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

@batch_space.register(BatchedSpace)
def _batch_space_batched(space: BatchedSpace, n: int = 1):
    return BatchedSpace(
        single_space=space.single_space,
        batch_shape=(n,) + space.batch_shape,
    )

@batch_space.register(TextSpace)
@batch_space.register(UnionSpace)
def _batch_space_text(space: typing.Union[TextSpace, UnionSpace], n: int = 1):
    return BatchedSpace(
        space,
        batch_shape=(n,),
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

@batch_differing_spaces.register(DynamicBoxSpace)
def _batch_differing_spaces_dynamic_box(spaces: typing.Sequence[DynamicBoxSpace], device : Optional[Any] = None):
    assert all(
        spaces[0].dtype == space.dtype for space in spaces
    ), f"Expected all dtypes to be equal, actually {[space.dtype for space in spaces]}"
    assert all(
        spaces[0].shape_low == space.shape_low for space in spaces
    ), f"Expected all DynamicBoxSpace.low shape to be equal, actually {[space.low.shape for space in spaces]}"
    assert all(
        spaces[0].shape_high == space.shape_high for space in spaces
    ), f"Expected all DynamicBoxSpace.high shape to be equal, actually {[space.high.shape for space in spaces]}"
    assert all(
        spaces[0].fill_value == space.fill_value for space in spaces
    ), f"Expected all DynamicBoxSpace.fill_value to be equal, actually {[space.fill_value for space in spaces]}"

    backend = spaces[0].backend
    target_low = backend.stack(backend.broadcast_arrays(*[space._low for space in spaces]), axis=0)
    target_high = backend.stack(backend.broadcast_arrays(*[space._high for space in spaces]), axis=0)

    return DynamicBoxSpace(
        backend=backend,
        low=target_low,
        high=target_high,
        shape_low=(len(spaces),) + spaces[0].shape_low,
        shape_high=(len(spaces),) + spaces[0].shape_high,
        dtype=spaces[0].dtype,
        device=device if device is not None else spaces[0].device,
        fill_value=spaces[0].fill_value,
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
            shape=space.shape[1:],
        )

@unbatch_spaces.register(DynamicBoxSpace)
def _unbatch_spaces_dynamic_box(space: DynamicBoxSpace):
    assert len(space.shape_low) > 0, "Expected DynamicBoxSpace to be batched, but it is not."
    low = space._low
    high = space._high
    for i in range(space.shape_low[0]):
        yield DynamicBoxSpace(
            backend=space.backend,
            low=low[i] if low.shape[0] > i else low[0],
            high=high[i] if high.shape[0] > i else high[0],
            shape_low=space.shape_low[1:],
            shape_high=space.shape_high[1:],
            dtype=space.dtype,
            device=space.device,
            fill_value=space.fill_value,
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

@unbatch_spaces.register(BatchedSpace)
def _unbatch_spaces_batched(space: BatchedSpace):
    assert len(space.batch_shape) > 0, "Expected BatchedSpace to be batched, but it is not."
    unbatched_space = space.single_space if len(space.batch_shape) == 1 else BatchedSpace(
        single_space=space.single_space,
        batch_shape=space.batch_shape[1:],
    )
    for i in range(space.batch_shape[0]):
        yield unbatched_space

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
@get_at.register(DynamicBoxSpace)
def _get_at_common(space: typing.Union[BoxSpace, BinarySpace, DynamicBoxSpace], items: BArrayType, index: ArrayAPIGetIndex):
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

@get_at.register(BatchedSpace)
def _get_at_batched(space: BatchedSpace, items: np.ndarray, index: ArrayAPIGetIndex) -> typing.Union[np.ndarray, Any]:
    return items[index]

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
    value: typing.Union[BArrayType, float, int],
) -> BArrayType:
    return space.backend.at(items)[index].set(value)

@set_at.register(DynamicBoxSpace)
def _set_at_dynamic_box(
    space: DynamicBoxSpace,
    items: BArrayType,
    index: ArrayAPISetIndex,
    value: typing.Union[BArrayType, float, int],
) -> BArrayType:
    try:
        return space.backend.at(items)[index].set(value)
    except Exception as e:
        value = space.pad_data(value, start_axis=1)
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
    value: typing.Union[typing.Dict[str, Any], float, int],
) -> dict[str, Any]:
    return {
        key: set_at(subspace, items[key], index, value[key]) if not isinstance(value, (float, int)) else set_at(subspace, items[key], index, value)
        for key, subspace in space.spaces.items()
    }

@set_at.register(TupleSpace)
def _set_at_tuple(
    space: TupleSpace,
    items: typing.Tuple[Any, ...],
    index: ArrayAPISetIndex,
    value: typing.Union[typing.Tuple[Any, ...], float, int],
) -> tuple[Any, ...]:
    return tuple(
        set_at(subspace, items[i], index, value[i]) if not isinstance(value, (float, int)) else set_at(subspace, items[i], index, value)
        for i, subspace in enumerate(space.spaces)
    )

@set_at.register(BatchedSpace)
def _set_at_batched(
    space: BatchedSpace,
    items: np.ndarray,
    index: ArrayAPISetIndex,
    value: typing.Union[np.ndarray, Any],
) -> np.ndarray:
    new_data = items.copy()
    new_data[index] = value
    return new_data

@singledispatch
def concatenate(
    space: Space, items: Iterable[Any], axis : int = 0,
) -> Any:
    raise TypeError(
        f"The space provided to `concatenate` is not a Space instance, type: {type(space)}, {space}"
    )

@concatenate.register(BoxSpace)
@concatenate.register(BinarySpace)
def _concatenate_base(
    space: typing.Union[BoxSpace, BinarySpace],
    items: Iterable[BArrayType],
    axis: int = 0,
) -> Any:
    return space.backend.stack(items, axis=axis)

@concatenate.register(DynamicBoxSpace)
def _concatenate_dynamic_box(
    space: DynamicBoxSpace, items: Iterable[BArrayType], axis: int = 0
) -> BArrayType:
    # Ensure all items have the same shape
    shapes = [item.shape for item in items]
    assert all(
        len(shape) == len(shapes[0]) for shape in shapes
    ), f"Expected all items to have the same number of dimensions, actual shapes: {[shape for shape in shapes]}"
    if not all(shape == shapes[0] for shape in shapes):
        if axis < 0:
            axis += len(shapes[0])
        max_shape = list(shapes[0])
        for shape in shapes:
            for i in range(len(shape)):
                max_shape[i] = max(max_shape[i], shape[i])
        items = list(items)
        for i in range(len(items)):
            if items[i].shape != max_shape:
                for pad_axis in range(len(max_shape)):
                    items[i] = space.pad_array_on_axis(
                        space.backend,
                        items[i],
                        pad_axis,
                        max_shape[pad_axis],
                        fill_value=space.fill_value,
                    )
    
    # Stack the items along the specified axis
    return space.backend.stack(items, axis=axis)

@concatenate.register(GraphSpace)
def _concatenate_graph(
    space: GraphSpace, items: Iterable[GraphInstance], axis: int = 0
) -> GraphInstance:
    n_nodes = space.backend.stack([item.n_nodes for item in items], axis=axis)
    n_edges = (
        space.backend.stack([item.n_edges for item in items], axis=axis)
        if all(item.n_edges is not None for item in items)
        else None
    )
    nodes_features = (
        space.backend.stack([item.nodes_features for item in items], axis=axis)
        if all(item.nodes_features is not None for item in items)
        else None
    )
    edges_features = (
        space.backend.stack([item.edges_features for item in items], axis=axis)
        if all(item.edges_features is not None for item in items)
        else None
    )
    edges = (
        space.backend.stack([item.edges for item in items], axis=axis)
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
    space: DictSpace, items: Iterable, axis: int = 0
) -> dict[str, Any]:
    return {
        key: concatenate(subspace, [item[key] for item in items], axis=axis)
        for key, subspace in space.spaces.items()
    }

@concatenate.register(TupleSpace)
def _concatenate_tuple(
    space: TupleSpace, items: Iterable, axis: int = 0
) -> tuple[Any, ...]:
    if all(type(subspace) in concatenate.registry for subspace in space.spaces):
        return tuple(
            concatenate(subspace, [item[i] for item in items], axis=axis)
            for (i, subspace) in enumerate(space.spaces)
        )
    
    return tuple(items)

@concatenate.register(BatchedSpace)
def _concatenate_batched(
    space: BatchedSpace, items: Iterable, axis: int = 0
) -> Any:
    items = list(items)
    if len(items) == 0:
        return np.array([], dtype=object)
    if isinstance(items[0], np.ndarray) and items[0].dtype == object:
        return np.concatenate(items, axis=axis)
    else:
        assert axis == 0, "Expected axis to be 0 when concatenating non-numpy arrays"
        return np.asarray(items, dtype=object)