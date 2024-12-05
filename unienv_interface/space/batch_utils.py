from __future__ import annotations

import typing
import copy
from functools import singledispatch
from typing import Optional, Any, Iterable, Iterator
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

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
    "batch_size",
    "batch_space",
    "batch_differing_spaces",
    "unbatch_spaces",
    "swap_batch_dims",
    "swap_batch_dims_in_data",
    "iterate",
    "concatenate",
    "read_batched_data_with_mask",
    "write_batched_data_with_mask"
]

def _tensor_transpose(backend : ComputeBackend, tensor : BArrayType, dim1 : int, dim2 : int) -> BArrayType:
    dims = list(range(tensor.ndim))
    dims[dim1] = dim2
    dims[dim2] = dim1
    return backend.array_api_namespace.permute_dims(tensor, axes=tuple(dims))

def _shape_transpose(shape : tuple[int, ...], dim1 : int, dim2 : int) -> tuple[int, ...]:
    shape = list(shape)
    shape[dim1], shape[dim2] = shape[dim2], shape[dim1]
    return tuple(shape)

@singledispatch
def swap_batch_dims(space: Space, dim1: int, dim2: int) -> Space:
    raise TypeError(
        f"The space provided to `swap_batch_dims` is not a Space instance, type: {type(space)}, {space}"
    )

@swap_batch_dims.register(Box)
def _swap_batch_dims_box(space: Box, dim1: int, dim2: int):
    return Box(
        backend=space.backend,
        low=_tensor_transpose(space.backend, space.low, dim1, dim2),
        high=_tensor_transpose(space.backend, space.high, dim1, dim2),
        dtype=space.dtype,
        device=space.device,
    )

@swap_batch_dims.register(MultiDiscrete)
def _swap_batch_dims_multi_discrete(space: MultiDiscrete, dim1: int, dim2: int):
    return MultiDiscrete(
        backend=space.backend,
        nvec=_tensor_transpose(space.backend, space.nvec, dim1, dim2),
        start=_tensor_transpose(space.backend, space.start, dim1, dim2),
        dtype=space.dtype,
        device=space.device,
    )

@swap_batch_dims.register(MultiBinary)
def _swap_batch_dims_multi_binary(space: MultiBinary, dim1: int, dim2: int):
    return MultiBinary(
        backend=space.backend,
        shape=_shape_transpose(space.shape, dim1, dim2),
        dtype=space.dtype,
        device=space.device,
    )

@swap_batch_dims.register(Dict)
def _swap_batch_dims_dict(space: Dict, dim1: int, dim2: int):
    return Dict(
        backend=space.backend,
        spaces={key: swap_batch_dims(subspace, dim1, dim2) for key, subspace in space.spaces.items()},
        device=space.device,
    )

@swap_batch_dims.register(Tuple)
def _swap_batch_dims_tuple(space: Tuple, dim1: int, dim2: int):
    assert all(type(subspace) in swap_batch_dims.registry for subspace in space.spaces), "Expected all subspaces in Tuple to be swappable"
    return Tuple(
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

@swap_batch_dims_in_data.register(Box)
@swap_batch_dims_in_data.register(MultiDiscrete)
@swap_batch_dims_in_data.register(MultiBinary)
def _swap_batch_dims_in_data_common(space: Box, data: BArrayType, dim1: int, dim2: int):
    return _tensor_transpose(space.backend, data, dim1, dim2)

@swap_batch_dims_in_data.register(Dict)
def _swap_batch_dims_in_data_dict(space: Dict, data: dict[str, Any], dim1: int, dim2: int):
    return {
        key: swap_batch_dims_in_data(subspace, data[key], dim1, dim2)
        for key, subspace in space.spaces.items()
    }

@swap_batch_dims_in_data.register(Tuple)
def _swap_batch_dims_in_data_tuple(space: Tuple, data: tuple[Any, ...], dim1: int, dim2: int):
    assert all(type(subspace) in swap_batch_dims_in_data.registry for subspace in space.spaces), "Expected all subspaces in Tuple to be swappable"
    return tuple(
        swap_batch_dims_in_data(subspace, data[i], dim1, dim2)
        for i, subspace in enumerate(space.spaces)
    )

@singledispatch
def batch_size(space: Space) -> int:
    raise TypeError(
        f"The space provided to `batch_size` is not a batched Space instance, type: {type(space)}, {space}"
    )

@batch_size.register(Box)
@batch_size.register(MultiDiscrete)
@batch_size.register(MultiBinary)
def _batch_size_box(space: Box):
    return space.shape[0]

@batch_size.register(Dict)
def _batch_size_dict(space: Dict):
    for subspace in space.spaces.values():
        try:
            return batch_size(subspace)
        except:
            continue

@batch_size.register(Tuple)
def _batch_size_tuple(space: Tuple):
    if all(type(subspace) in batch_size.registry for subspace in space.spaces):
        for subspace in space.spaces:
            try:
                return batch_size(subspace)
            except:
                continue

    return len(space.spaces)

@singledispatch
def batch_space(space: Space, n: int = 1) -> Space:
    raise TypeError(
        f"The space provided to `batch_space` is not a Space instance, type: {type(space)}, {space}"
    )

@batch_space.register(Box)
def _batch_space_box(space: Box, n: int = 1):
    return Box(
        backend=space.backend,
        low=space.backend.array_api_namespace.stack([space.low] * n),
        high=space.backend.array_api_namespace.stack([space.high] * n),
        dtype=space.dtype,
        device=space.device,
    )

@batch_space.register(Dict)
def _batch_space_dict(space: Dict, n: int = 1):
    return Dict(
        backend=space.backend,
        spaces={key: batch_space(subspace, n=n) for key, subspace in space.spaces.items()},
        device=space.device,
    )

@batch_space.register(Discrete)
def _batch_space_discrete(space: Discrete, n: int = 1):
    return MultiDiscrete(
        backend=space.backend,
        nvec=space.backend.array_api_namespace.full((n,), space.n, dtype=space.dtype),
        start=space.backend.array_api_namespace.full((n,), space.start, dtype=space.dtype) if space.start != 0 else None,
        dtype=space.dtype,
        device=space.device,
    )

@batch_space.register(MultiBinary)
def _batch_space_multibinary(space: MultiBinary, n: int = 1):
    return MultiBinary(
        backend=space.backend,
        shape=(n,) + space.shape,
        device=space.device,
        dtype=space.dtype,
    )

@batch_space.register(MultiDiscrete)
def _batch_space_multidiscrete(space: MultiDiscrete, n: int = 1):
    return MultiDiscrete(
        backend=space.backend,
        nvec=space.backend.array_api_namespace.stack([space.nvec] * n),
        start=space.backend.array_api_namespace.stack([space.start] * n),
        dtype=space.dtype,
        device=space.device,
    )

@batch_space.register(Tuple)
def _batch_space_tuple(space: Tuple, n: int = 1):
    return Tuple(
        backend=space.backend,
        spaces=[batch_space(subspace, n=n) for subspace in space.spaces],
        device=space.device,
    )

@batch_space.register(Graph)
@batch_space.register(Text)
@batch_space.register(Sequence)
@batch_space.register(Union)
@batch_space.register(Space)
def _batch_space_custom(space: Graph | Text | Sequence | Union, n: int = 1):
    batched_space = Tuple(
        backend=space.backend,
        spaces=[copy.deepcopy(space) for _ in range(n)],
        device=space.device,
    )
    return batched_space

@singledispatch
def batch_differing_spaces(spaces: typing.Sequence[Space], device : Optional[Any] = None) -> Space:
    assert len(spaces) > 0, "Expects a non-empty list of spaces"
    assert all(
        isinstance(space, type(spaces[0])) for space in spaces
    ), f"Expects all spaces to be the same shape, actual types: {[type(space) for space in spaces]}"
    assert all(
        spaces[0].backend == space.backend for space in spaces
    ), f"Expects all spaces to have the same backend, actual backends: {[space.backend for space in spaces]}"
    assert (
        type(spaces[0]) in batch_differing_spaces.registry
    ), f"Requires the Space type to have a registered `batch_differing_space`, current list: {batch_differing_spaces.registry}"

    return batch_differing_spaces.dispatch(type(spaces[0]))(spaces, device)

@batch_differing_spaces.register(Box)
def _batch_differing_spaces_box(spaces: typing.Sequence[Box], device : Optional[Any] = None):
    assert all(
        spaces[0].dtype == space.dtype for space in spaces
    ), f"Expected all dtypes to be equal, actually {[space.dtype for space in spaces]}"
    assert all(
        spaces[0].low.shape == space.low.shape for space in spaces
    ), f"Expected all Box.low shape to be equal, actually {[space.low.shape for space in spaces]}"
    assert all(
        spaces[0].high.shape == space.high.shape for space in spaces
    ), f"Expected all Box.high shape to be equal, actually {[space.high.shape for space in spaces]}"

    backend = spaces[0].backend
    return Box(
        backend=backend,
        low=backend.array_api_namespace.stack([space.low for space in spaces]),
        high=backend.array_api_namespace.stack([space.high for space in spaces]),
        dtype=spaces[0].dtype,
        device=device if device is not None else spaces[0].device,
    )

@batch_differing_spaces.register(Dict)
def _batch_differing_spaces_dict(spaces: typing.Sequence[Dict], device : Optional[Any] = None):
    assert all(spaces[0].keys() == space.keys() for space in spaces)

    return Dict(
        backend=spaces[0].backend,
        spaces={
            key: batch_differing_spaces([space[key] for space in spaces], device)
            for key in spaces[0].keys()
        },
        device=device if device is not None else spaces[0].device,
    )

@batch_differing_spaces.register(Discrete)
def _batch_differing_spaces_discrete(spaces: typing.Sequence[Discrete], device : Optional[Any] = None):
    backend = spaces[0].backend
    return MultiDiscrete(
        backend=backend,
        nvec=backend.array_api_namespace.asarray([space.n for space in spaces], dtype=spaces[0].dtype, device=device),
        start=backend.array_api_namespace.asarray([space.start for space in spaces], dtype=spaces[0].dtype, device=device),
        dtype=spaces[0].dtype,
        device=device if device is not None else spaces[0].device,
    )

@batch_differing_spaces.register(MultiBinary)
def _batch_differing_spaces_multi_binary(spaces: typing.Sequence[MultiBinary], device : Optional[Any] = None):
    assert all(spaces[0].shape == space.shape for space in spaces)
    
    backend=spaces[0].backend
    return MultiBinary(
        backend=backend,
        shape=(len(spaces),) + spaces[0].shape,
        dtype=spaces[0].dtype,
        device=device if device is not None else spaces[0].device,
    )

@batch_differing_spaces.register(MultiDiscrete)
def _batch_differing_spaces_multi_discrete(spaces: typing.Sequence[MultiDiscrete], device : Optional[Any] = None):
    assert all(
        spaces[0].dtype == space.dtype for space in spaces
    ), f"Expected all dtypes to be equal, actually {[space.dtype for space in spaces]}"
    assert all(
        spaces[0].nvec.shape == space.nvec.shape for space in spaces
    ), f"Expects all MultiDiscrete.nvec shape, actually {[space.nvec.shape for space in spaces]}"
    assert all(
        spaces[0].start.shape == space.start.shape for space in spaces
    ), f"Expects all MultiDiscrete.start shape, actually {[space.start.shape for space in spaces]}"

    backend = spaces[0].backend
    return MultiDiscrete(
        backend=backend,
        nvec=backend.array_api_namespace.stack([space.nvec for space in spaces]),
        start=backend.array_api_namespace.stack([space.start for space in spaces]),
        dtype=spaces[0].dtype,
        device=device if device is not None else spaces[0].device,
    )


@batch_differing_spaces.register(Tuple)
def _batch_differing_spaces_tuple(spaces: typing.Sequence[Tuple], device : Optional[Any] = None):
    return Tuple(
        backend=spaces[0].backend,
        spaces=[
            batch_differing_spaces(subspaces, device)
            for subspaces in zip(*[space.spaces for space in spaces])
        ],
        device=device if device is not None else spaces[0].device,
    )

@batch_differing_spaces.register(Graph)
@batch_differing_spaces.register(Text)
@batch_differing_spaces.register(Sequence)
@batch_differing_spaces.register(Union)
def _batch_differing_spaces_undefined(spaces: typing.Sequence[Graph | Text | Sequence | Union], device : Optional[Any] = None):
    return Tuple(
        backend=spaces[0].backend,
        spaces=[copy.deepcopy(space) for space in spaces],
        device=spaces[0].device, 
    )

@singledispatch
def unbatch_spaces(space: Space) -> Iterable[Space]:
    raise TypeError(
        f"The space provided to `unbatch_spaces` is not a batched Space instance, type: {type(space)}, {space}"
    )

@unbatch_spaces.register(Box)
def _unbatch_spaces_box(space: Box):
    for i in range(space.shape[0]):
        yield Box(
            backend=space.backend,
            low=space.low[i],
            high=space.high[i],
            dtype=space.dtype,
            device=space.device,
        )

@unbatch_spaces.register(Dict)
def _unbatch_spaces_dict(space: Dict):
    subspace_iterators = {}
    for key, subspace in space.spaces.items():
        subspace_iterators[key] = unbatch_spaces(subspace)
    for items in zip(*subspace_iterators.values()):
        yield Dict(
            backend=space.backend,
            spaces={key: value for key, value in zip(subspace_iterators.keys(), items)},
            device=space.device,
        )

@unbatch_spaces.register(MultiDiscrete)
def _unbatch_spaces_multi_discrete(space: MultiDiscrete):
    for i in range(space.nvec.shape[0]):
        if len(space.shape) == 1:
            yield Discrete(
                backend=space.backend,
                n=int(space.nvec[i]),
                start=int(space.start[i]),
                device=space.device,
                dtype=space.dtype,
            )
        else:
            yield MultiDiscrete(
                backend=space.backend,
                nvec=space.nvec[i],
                start=space.start[i],
                device=space.device,
                dtype=space.dtype,
            )

@unbatch_spaces.register(MultiBinary)
def _unbatch_spaces_multi_binary(space: MultiBinary):
    for i in range(space.shape[0]):
        yield MultiBinary(
            backend=space.backend,
            shape=space.shape[1:],
            device=space.device,
            dtype=space.dtype,
        )

@unbatch_spaces.register(Tuple)
def _unbatch_spaces_tuple(space: Tuple):
    if all(type(subspace) in unbatch_spaces.registry for subspace in space.spaces):
        for unbatched_subspaces_i in zip(*[unbatch_spaces(subspace) for subspace in space.spaces]):
            yield Tuple(
                backend=space.backend,
                spaces=unbatched_subspaces_i,
                device=space.device,
            )

    yield from space.spaces

@singledispatch
def iterate(space: Space, items: Any) -> Iterator:
    if isinstance(space, Space):
        raise NotImplementedError(
            f"Space of type `{type(space)}` doesn't have an registered `iterate` function. Register `{type(space)}` for `iterate` to support it."
        )
    else:
        raise TypeError(
            f"The space provided to `iterate` is not a Space instance, type: {type(space)}, {space}"
        )

@iterate.register(Discrete)
def _iterate_discrete(space: Discrete, items: Iterable):
    raise TypeError("Unable to iterate over a space of type `Discrete`.")

@iterate.register(Box)
@iterate.register(MultiDiscrete)
@iterate.register(MultiBinary)
def _iterate_base(space: Box | MultiDiscrete | MultiBinary, items: Any):
    try:
        for i in items.shape[0]:
            yield items[i]
    except TypeError as e:
        raise TypeError(
            f"Unable to iterate over the following elements: {items}"
        ) from e

@iterate.register(Dict)
def _iterate_dict(space: Dict, items: dict[str, Any]):
    keys, values = zip(
        *[
            (key, iterate(subspace, items[key]))
            for key, subspace in space.spaces.items()
        ]
    )
    for item in zip(*values):
        yield {key: value for key, value in zip(keys, item)}

@iterate.register(Tuple)
def _iterate_tuple(space: Tuple, items: tuple[Any, ...]):
    # If this is a tuple of custom subspaces only, then simply iterate over items
    if all(type(subspace) in iterate.registry for subspace in space.spaces):
        return zip(*[iterate(subspace, items[i]) for i, subspace in enumerate(space.spaces)])

    try:
        return iter(items)
    except Exception as e:
        unregistered_spaces = [
            type(subspace)
            for subspace in space.spaces
            if type(subspace) not in iterate.registry
        ]
        raise TypeError(
            f"Could not iterate through {space} as no custom iterate function is registered for {unregistered_spaces} and `iter(items)` raised the following error: {e}."
        ) from e


@singledispatch
def concatenate(
    space: Space, items: Iterable[Any]
) -> Any:
    raise TypeError(
        f"The space provided to `concatenate` is not a Space instance, type: {type(space)}, {space}"
    )

@concatenate.register(Box)
@concatenate.register(Discrete)
@concatenate.register(MultiDiscrete)
@concatenate.register(MultiBinary)
def _concatenate_base(
    space: Box | MultiDiscrete | MultiBinary,
    items: Iterable,
) -> Any:
    return space.backend.array_api_namespace.stack(items, axis=0)

@concatenate.register(Dict)
def _concatenate_dict(
    space: Dict, items: Iterable
) -> dict[str, Any]:
    return {
        key: concatenate(subspace, [item[key] for item in items])
        for key, subspace in space.spaces.items()
    }

@concatenate.register(Tuple)
def _concatenate_tuple(
    space: Tuple, items: Iterable
) -> tuple[Any, ...]:
    if all(type(subspace) in concatenate.registry for subspace in space.spaces):
        return tuple(
            concatenate(subspace, [item[i] for item in items])
            for (i, subspace) in enumerate(space.spaces)
        )
    
    return tuple(items)

@singledispatch
def read_batched_data_with_mask(
    batched_space : Space,
    data : Any,
    mask : BArrayType,
):
    raise TypeError(
        f"The space provided to `read_batched_data_with_mask` is not a Space instance, type: {type(batched_space)}, {batched_space}"
    )

@read_batched_data_with_mask.register(Box)
@read_batched_data_with_mask.register(MultiDiscrete)
@read_batched_data_with_mask.register(MultiBinary)
def _read_batched_data_with_mask_base(
    batched_space : Box | MultiDiscrete | MultiBinary,
    data : BArrayType,
    mask : BArrayType,
):
    assert batched_space.backend.dtype_is_boolean(mask.dtype), f"Expected mask to be of boolean dtype, actually {mask.dtype}"
    assert len(mask.shape) == 1, f"Expected mask to be 1D, actually {mask.shape}"
    return data[mask]

@read_batched_data_with_mask.register(Dict)
def _read_batched_data_with_mask_dict(
    batched_space : Dict,
    data : dict[str, Any],
    mask : BArrayType,
):
    assert batched_space.backend.dtype_is_boolean(mask.dtype), f"Expected mask to be of boolean dtype, actually {mask.dtype}"
    assert len(mask.shape) == 1, f"Expected mask to be 1D, actually {mask.shape}"
    return {
        key: read_batched_data_with_mask(subspace, data[key], mask)
        for key, subspace in batched_space.spaces.items()
    }

@read_batched_data_with_mask.register(Tuple)
def _read_batched_data_with_mask_tuple(
    batched_space : Tuple,
    data : tuple[Any, ...],
    mask : BArrayType,
):
    assert batched_space.backend.dtype_is_boolean(mask.dtype), f"Expected mask to be of boolean dtype, actually {mask.dtype}"
    assert len(mask.shape) == 1, f"Expected mask to be 1D, actually {mask.shape}"
    if all(type(subspace) in read_batched_data_with_mask.registry for subspace in batched_space.spaces):
        return tuple(
            read_batched_data_with_mask(subspace, data[i], mask)
            for i, subspace in enumerate(batched_space.spaces)
        )
    else:
        ret = []
        for i, subspace in enumerate(batched_space.spaces):
            subspace_in_mask = bool(mask[i])
            if not subspace_in_mask:
                continue
            ret.append(data[i])
        return tuple(ret)

@singledispatch
def write_batched_data_with_mask(
    batched_space : Space,
    data : Any,
    mask : BArrayType,
    value : Any,
):
    raise TypeError(
        f"The space provided to `write_batched_data_with_mask` is not a Space instance, type: {type(batched_space)}, {batched_space}"
    )

@write_batched_data_with_mask.register(Box)
@write_batched_data_with_mask.register(MultiDiscrete)
@write_batched_data_with_mask.register(MultiBinary)
def _write_batched_data_with_mask_base(
    batched_space : Box | MultiDiscrete | MultiBinary,
    data : BArrayType,
    mask : BArrayType,
    value : BArrayType,
):
    assert batched_space.backend.dtype_is_boolean(mask.dtype), f"Expected mask to be of boolean dtype, actually {mask.dtype}"
    assert len(mask.shape) == 1, f"Expected mask to be 1D, actually {mask.shape}"
    data = batched_space.backend.replace_inplace(data, mask, value)
    return data

@write_batched_data_with_mask.register(Dict)
def _write_batched_data_with_mask_dict(
    batched_space : Dict,
    data : dict[str, Any],
    mask : BArrayType,
    value : dict[str, Any],
):
    assert batched_space.backend.dtype_is_boolean(mask.dtype), f"Expected mask to be of boolean dtype, actually {mask.dtype}"
    assert len(mask.shape) == 1, f"Expected mask to be 1D, actually {mask.shape}"
    return {
        key: write_batched_data_with_mask(subspace, data[key], mask, value[key])
        for key, subspace in batched_space.spaces.items()
    }

@write_batched_data_with_mask.register(Tuple)
def _write_batched_data_with_mask_tuple(
    batched_space : Tuple,
    data : tuple[Any, ...],
    mask : BArrayType,
    value : tuple[Any, ...],
):
    assert batched_space.backend.dtype_is_boolean(mask.dtype), f"Expected mask to be of boolean dtype, actually {mask.dtype}"
    assert len(mask.shape) == 1, f"Expected mask to be 1D, actually {mask.shape}"
    if all(type(subspace) in write_batched_data_with_mask.registry for subspace in batched_space.spaces):
        return tuple(
            write_batched_data_with_mask(subspace, data[i], mask, value[i])
            for i, subspace in enumerate(batched_space.spaces)
        )
    else:
        val_counter = 0
        ret = []
        for i in range(len(batched_space.spaces)):
            subspace_in_mask = bool(mask[i])
            if not subspace_in_mask:
                ret.append(value[val_counter])
                val_counter += 1
            else:
                ret.append(data[i])
        return tuple(ret)