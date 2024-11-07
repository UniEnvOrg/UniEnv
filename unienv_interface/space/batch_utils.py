from __future__ import annotations

import typing
from copy import deepcopy
from functools import singledispatch
from typing import Optional, Any, Iterable, Iterator

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
    "batch_space",
    "batch_differing_spaces",
    "iterate",
    "concatenate"
]

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
    if len(space.spaces) == 0:
        raise ValueError("Tuple space must have at least one subspace to determine the batch size.")
    
    first_subspace = space.spaces[0]
    if isinstance(first_subspace, (Graph, Text, Sequence, Union)):
        return len(space.spaces)
    
    for subspace in space.spaces:
        try:
            return batch_size(subspace)
        except:
            continue

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
    # Without deepcopy, then the space.np_random is batched_space.spaces[0].np_random
    # Which is an issue if you are sampling actions of both the original space and the batched space
    batched_space = Tuple(
        backend=space.backend,
        spaces=[deepcopy(space) for _ in range(n)],
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
def _batch_spaces_undefined(spaces: typing.Sequence[Graph | Text | Sequence | Union], device : Optional[Any] = None):
    return Tuple(
        backend=spaces[0].backend,
        spaces=[deepcopy(space) for space in spaces],
        device=spaces[0].device, 
    )

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
    space: Box | Discrete | MultiDiscrete | MultiBinary,
    items: Iterable,
) -> Any:
    return space.backend.stack(items)


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
    return tuple(
        concatenate(subspace, [item[i] for item in items])
        for (i, subspace) in enumerate(space.spaces)
    )

@concatenate.register(Graph)
@concatenate.register(Text)
@concatenate.register(Sequence)
@concatenate.register(Space)
@concatenate.register(Union)
def _concatenate_custom(space: Space, items: Iterable, out: None) -> tuple[Any, ...]:
    return tuple(items)
