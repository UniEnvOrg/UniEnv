"""Implementation of a space that represents the cartesian product of other spaces."""

from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple as TupleType, Type, Literal, List, Dict
import numpy as np
from .space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
import gymnasium as gym
import copy

class Tuple(Space[TupleType[Any, ...], TupleType[Any, ...], BDeviceType, BDtypeType, BRNGType]):
    def __init__(
        self,
        backend : ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType],
        spaces: Iterable[Space[Any, Any, BDeviceType, BDtypeType, BRNGType]],
        device : Optional[BDeviceType] = None,
    ):
        new_spaces = []
        for space in spaces:
            assert isinstance(
                space, Space
            ), f"{space} does not inherit from `gymnasium.Space`. Actual Type: {type(space)}"
            assert space.backend == backend, f"Backend mismatch: {space.backend} != {backend}"
            if device is not None:
                new_spaces.append(space.to_device(device))
            else:
                new_spaces.append(space)
        
        if device is None:
            device = Space.abbr_device(new_spaces)

        self.spaces : TupleType[Space[Any, Any, BDeviceType, BDtypeType, BRNGType], ...] = tuple(new_spaces)
        super().__init__(
            backend=backend,
            shape=None,
            device=device,
            dtype=None,
        )

    @property
    def is_flattenable(self):
        return all(space.is_flattenable for space in self.spaces)

    @property
    def flat_dim(self) -> Optional[int]:
        """Return the shape of the space as an immutable property."""
        if not self.is_flattenable:
            return None
        return sum(space.flat_dim for space in self.spaces)
    
    def flatten(self, data : TupleType[Any, ...]) -> Any:
        """Flatten the data."""
        return self.backend.array_api_namespace.concat([
            space.flatten(data[i]) for i, space in enumerate(self.spaces)
        ])
    
    def unflatten(self, data : Any) -> TupleType[Any, ...]:
        """Unflatten the data."""
        result = []
        start = 0
        for space in self.spaces:
            end = start + space.flat_dim
            result.append(space.unflatten(data[start:end]))
            start = end
        return result

    def to_device(self, device : BDeviceType) -> "Tuple[BDeviceType, BDtypeType, BRNGType]":
        return Tuple(
            backend=self.backend,
            spaces=[space.to_device(device) for space in self.spaces],
            device=device
        )

    def to_backend(self, backend : ComputeBackend, device : Optional[Any]) -> "Tuple":
        return Tuple(
            backend=backend,
            spaces=[space.to_backend(backend, device) for space in self.spaces],
            device=device
        )
    
    def sample(self, rng : BRNGType) -> TupleType[BRNGType, TupleType[Any, ...]]:
        samples = []
        for space in self.spaces:
            rng, sample = space.sample(rng)
            samples.append(sample)
        return rng, tuple(samples)

    def contains(self, x: TupleType[Any, ...]) -> bool:
        return (
            isinstance(x, TupleType)
            and len(x) == len(self.spaces)
            and all(space.contains(part) for (space, part) in zip(self.spaces, x))
        )

    def get_repr(
        self, 
        include_backend = True, 
        include_device = True, 
        include_dtype = True
    ):
        next_include_device = include_device and self.device is None
        ret = f"Tuple({', '.join([space.get_repr(False, next_include_device, include_dtype) for space in self.spaces])}"
        if include_backend:
            ret += f", backend={self.backend}"
        if include_device and self.device is not None:
            ret += f", device={self.device}"
        ret += ")"
        return ret

    def __getitem__(self, index: int) -> Space[Any, Any, BDeviceType, BDtypeType, BRNGType]:
        """Get the subspace at specific `index`."""
        return self.spaces[index]

    def __len__(self) -> int:
        """Get the number of subspaces that are involved in the cartesian product."""
        return len(self.spaces)

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return isinstance(other, Tuple) and self.spaces == other.spaces
    
    def __copy__(self) -> "Tuple[BDeviceType, BDtypeType, BRNGType]":
        """Create a shallow copy of the Dict space."""
        return Tuple(
            backend=self.backend,
            spaces=copy.copy(self.spaces),
            device=self.device
        )

    def to_jsonable(
        self, sample_n: Sequence[TupleType[Any, ...]]
    ) -> List[Any]:
        """Convert a batch of samples from this space to a JSONable data type."""
        # serialize as list-repr of tuple of vectors
        return [
            space.to_jsonable([sample[i] for sample in sample_n])
            for i, space in enumerate(self.spaces)
        ]

    def from_jsonable(self, sample_n: List[Any]) -> List[TupleType[Any, ...]]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [
            sample
            for sample in zip(
                *[
                    space.from_jsonable(sample_n[i])
                    for i, space in enumerate(self.spaces)
                ]
            )
        ]

    def from_gym_data(self, gym_data : TupleType[Any, ...]) -> TupleType[Any, ...]:
        return tuple(space.from_gym_data(part) for (space, part) in zip(self.spaces, gym_data))
    
    def to_gym_data(self, data : TupleType[Any, ...]) -> TupleType[Any, ...]:
        return tuple(space.to_gym_data(part) for (space, part) in zip(self.spaces, data))
    
    def from_other_backend(self, other_data : TupleType[Any, ...], backend : ComputeBackend) -> TupleType[Any, ...]:
        return tuple(space.from_other_backend(part, backend) for (space, part) in zip(self.spaces, other_data))
    
    def from_same_backend(self, other_data : TupleType[Any, ...]) -> TupleType[Any, ...]:
        return tuple(space.from_same_backend(part) for (space, part) in zip(self.spaces, other_data))

    def to_gym_space(self) -> gym.Space:
        return gym.spaces.Tuple([space.to_gym_space() for space in self.spaces])
