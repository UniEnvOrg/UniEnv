"""Implementation of a space that represents the cartesian product of other spaces."""

from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple as TupleType, Type, Literal, List, Dict
import numpy as np
from .space import Space, register_space_to_gym_mapping
from unienv_interface.backends.base import ComputeBackend
import array_api_compat
import gymnasium as gym

_TupleBDeviceT = TypeVar("_BoxBDeviceT", covariant=True)
_TupleBDTypeT = TypeVar("_BoxBDTypeT", covariant=True)
_TupleBDRNGT = TypeVar("_BoxBDRNGT", covariant=True)

@register_space_to_gym_mapping(gym.spaces.Tuple)
class Tuple(Space[TupleType[Any, ...], TupleType[Any, ...], _TupleBDeviceT, _TupleBDTypeT, _TupleBDRNGT]):
    def __init__(
        self,
        backend : Type[ComputeBackend[Any, _TupleBDeviceT, _TupleBDTypeT, _TupleBDRNGT]],
        spaces: Iterable[Space[Any, Any, _TupleBDeviceT, _TupleBDTypeT, _TupleBDRNGT]],
        device : Optional[_TupleBDeviceT] = None,
        seed: Optional[int] = None,
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
        self.spaces : TupleType[Space[Any, Any, _TupleBDeviceT, _TupleBDTypeT, _TupleBDRNGT], ...] = tuple(new_spaces)
        super().__init__(
            backend=backend,
            shape=None,
            device=device,
            dtype=None,
            seed=seed
        )

    @property
    def is_flattenable(self):
        return all(space.is_flattenable for space in self.spaces)

    def seed(self, seed: Optional[int] = None) -> None:
        for space in self.spaces:
            space.seed(seed=seed)

    def sample(self) -> TupleType[Any, ...]:
        return tuple(space.sample() for space in self.spaces)

    def contains(self, x: TupleType[Any, ...]) -> bool:
        return (
            isinstance(x, TupleType)
            and len(x) == len(self.spaces)
            and all(space.contains(part) for (space, part) in zip(self.spaces, x))
        )

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return "Tuple(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def __getitem__(self, index: int) -> Space[Any]:
        """Get the subspace at specific `index`."""
        return self.spaces[index]

    def __len__(self) -> int:
        """Get the number of subspaces that are involved in the cartesian product."""
        return len(self.spaces)

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return isinstance(other, Tuple) and self.spaces == other.spaces
    
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
    
    def from_other_backend(self, other_data : TupleType[Any, ...]) -> TupleType[Any, ...]:
        return tuple(space.from_other_backend(part) for (space, part) in zip(self.spaces, other_data))
    
    def from_same_backend(self, other_data : TupleType[Any, ...]) -> TupleType[Any, ...]:
        return tuple(space.from_same_backend(part) for (space, part) in zip(self.spaces, other_data))

    def to_gym_space(self) -> gym.Space:
        return gym.spaces.Tuple([space.to_gym_space() for space in self.spaces])
    
    @staticmethod
    def from_gym_space(
        gym_space : gym.spaces.Tuple,
        backend : Type[ComputeBackend[Any, _TupleBDeviceT, _TupleBDTypeT, _TupleBDRNGT]],
        dtype : Optional[_TupleBDTypeT] = None,
        device : Optional[_TupleBDeviceT] = None,
    ) -> "Tuple[_TupleBDeviceT, _TupleBDTypeT, _TupleBDRNGT]":
        return Tuple(
            backend=backend,
            spaces=[Space.from_gym_space(subspace, backend=backend, dtype=dtype, device=device) for subspace in gym_space.spaces],
            device=device
        )