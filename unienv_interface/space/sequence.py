"""Implementation of a space that represents finite-length sequences."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence as SequenceType, TypeVar, Optional, Tuple, Type, Literal, List as ListType, Union
import numpy as np
from .space import Space
from unienv_interface.backends import ComputeBackend
import array_api_compat
import gymnasium as gym
from .discrete import Discrete

_SequenceBDeviceT = TypeVar("_MultiBDeviceT", covariant=True)
_SequenceBDTypeT = TypeVar("_MultiBDTypeT", covariant=True)
_SequenceBDRNGT = TypeVar("_MultiBDRNGT", covariant=True)

class Sequence(Space[Union[Any, Tuple[Any, ...]], Union[Any, Tuple[Any, ...]], _SequenceBDeviceT, _SequenceBDTypeT, _SequenceBDRNGT]):
    def __init__(
        self,
        backend : Type[ComputeBackend[Any, _SequenceBDeviceT, _SequenceBDTypeT, _SequenceBDRNGT]],
        space: Space[Any, Any, _SequenceBDeviceT, _SequenceBDTypeT, _SequenceBDRNGT],
        seed: Optional[int] = None,
    ):
        assert isinstance(
            space, Space
        ), f"Expects the feature space to be instance of a gym Space, actual type: {type(space)}"
        assert space.backend == backend, f"Expects the backend of the feature space to be the same as the Sequence backend, actual backend: {space.backend}, expected backend: {backend}"
        
        self.feature_space = space
        super().__init__(
            backend=backend,
            shape=None,
            device=None,
            dtype=None,
            seed=seed,
        )

    @property
    def device(self) -> _SequenceBDeviceT:
        return self.feature_space.device

    def to_device(self, device : _SequenceBDeviceT) -> "Sequence[_SequenceBDeviceT, _SequenceBDTypeT, _SequenceBDRNGT]":
        return Sequence(
            backend=self.backend,
            space=self.feature_space.to_device(device)
        )

    def to_backend(self, backend : Type[ComputeBackend], device : Optional[Any]) -> "Space":
        return Sequence(
            backend=backend,
            space=self.feature_space.to_backend(backend, device)
        )

    def seed(self, seed: Optional[int]) -> None:
        super().seed(seed)
        self.feature_space.seed(seed)

    @property
    def is_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return False

    @property
    def flat_dim(self) -> None:
        """Return the shape of the space as an immutable property."""
        return None
    
    def flatten(self, data : ListType[Any]) -> Any:
        """Flatten the data."""
        raise NotImplementedError("Sequence space is not flattenable")
    
    def unflatten(self, data : Any) -> ListType[Any]:
        """Unflatten the data."""
        raise NotImplementedError("Sequence space is not flattenable")

    def sample(self) -> tuple[Any] | Any:
        length = self.np_rng.geometric(0.25)

        # Generate sample values from feature_space.
        sampled_values = tuple(
            self.feature_space.sample() for _ in range(length)
        )

        return sampled_values

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        # by definition, any sequence is an iterable
        return isinstance(x, tuple) and all(
            self.feature_space.contains(item) for item in x
        )

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"Sequence({self.feature_space})"

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Sequence)
            and self.feature_space == other.feature_space
        )
    
    def to_jsonable(
        self, sample_n: SequenceType[Tuple[Any, ...] | Any]
    ) -> ListType[Any]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [self.feature_space.to_jsonable(sample) for sample in sample_n]

    def from_jsonable(self, sample_n: ListType[Any]) -> ListType[Any]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [
            tuple(self.feature_space.from_jsonable(sample)) for sample in sample_n
        ]
    
    def from_gym_data(self, gym_data : Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Convert a gym space to this space."""
        return tuple(self.feature_space.from_gym_data(part) for part in gym_data)
    
    def to_gym_data(self, data : Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Convert this space to a gym space."""
        return tuple(self.feature_space.to_gym_data(part) for part in data)
    
    def from_other_backend(self, other_data : Tuple[Any, ...]) -> Tuple[Any, ...]:
        return tuple(self.feature_space.from_other_backend(part) for part in other_data)
    
    def from_same_backend(self, other_data : Tuple[Any, ...]) -> Tuple[Any, ...]:
        return tuple(self.feature_space.from_same_backend(part) for part in other_data)

    def to_gym_space(self) -> gym.Space:
        return gym.spaces.Sequence(self.feature_space.to_gym_space(), seed=self.np_rng.integers(0, 4096), stack=False)
