"""Implementation of a space that represents finite-length sequences."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence as SequenceType, TypeVar, Optional, Tuple, Type, Literal, List as ListType, Union
import numpy as np
from .space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
import array_api_compat
import gymnasium as gym

class Sequence(
    Space[Union[Any, Tuple[Any, ...]], Union[Any, Tuple[Any, ...]], BDeviceType, BDtypeType, BRNGType],
    Generic[BDeviceType, BDtypeType, BRNGType],
):
    def __init__(
        self,
        backend : ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType],
        space: Space[Any, Any, BDeviceType, BDtypeType, BRNGType],
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
        )

    @property
    def device(self) -> BDeviceType:
        return self.feature_space.device

    def to_device(self, device : BDeviceType) -> "Sequence[BDeviceType, BDtypeType, BRNGType]":
        return Sequence(
            backend=self.backend,
            space=self.feature_space.to_device(device)
        )

    def to_backend(self, backend : ComputeBackend, device : Optional[Any]) -> "Space":
        return Sequence(
            backend=backend,
            space=self.feature_space.to_backend(backend, device)
        )

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

    def sample(self, rng : BRNGType) -> Tuple[Any, ...]:
        length = int(self.backend.random_geometric(rng, (1,), p=0.25)[0])

        # Generate sample values from feature_space.
        sampled_values = []
        for _ in range(length):
            rng, sampled_value = self.feature_space.sample(rng)
            sampled_values.append(sampled_value)
        
        return rng, tuple(sampled_values)

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
    
    def from_other_backend(self, other_data : Tuple[Any, ...], backend : ComputeBackend) -> Tuple[Any, ...]:
        return tuple(self.feature_space.from_other_backend(part, backend) for part in other_data)
    
    def from_same_backend(self, other_data : Tuple[Any, ...]) -> Tuple[Any, ...]:
        return tuple(self.feature_space.from_same_backend(part) for part in other_data)

    def to_gym_space(self) -> gym.Space:
        return gym.spaces.Sequence(self.feature_space.to_gym_space(), stack=False)
