"""Implementation of a space that represents the cartesian product of other spaces."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal, List
import numpy as np
from .space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
import array_api_compat
import gymnasium as gym

class Union(Space[Tuple[int, Any], Tuple[int, Any], BDeviceType, BDtypeType, BRNGType]):
    def __init__(
        self,
        backend: Type[ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]],
        spaces: Iterable[Space[Any, Any, BDeviceType, BDtypeType, BRNGType]],
        device: Optional[BDeviceType] = None,
    ):
        assert isinstance(spaces, Iterable), f"{spaces} is not an iterable"
        self.spaces = tuple(spaces if device is None else [space.to_device(device) for space in spaces])
        assert len(self.spaces) > 0, "Cannot have an empty Union space"
        for space in self.spaces:
            assert isinstance(
                space, Space
            ), f"{space} does not inherit from `Space`. Actual Type: {type(space)}"
            assert space.backend == backend, f"Backend mismatch: {space.backend} != {backend}"

        super().__init__(
            backend=backend,
            shape=None,
            device=device,
            dtype=None,
        )

    @property
    def is_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return all(space.is_flattenable for space in self.spaces)
    
    @property
    def flat_dim(self) -> int | None:
        return 1 + max(space.flat_dim for space in self.spaces)
    
    def flatten(self, data : Tuple[int, Any]) -> Any:
        space_idx, space_data = data
        flat_sample = self.spaces[space_idx].flatten(space_data)
        padding_size = self.flat_dim - len(flat_sample)
        if padding_size > 0:
            padding = self.backend.array_api_namespace.zeros(padding_size, dtype=self.dtype)
            flat_sample = self.backend.array_api_namespace.concat((flat_sample, padding))
        index_array = self.backend.array_api_namespace.full(1, space_idx)
        return self.backend.array_api_namespace.concat((index_array, flat_sample))
    
    def unflatten(self, data : Any) -> Tuple[int, Any]:
        """Unflatten the data."""
        space_idx = data[0]
        subspace = self.spaces[space_idx]
        subspace_data = data[1:subspace.flat_dim+1]
        return (space_idx, subspace.unflatten(subspace_data))

    def to_device(self, device : BDeviceType) -> "Union[BDeviceType, BDtypeType, BRNGType]":
        return Union(
            backend=self.backend,
            spaces=self.spaces, #[space.to_device(device) for space in self.spaces],
            device=device,
        )

    def to_backend(self, backend : Type[ComputeBackend], device : Optional[Any]) -> "Union":
        return Union(
            backend=backend,
            spaces=[space.to_backend(backend, device) for space in self.spaces],
            device=device,
        )

    def sample(self, rng : BRNGType) -> Tuple[
        BRNGType,
        Tuple[int, Any]
    ]:
        rng, subspace_idx = self.backend.random_discrete_uniform(
            rng,
            shape=(1,),
            low=0,
            to_num=len(self.spaces)
        )
        subspace_idx = int(subspace_idx[0])

        subspace = self.spaces[subspace_idx]
        rng, sample = subspace.sample(rng)
        return rng, (subspace_idx, sample)

    def contains(self, x: Tuple[int, Any]) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return (
            isinstance(x, Tuple)
            and len(x) == 2
            and isinstance(x[0], (np.int64, int))
            and 0 <= x[0] < len(self.spaces)
            and self.spaces[x[0]].contains(x[1])
        )

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return "Union(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def __getitem__(self, index: int) -> Space[Any, Any, BDeviceType, BDtypeType, BRNGType]:
        """Get the subspace at specific `index`."""
        return self.spaces[index]

    def __len__(self) -> int:
        """Get the number of subspaces that are involved in the cartesian product."""
        return len(self.spaces)

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return isinstance(other, Union) and self.spaces == other.spaces

    def to_jsonable(
        self, sample_n: Sequence[tuple[int, Any]]
    ) -> List[Tuple[int, Any]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [
            (space_idx, self.spaces[space_idx].to_jsonable([sample])) for space_idx, sample in sample_n
        ]

    def from_jsonable(self, sample_n: List[Tuple[int,Any]]) -> List[Tuple[int, Any]]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [
            (
                np.int64(space_idx),
                self.spaces[space_idx].from_jsonable(jsonable_sample)[0],
            )
            for space_idx, jsonable_sample in sample_n
        ]

    def from_gym_data(self, gym_data : Tuple[int, Any]) -> Tuple[int, Any]:
        """Convert a gym space to this space."""
        return (gym_data[0], self.spaces[gym_data[0]].from_gym_data(gym_data[1]))
    
    def to_gym_data(self, data : Tuple[int, Any]) -> Tuple[int, Any]:
        """Convert this space to a gym space."""
        return (data[0], self.spaces[data[0]].to_gym_data(data[1]))
    
    def from_other_backend(self, other_data : Tuple[int, Any], backend : Type[ComputeBackend]) -> Tuple[int, Any]:
        """Convert data from another backend to this backend."""
        return (other_data[0], self.spaces[other_data[0]].from_other_backend(other_data[1], backend))
    
    def from_same_backend(self, other_data : Tuple[int, Any]) -> Tuple[int, Any]:
        """Convert data from another device to this device."""
        return (other_data[0], self.spaces[other_data[0]].from_same_backend(other_data[1]))

    def to_gym_space(self) -> "gym.spaces.OneOf":
        if "OneOf" in gym.spaces.__all__:
            return gym.spaces.OneOf([space.to_gym_space() for space in self.spaces])
        else:
            raise NotImplementedError("Conversion to gym space requires gymnasium >= 1.0.0a")
    

