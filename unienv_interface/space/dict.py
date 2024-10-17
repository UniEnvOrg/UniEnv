"""Implementation of a space that represents the cartesian product of other spaces as a dictionary."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal, List, Dict as DictType
import numpy as np
from .space import Space
from unienv_interface.backends import ComputeBackend
import array_api_compat
import gymnasium as gym
from collections.abc import KeysView

_DictBDeviceT = TypeVar("_BoxBDeviceT", covariant=True)
_DictBDTypeT = TypeVar("_BoxBDTypeT", covariant=True)
_DictBDRNGT = TypeVar("_BoxBDRNGT", covariant=True)

class Dict(Space[DictType[str, Any], DictType[str, Any], _DictBDeviceT, _DictBDTypeT, _DictBDRNGT]):
    def __init__(
        self,
        backend: Type[ComputeBackend[Any, _DictBDeviceT, _DictBDTypeT, _DictBDRNGT]],
        spaces: None | DictType[str, Space[Any, Any, _DictBDeviceT, _DictBDTypeT, _DictBDRNGT]] | Sequence[tuple[str, Space[Any, Any, _DictBDeviceT, _DictBDTypeT, _DictBDRNGT]]] = None,
        device : Optional[_DictBDeviceT] = None,
        seed: Optional[int] = None,
    ):
        """Constructor of :class:`Dict` space.

        This space can be instantiated in one of two ways: Either you pass a dictionary
        of spaces to :meth:`__init__` via the ``spaces`` argument, or you pass the spaces as separate
        keyword arguments (where you will need to avoid the keys ``spaces`` and ``seed``)

        Args:
            spaces: A dictionary of spaces. This specifies the structure of the :class:`Dict` space
            seed: Optionally, you can use this argument to seed the RNGs of the spaces that make up the :class:`Dict` space.
            **spaces_kwargs: If ``spaces`` is ``None``, you need to pass the constituent spaces as keyword arguments, as described above.
        """
        # Convert the spaces into an OrderedDict
        if isinstance(spaces, Mapping):
            # for legacy reasons, we need to preserve the sorted dictionary items.
            # as this could matter for projects flatten the dictionary.
            try:
                spaces = dict(sorted(spaces.items()))
            except TypeError:
                # Incomparable types (e.g. `int` vs. `str`, or user-defined types) found.
                # The keys remain in the insertion order.
                spaces = dict(spaces.items())
        elif isinstance(spaces, Sequence):
            spaces = dict(spaces)
        elif spaces is None:
            spaces = dict()
        else:
            raise TypeError(
                f"Unexpected Dict space input, expecting dict, OrderedDict or Sequence, actual type: {type(spaces)}"
            )

        new_spaces: DictType[str, Space[Any, Any, _DictBDeviceT, _DictBDTypeT, _DictBDRNGT]] = {}
        for key, space in spaces.items():
            assert isinstance(
                space, Space
            ), f"Dict space element is not an instance of Space: key='{key}', space={space}"
            assert space.backend == backend, f"Backend mismatch: {space.backend} != {backend}"
            if device is not None:
                new_spaces[key] = space.to_device(device)
            else:
                new_spaces[key] = space
        self.spaces = new_spaces

        # None for shape and dtype, since it'll require special handling
        super().__init__(
            backend=backend,
            shape=None,
            device=device,
            dtype=None,
            seed=seed,
        )

    @property
    def is_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return all(space.is_flattenable for space in self.spaces.values())

    @property
    def flat_dim(self) -> Optional[int]:
        """Return the shape of the space as an immutable property."""
        if not self.is_flattenable:
            return None
        return sum(space.flat_dim for space in self.spaces.values())
    
    def flatten(self, data : DictType[str, Any]) -> Any:
        """Flatten the data."""
        return self.backend.array_api_namespace.concat([
            space.flatten(data[key]) for key, space in self.spaces.items()
        ])
    
    def unflatten(self, data : Any) -> DictType[str, Any]:
        """Unflatten the data."""
        result = {}
        start = 0
        for key, space in self.spaces.items():
            end = start + space.flat_dim
            result[key] = space.unflatten(data[start:end])
            start = end
        return result

    def to_device(self, device : _DictBDeviceT) -> "Dict[_DictBDeviceT, _DictBDTypeT, _DictBDRNGT]":
        return Dict(
            backend=self.backend,
            spaces={key: space.to_device(device) for key, space in self.spaces.items()},
            device=device
        )

    def to_backend(self, backend : Type[ComputeBackend], device : Optional[Any]) -> "Dict":
        return Dict(
            backend=backend,
            spaces={key: space.to_backend(backend, device) for key, space in self.spaces.items()},
            device=device
        )

    def seed(self, seed: int | None = None) -> None:
        super().seed(seed)
        for key, subspace in self.spaces.items():
            subspace.seed(seed)

    def sample(self) -> DictType[str, Any]:
        return {k: space.sample() for k, space in self.spaces.items()}

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, dict) and x.keys() == self.spaces.keys():
            return all(x[key] in self.spaces[key] for key in self.spaces.keys())
        return False

    def __getitem__(self, key: str) -> Space[Any, Any, _DictBDeviceT, _DictBDTypeT, _DictBDRNGT]:
        """Get the space that is associated to `key`."""
        return self.spaces[key]

    def keys(self) -> KeysView:
        """Returns the keys of the Dict."""
        return KeysView(self.spaces)

    def __setitem__(self, key: str, value: Space[Any, Any, _DictBDeviceT, _DictBDTypeT, _DictBDRNGT]) -> None:
        """Set the space that is associated to `key`."""
        assert isinstance(
            value, Space
        ), f"Trying to set {key} to Dict space with value that is not a Space, actual type: {type(value)}"
        self.spaces[key] = value.to_device(self.device) if self.device is not None else value

    def __iter__(self):
        """Iterator through the keys of the subspaces."""
        yield from self.spaces

    def __len__(self) -> int:
        """Gives the number of simpler spaces that make up the `Dict` space."""
        return len(self.spaces)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return (
            "Dict(" + ", ".join([f"{k!r}: {s}" for k, s in self.spaces.items()]) + ")"
        )

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return (
            isinstance(other, Dict)
            # Comparison of `OrderedDict`s is order-sensitive
            and self.spaces == other.spaces  # OrderedDict.__eq__
        )

    def to_jsonable(self, sample_n: Sequence[DictType[str, Any]]) -> DictType[str, List[Any]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        # serialize as dict-repr of vectors
        return {
            key: space.to_jsonable([sample[key] for sample in sample_n])
            for key, space in self.spaces.items()
        }

    def from_jsonable(self, sample_n: DictType[str, List[Any]]) -> List[DictType[str, Any]]:
        """Convert a JSONable data type to a batch of samples from this space."""
        dict_of_list: DictType[str, list[Any]] = {
            key: space.from_jsonable(sample_n[key])
            for key, space in self.spaces.items()
        }

        n_elements = len(next(iter(dict_of_list.values())))
        result = [
            {key: value[n] for key, value in dict_of_list.items()}
            for n in range(n_elements)
        ]
        return result
    
    def from_gym_data(self, gym_data : DictType[str, Any]) -> DictType[str, Any]:
        return {
            key: space.from_gym_data(gym_data[key]) for key, space in self.spaces.items()
        }
    
    def to_gym_data(self, data : DictType[str, Any]) -> DictType[str, Any]:
        return {
            key: space.to_gym_data(data[key]) for key, space in self.spaces.items()
        }
    
    def from_other_backend(self, other_data : DictType[str, Any]) -> DictType[str, Any]:
        return {
            key: space.from_other_backend(other_data[key]) for key, space in self.spaces.items()
        }
    
    def from_same_backend(self, other_data : DictType[str, Any]) -> DictType[str, Any]:
        return {
            key: space.from_same_backend(other_data[key]) for key, space in self.spaces.items()
        }

    def to_gym_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {key: space.to_gym_space() for key, space in self.spaces.items()},
            seed=self.seed,
        )
