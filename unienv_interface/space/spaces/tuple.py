"""Implementation of a space that represents the cartesian product of other spaces."""

from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal, List, Dict
import numpy as np
from ..space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
import copy

class TupleSpace(Space[Tuple[Any, ...], BDeviceType, BDtypeType, BRNGType]):
    def __init__(
        self,
        backend : ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType],
        spaces: Iterable[Space[Any, BDeviceType, BDtypeType, BRNGType]],
        device : Optional[BDeviceType] = None,
    ):
        new_spaces = []
        for space in spaces:
            assert isinstance(
                space, Space
            ), f"{space} does not inherit from `gymnasium.Space`. Actual Type: {type(space)}"
            assert space.backend == backend, f"Backend mismatch: {space.backend} != {backend}"
            if device is not None:
                new_spaces.append(space.to(device=device))
            else:
                new_spaces.append(space)
        
        if device is None:
            device = Space.abbr_device(new_spaces)

        self.spaces : Tuple[Space[Any, BDeviceType, BDtypeType, BRNGType], ...] = tuple(new_spaces)
        super().__init__(
            backend=backend,
            shape=None,
            device=device,
            dtype=None,
        )

    def to(self, backend = None, device = None):
        if (backend is None or backend==self.backend) and (device is None or device==self.device):
            return self
        
        new_device = device if backend is not None else (device or self.device)
        return Tuple(
            backend=backend or self.backend,
            spaces=[space.to(backend, new_device) for space in self.spaces],
            device=new_device
        )

    def sample(self, rng : BRNGType) -> Tuple[BRNGType, Tuple[Any, ...]]:
        samples = []
        for space in self.spaces:
            rng, sample = space.sample(rng)
            samples.append(sample)
        return rng, tuple(samples)

    def create_empty(self) -> Tuple[Any, ...]:
        """Create an empty data structure for this space."""
        return tuple(space.create_empty() for space in self.spaces)

    def is_bounded(self, manner = "both"):
        return all(space.is_bounded(manner) for space in self.spaces)

    def contains(self, x: Tuple[Any, ...]) -> bool:
        return (
            isinstance(x, Tuple)
            and len(x) == len(self.spaces)
            and all(space.contains(part) for (space, part) in zip(self.spaces, x))
        )

    def get_repr(
        self,
        abbreviate = False, 
        include_backend = True, 
        include_device = True, 
        include_dtype = True
    ):
        next_include_device = include_device and self.device is None
        ret = f"TupleSpace({', '.join([space.get_repr(abbreviate, False, next_include_device, include_dtype) for space in self.spaces])}"
        if include_backend:
            ret += f", backend={self.backend}"
        if include_device and self.device is not None:
            ret += f", device={self.device}"
        ret += ")"
        return ret

    def __getitem__(self, index: int) -> Space[Any, BDeviceType, BDtypeType, BRNGType]:
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
    
    def data_to(self, data, backend = None, device = None):
        return tuple(
            space.data_to(part, backend=backend, device=device) 
            for (space, part) in zip(self.spaces, data)
        )