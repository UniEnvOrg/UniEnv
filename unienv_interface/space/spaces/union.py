"""Implementation of a space that represents the cartesian product of other spaces."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal, List
import numpy as np
from ..space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType


class UnionSpace(Space[Tuple[int, Any], BDeviceType, BDtypeType, BRNGType]):
    def __init__(
        self,
        backend: ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType],
        spaces: Iterable[Space[Any, BDeviceType, BDtypeType, BRNGType]],
        device: Optional[BDeviceType] = None,
    ):
        assert isinstance(spaces, Iterable), f"{spaces} is not an iterable"
        self.spaces = tuple(spaces if device is None else [space.to(device=device) for space in spaces])
        assert len(self.spaces) > 0, "Cannot have an empty Union space"
        for space in self.spaces:
            assert isinstance(
                space, Space
            ), f"{space} does not inherit from `Space`. Actual Type: {type(space)}"
            assert space.backend == backend, f"Backend mismatch: {space.backend} != {backend}"

        if device is None:
            device = Space.abbr_device(self.spaces)

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
        return UnionSpace(
            backend=backend or self.backend,
            spaces=[space.to(backend, new_device) for space in self.spaces],
            device=new_device
        )
    
    def sample(self, rng : BRNGType) -> Tuple[
        BRNGType,
        Tuple[int, Any]
    ]:
        rng, subspace_idx = self.backend.random.random_discrete_uniform(
            1,
            0,
            len(self.spaces),
            rng=rng,
        )
        subspace_idx = int(subspace_idx[0])

        subspace = self.spaces[subspace_idx]
        rng, sample = subspace.sample(rng)
        return rng, (subspace_idx, sample)

    def create_empty(self) -> Tuple[int, Any]:
        """Create an empty data structure for this space."""
        return (0, self.spaces[0].create_empty())

    def is_bounded(self, manner = "both"):
        return all(
            space.is_bounded(manner) for space in self.spaces
        )

    def contains(self, x: Tuple[int, Any]) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return (
            isinstance(x, Tuple)
            and len(x) == 2
            and isinstance(x[0], (np.int64, int))
            and 0 <= x[0] < len(self.spaces)
            and self.spaces[x[0]].contains(x[1])
        )

    def get_repr(
        self, 
        abbreviate : bool = False,
        include_backend = True, 
        include_device = True, 
        include_dtype = True
    ):
        next_include_device = include_device and self.device is None
        ret = f"UnionSpace({', '.join([space.get_repr(abbreviate, False, next_include_device, include_dtype) for space in self.spaces])}"
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
        return isinstance(other, UnionSpace) and self.spaces == other.spaces

    def data_to(self, data, backend = None, device = None):
        if (backend is None or backend==self.backend) and (device is None or device==self.device):
            return data
        space = self.spaces[data[0]]
        return (data[0], space.data_to(data[1], backend=backend, device=device))

    

