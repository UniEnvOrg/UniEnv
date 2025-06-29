"""Implementation of a space that represents the cartesian product of other spaces as a dictionary."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal, List, Dict, Union
import numpy as np
from ..space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from collections.abc import KeysView

class DictSpace(Space[Dict[str, Any], BDeviceType, BDtypeType, BRNGType]):
    def __init__(
        self,
        backend: ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType],
        spaces: Optional[Union[
            Dict[str, Space[Any, BDeviceType, BDtypeType, BRNGType]], 
            Sequence[Tuple[str, Space[Any, BDeviceType, BDtypeType, BRNGType]]]
        ]] = None,
        device : Optional[BDeviceType] = None,
    ):
        if isinstance(spaces, Mapping):
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
        
        new_spaces: Dict[str, Space[Any, BDeviceType, BDtypeType, BRNGType]] = {}

        for key, space in spaces.items():
            assert isinstance(
                space, Space
            ), f"Dict space element is not an instance of Space: key='{key}', space={space}"
            assert space.backend == backend, f"Backend mismatch: {space.backend} != {backend}"
            if device is not None:
                new_spaces[key] = space.to(device=device)
            else:
                new_spaces[key] = space
        self.spaces = new_spaces
        
        if device is None:
            device = Space.abbr_device(new_spaces.values())

        # None for shape and dtype, since it'll require special handling
        super().__init__(
            backend=backend,
            shape=None,
            device=device,
            dtype=None,
        )

    def to(
        self, 
        backend : Optional[ComputeBackend] = None, 
        device : Optional[Union[BDeviceType, Any]] = None
    ) -> Union["DictSpace[BDeviceType, BDtypeType, BRNGType]", "DictSpace"]:
        if (backend is None or backend==self.backend) and (device is None or device==self.device):
            return self
        return DictSpace(
            backend=backend or self.backend,
            spaces={
                key: space.to(backend=backend, device=device) 
                for key, space in self.spaces.items()
            },
            device=device if backend is not None else (device or self.device)
        )

    def sample(self, rng : BDeviceType) -> Tuple[BDeviceType, Dict[str, Any]]:
        ret_dict = {}
        for key, space in self.spaces.items():
            rng, ret_dict[key] = space.sample(rng)
        return rng, ret_dict
    
    def create_empty(self) -> Dict[str, Any]:
        """Create an empty data structure for this space."""
        return {
            key: space.create_empty() 
            for key, space in self.spaces.items()
        }

    def is_bounded(self, manner = "both"):
        return all(
            space.is_bounded(manner) for space in self.spaces.values()
        )

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, Mapping) and x.keys() == self.spaces.keys():
            return all(x[key] in self.spaces[key] for key in self.spaces.keys())
        return False

    def get_repr(
        self,
        abbreviate : bool = False,
        include_backend : bool = True,
        include_device : bool = True,
        include_dtype : bool = True,
    ) -> str:
        next_include_device = include_device and self.device is None
        if abbreviate:
            ret = "{" + ", ".join([
                f"{k!r}: {s.get_repr(True, False, next_include_device, include_dtype)}" 
                for k, s in self.spaces.items()
            ])
        else:
            ret = "DictSpace(" + ", ".join([
                f"{k!r}: {s.get_repr(False, False, next_include_device, include_dtype)}" 
                for k, s in self.spaces.items()
            ])
        if include_backend:
            ret += f", backend={self.backend}"
        if include_device and self.device is not None:
            ret += f", device={self.device}"
        if abbreviate:
            ret += "}"
        else:
            ret += ")"
        
        return ret

    def __getitem__(self, key: str) -> Space[Any, BDeviceType, BDtypeType, BRNGType]:
        """Get the space that is associated to `key`."""
        return self.spaces[key]

    def keys(self) -> KeysView:
        """Returns the keys of the Dict."""
        return KeysView(self.spaces)

    def __setitem__(self, key: str, value: Space[Any, BDeviceType, BDtypeType, BRNGType]) -> None:
        """Set the space that is associated to `key`."""
        assert isinstance(
            value, Space
        ), f"Trying to set {key} to Dict space with value that is not a Space, actual type: {type(value)}"
        assert value.backend == self.backend, (
            f"Trying to set {key} to Dict space with value that has a different backend, "
            f"expected {self.backend}, actual {value.backend}"
        )
        new_space = value.to(device=self.device) if self.device is not None else value
        self.spaces[key] = new_space

    def __delitem__(self, key: str) -> None:
        """Delete the space that is associated to `key`."""
        del self.spaces[key]

    def __iter__(self):
        """Iterator through the keys of the subspaces."""
        yield from self.spaces

    def __len__(self) -> int:
        """Gives the number of simpler spaces that make up the `Dict` space."""
        return len(self.spaces)

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return (
            isinstance(other, DictSpace)
            # Comparison of `OrderedDict`s is order-sensitive
            and self.spaces == other.spaces  # OrderedDict.__eq__
        )   

    def __copy__(self) -> "DictSpace[BDeviceType, BDtypeType, BRNGType]":
        """Create a shallow copy of the Dict space."""
        return DictSpace(
            backend=self.backend,
            spaces=self.spaces.copy(),
            device=self.device
        )

    def data_to(self, data, backend = None, device = None):
        return {
            key: space.data_to(data[key], backend=backend, device=device) 
            for key, space in self.spaces.items()
        }

