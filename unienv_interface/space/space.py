from typing import Any, Generic, Iterable, Union, Mapping, Sequence, TypeVar, Optional, Tuple, Type, List
import numpy as np
from unienv_interface.backends import ComputeBackend
import gymnasium as gym
import abc

SpaceDataT = TypeVar("SpaceDataT", covariant=True)
_SpaceBDeviceT = TypeVar("_SpaceBDeviceT", covariant=True)
_SpaceBDTypeT = TypeVar("_SpaceBDTypeT", covariant=True)
_SpaceBDRNGT = TypeVar("_SpaceBDRNGT", covariant=True)
_GymDataT = TypeVar("_GymDataT", covariant=True)
class Space(abc.ABC, Generic[SpaceDataT, _GymDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]):
    def __init__(
        self,
        backend : ComputeBackend[Any, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT],
        shape: Optional[Sequence[int]] = None,
        device : Optional[_SpaceBDeviceT] = None,
        dtype: Optional[_SpaceBDTypeT] = None,
    ):
        self.backend = backend
        self._shape = None if shape is None else tuple(shape)
        self.dtype = dtype
        self._device = device

    @property
    def device(self) -> Optional[_SpaceBDeviceT]:
        return self._device
    
    @abc.abstractmethod
    def to_device(self, device : _SpaceBDeviceT) -> "Space[SpaceDataT, _GymDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]":
        raise NotImplementedError

    @abc.abstractmethod
    def to_backend(self, backend : ComputeBackend, device : Optional[Any]) -> "Space":
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Return the shape of the space as an immutable property."""
        return self._shape

    @property
    def is_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`Box`."""
        return False
    
    @property
    def is_batch_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`Box`."""
        return False

    @property
    def flat_dim(self) -> int | None:
        """Return the shape of the space as an immutable property."""
        return None
    
    @property
    def batch_flat_dim(self) -> int | None:
        """Return the shape of the space as an immutable property."""
        return None

    def flatten(self, data : SpaceDataT) -> Any:
        """Flatten the data."""
        raise NotImplementedError
    
    def flatten_batch(self, data : SpaceDataT) -> Any:
        """Flatten the data."""
        raise NotImplementedError

    def unflatten(self, data : Any) -> SpaceDataT:
        """Unflatten the data."""
        raise NotImplementedError
    
    def unflatten_batch(self, data : Any) -> SpaceDataT:
        """Unflatten the data."""
        raise NotImplementedError

    def sample(self, rng : _SpaceBDRNGT, **kwargs) -> Tuple[_SpaceBDRNGT, SpaceDataT]:
        raise NotImplementedError

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        raise NotImplementedError

    def __contains__(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return self.contains(x)
    
    def __repr__(self) -> str:
        return self.get_repr(
            include_backend=True,
            include_device=True,
            include_dtype=True
        )

    def get_repr(
        self,
        include_backend : bool = True,
        include_device : bool = True,
        include_dtype : bool = True,
    ) -> str:
        """Return a string representation of the space."""
        raise NotImplementedError

    def to_jsonable(self, sample_n: Sequence[SpaceDataT]) -> Any:
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return list(sample_n)

    def from_jsonable(self, sample_n: List[Any]) -> List[SpaceDataT]:
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n
    
    def from_gym_data(self, gym_data : _GymDataT) -> SpaceDataT:
        """Convert a gym space to this space."""
        return gym_data
    
    def to_gym_data(self, data : SpaceDataT) -> _GymDataT:
        """Convert this space to a gym space."""
        return data
    
    @abc.abstractmethod
    def from_other_backend(self, other_data : Any, backend : ComputeBackend) -> SpaceDataT:
        """Convert data from another backend to this backend."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def from_same_backend(self, other_data : SpaceDataT) -> SpaceDataT:
        """Convert data from another device to this device."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_gym_space(self) -> gym.Space:
        """Convert this space to a gym space."""
        raise NotImplementedError

    @staticmethod
    def abbr_device(spaces : "Iterable[Space[Any, Any, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]]") -> Optional[_SpaceBDeviceT]:
        """Get the abbreviated device of the spaces."""
        
        iter_spaces = iter(spaces)
        try:
            first_space = next(iter_spaces)
        except StopIteration:
            return None
        device = first_space.device
        for space in spaces:
            if space.device != device:
                return None
        return device