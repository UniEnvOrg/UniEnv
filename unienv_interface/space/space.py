from typing import Any, Generic, Iterable, Union, Mapping, Sequence, TypeVar, Optional, Tuple, Type, List
import numpy as np
from unienv_interface.backends.base import ComputeBackend
import gymnasium as gym
import abc

_space_to_gym_mappings : Mapping[Type["Space"], Type[gym.Space]] = {}
_gym_to_space_mappings : Mapping[Type[gym.Space], Type["Space"]] = {}
def register_space_to_gym_mapping(gym_cls : Type[gym.Space]):
    def decorator(space_cls : Type["Space"]):
        _space_to_gym_mappings[space_cls] = gym_cls
        _gym_to_space_mappings[gym_cls] = space_cls
        return space_cls
    return decorator

SpaceDataT = TypeVar("SpaceDataT", covariant=True)
_SpaceBDeviceT = TypeVar("_SpaceBDeviceT", covariant=True)
_SpaceBDTypeT = TypeVar("_SpaceBDTypeT", covariant=True)
_SpaceBDRNGT = TypeVar("_SpaceBDRNGT", covariant=True)
_GymDataT = TypeVar("_GymDataT", covariant=True)
class Space(abc.ABC, Generic[SpaceDataT, _GymDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]):
    def __init__(
        self,
        backend : Type[ComputeBackend[Any, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]],
        shape: Optional[Sequence[int]] = None,
        device : Optional[_SpaceBDeviceT] = None,
        dtype: Optional[_SpaceBDTypeT] = None,
        seed: Optional[int] = None,
    ):
        self.backend = backend
        self._shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self._rng : Optional[_SpaceBDRNGT] = None
        self._np_rng : Optional[np.random.Generator] = None
        self._device = device
        if seed is not None:
            self.seed(seed)

    @property
    def device(self) -> _SpaceBDeviceT:
        return self._device
    
    @abc.abstractmethod
    def to_device(self, device : _SpaceBDeviceT) -> "Space[SpaceDataT, _GymDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]":
        raise NotImplementedError

    @abc.abstractmethod
    def to_backend(self, backend : Type[ComputeBackend], device : Optional[Any]) -> "Space":
        raise NotImplementedError

    @property
    def np_rng(self) -> np.random.Generator:
        if self._np_rng is None:
            self.seed()

        return self._np_rng

    @property
    def rng(self) -> _SpaceBDRNGT:
        if self._rng is None:
            self.seed()

        return self._rng

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Return the shape of the space as an immutable property."""
        return self._shape

    @property
    def is_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`."""
        raise NotImplementedError

    def sample(self, **kwargs) -> SpaceDataT:
        """Randomly sample an element of this space.

        Can be uniform or non-uniform sampling based on boundedness of space.

        Args:
            mask: A mask used for sampling, expected ``dtype=int`` and see sample implementation for expected shape.

        Returns:
            A sampled actions from the space
        """
        raise NotImplementedError

    def seed(self, seed: int | None = None) -> None:
        """Seed the PRNG of this space and possibly the PRNGs of subspaces."""
        self._rng = self.backend.random_number_generator(seed, self._device)
        self._np_rng = np.random.default_rng(seed)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        raise NotImplementedError

    def __contains__(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return self.contains(x)
    
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
    def from_other_backend(self, other_data : Any) -> SpaceDataT:
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
    def from_gym_space(
        gym_space : gym.Space,
        backend : Type[ComputeBackend[Any, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]],
        dtype : Optional[_SpaceBDTypeT] = None,
        device : Optional[_SpaceBDeviceT] = None,
    ) -> "Space[SpaceDataT, _GymDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]":
        if type(gym_space) not in _gym_to_space_mappings:
            raise NotImplementedError(f"Conversion from gym space of type {type(gym_space)} is not supported.")

        return _gym_to_space_mappings[type(gym_space)].from_gym_space(
            gym_space,
            backend=backend,
            dtype=dtype,
            device=device
        )
