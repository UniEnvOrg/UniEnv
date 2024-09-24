from typing import Any, Generic, Iterable, Union, Mapping, Sequence, TypeVar, Optional, Tuple, Type, List
import numpy as np
from unienv_interface.backends.base import ComputeBackend
import abc

SpaceDataT = TypeVar("SpaceDataT", covariant=True)
_SpaceBDeviceT = TypeVar("_SpaceBDeviceT", covariant=True)
_SpaceBDTypeT = TypeVar("_SpaceBDTypeT", covariant=True)
_SpaceBDRNGT = TypeVar("_SpaceBDRNGT", covariant=True)
class Space(abc.ABC, Generic[SpaceDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]):
    def __init__(
        self,
        backend : Type[ComputeBackend[Any, Any, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]],
        shape: Optional[Sequence[int]] = None,
        device : Optional[_SpaceBDeviceT] = None,
        dtype: Optional[_SpaceBDTypeT] = None,
        seed: Optional[int] = None,
    ):
        """Constructor of :class:`Space`.

        Args:
            shape (Optional[Sequence[int]]): If elements of the space are numpy arrays, this should specify their shape.
            dtype (Optional[Type | str]): If elements of the space are numpy arrays, this should specify their dtype.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space
        """
        self.backend = backend
        self._shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self._rng : Optional[_SpaceBDRNGT] = None
        self._device = device
        if seed is not None:
            self.seed(seed)

    @property
    def device(self) -> _SpaceBDeviceT:
        return self._device
    
    @abc.abstractmethod
    def to_device(self, device : Optional[_SpaceBDeviceT]) -> "Space[SpaceDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]":
        raise NotImplementedError

    @abc.abstractmethod
    def to_backend(self, backend : Type[ComputeBackend]) -> "Space":
        raise NotImplementedError

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

    def sample(self, mask: Any | None = None) -> SpaceDataT:
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

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        raise NotImplementedError

    def __contains__(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return self.contains(x)
    
    def to_jsonable(self, sample_n: Sequence[SpaceDataT]) -> List[Any]:
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return list(sample_n)

    def from_jsonable(self, sample_n: List[Any]) -> List[SpaceDataT]:
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n