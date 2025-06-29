from typing import Any, Generic, Iterable, Union, Mapping, Sequence, TypeVar, Optional, Tuple, Literal
import numpy as np
from unienv_interface.backends import ComputeBackend, ArrayAPIArray
import abc

SpaceDataT = TypeVar("SpaceDataT", covariant=True)
_SpaceBDeviceT = TypeVar("_SpaceBDeviceT", covariant=True)
_SpaceBDTypeT = TypeVar("_SpaceBDTypeT", covariant=True)
_SpaceBDRNGT = TypeVar("_SpaceBDRNGT", covariant=True)
class Space(abc.ABC, Generic[SpaceDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]):
    def __init__(
        self,
        backend : ComputeBackend[ArrayAPIArray, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT],
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
    def to(
        self, 
        backend: Optional[ComputeBackend] = None,
        device: Optional[Union[_SpaceBDeviceT, Any]] = None,
    ) -> Union["Space[SpaceDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]", "Space"]:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Return the shape of the space as an immutable property."""
        return self._shape

    @abc.abstractmethod
    def sample(self, rng : _SpaceBDRNGT, **kwargs) -> Tuple[_SpaceBDRNGT, SpaceDataT]:
        raise NotImplementedError

    @abc.abstractmethod
    def create_empty(
        self
    ) -> SpaceDataT:
        """Create an empty data structure for this space."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_bounded(self, manner: Literal["both", "below", "above"] = "both") -> bool:
        """Return boolean specifying if this space is bounded in the specified manner."""
        raise NotImplementedError

    @abc.abstractmethod
    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        raise NotImplementedError
    
    def __eq__(self, other : "Space"):
        """Return boolean specifying if this space is equal to another space."""
        return self is other

    def __contains__(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return self.contains(x)
    
    def __repr__(self) -> str:
        return self.get_repr(
            abbreviate=False,
            include_backend=True,
            include_device=True,
            include_dtype=True
        )

    def __str__(self) -> str:
        return self.get_repr(
            abbreviate=True,
            include_backend=True,
            include_device=True,
            include_dtype=False
        )

    @abc.abstractmethod
    def get_repr(
        self,
        abbreviate : bool = False,
        include_backend : bool = True,
        include_device : bool = True,
        include_dtype : bool = True,
    ) -> str:
        """Return a string representation of the space."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def data_to(
        self, 
        data : SpaceDataT, 
        backend : Optional[ComputeBackend] = None,
        device : Optional[Union[_SpaceBDeviceT, Any]] = None
    ) -> Union[SpaceDataT, Any]:
        """Convert data to another backend."""
        raise NotImplementedError

    @staticmethod
    def abbr_device(spaces : "Iterable[Space[Any, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]]") -> Optional[_SpaceBDeviceT]:
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