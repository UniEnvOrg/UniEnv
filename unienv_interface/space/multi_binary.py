"""Implementation of a space consisting of finitely many elements."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal
import numpy as np
from .space import Space
from unienv_interface.backends import ComputeBackend
import array_api_compat
import gymnasium as gym

MultiBArrayT = TypeVar("MultiBArrayT", covariant=True)
_MultiBDeviceT = TypeVar("_MultiBDeviceT", covariant=True)
_MultiBDTypeT = TypeVar("_MultiBDTypeT", covariant=True)
_MultiBDRNGT = TypeVar("_MultiBDRNGT", covariant=True)

class MultiBinary(Space[MultiBArrayT, np.ndarray, _MultiBDeviceT, _MultiBDTypeT, _MultiBDRNGT]):
    def __init__(
        self,
        backend : Type[ComputeBackend[MultiBArrayT, _MultiBDeviceT, _MultiBDTypeT, _MultiBDRNGT]],
        shape: Sequence[int],
        dtype: Optional[_MultiBDTypeT] = None,
        device : Optional[_MultiBDeviceT] = None,
        seed: Optional[int] = None,
    ):
        assert dtype is None or backend.dtype_is_real_integer(dtype), f"Invalid dtype {dtype}"
        super().__init__(
            backend=backend,
            shape=tuple(shape),
            device=device,
            dtype=dtype,
            seed=seed,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape  # type: ignore

    def to_device(self, device: _MultiBDeviceT) -> "MultiBinary[MultiBArrayT, _MultiBDeviceT, _MultiBDTypeT, _MultiBDRNGT]":
        return MultiBinary(
            backend=self.backend,
            shape=self.shape,
            device=device,
            dtype=self.dtype,
        )
    
    def to_backend(self, backend: ComputeBackend, device: Optional[Any]) -> "MultiBinary":
        return MultiBinary(
            backend=backend,
            shape=self.shape,
            device=device,
            dtype=self.dtype,
        )

    @property
    def is_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True
    
    @property
    def flat_dim(self) -> int:
        """Return the shape of the space as an immutable property."""
        return int(np.prod(self.shape))
    
    def flatten(self, data : MultiBArrayT) -> MultiBArrayT:
        return self.backend.array_api_namespace.reshape(data, (-1,))
    
    def unflatten(self, data : MultiBArrayT) -> MultiBArrayT:
        return self.backend.array_api_namespace.reshape(data, self.shape)

    def sample(self) -> MultiBArrayT:
        return self.backend.random_discrete_uniform(self.rng, self.shape, 0, 2, self.dtype, self.device)

    def contains(self, x: MultiBArrayT) -> bool:
        return bool(
            self.backend.is_backendarray(x)
            and self.shape == x.shape
            and self.backend.array_api_namespace.all(self.backend.array_api_namespace.logical_or(x == 0, x == 1))
        )

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"MultiBinary({self.backend}, {self.shape}, {self.dtype}, {self.device})"

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return isinstance(other, MultiBinary) and self.shape == other.shape
    
    def to_jsonable(self, sample_n: Sequence[MultiBArrayT]) -> list[Sequence[int]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return np.array([self.backend.to_numpy(sample) for sample in sample_n]).tolist()

    def from_jsonable(self, sample_n: list[Sequence[int]]) -> list[MultiBArrayT]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [self.backend.from_numpy(np.asarray(sample, self.dtype), self.dtype, self.device) for sample in sample_n]

    def from_gym_data(self, gym_data : np.ndarray) -> MultiBArrayT:
        """Convert a gym space to this space."""
        return self.backend.from_numpy(gym_data, self.dtype, self.device)
    
    def from_other_backend(self, other_data : Any) -> MultiBArrayT:
        new_tensor = self.backend.from_dlpack(other_data)
        return self.from_same_backend(new_tensor)
    
    def from_same_backend(self, other_data : MultiBArrayT) -> MultiBArrayT:
        new_tensor = other_data
        if self.dtype is not None:
            new_tensor = self.backend.array_api_namespace.astype(new_tensor, self.dtype)
        if self.device is not None:
            new_tensor = array_api_compat.to_device(new_tensor, device=self.device)
        
        return new_tensor

    def to_gym_data(self, data : MultiBArrayT) -> np.ndarray:
        """Convert this space to a gym space."""
        return self.backend.to_numpy(data).astype(np.int8)
    
    def to_gym_space(self) -> gym.Space:
        """Convert this space to a gym space."""
        return gym.spaces.MultiBinary(self.shape, seed=self.np_rng)
