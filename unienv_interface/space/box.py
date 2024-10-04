from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, List, Type, Literal, Union
import numpy as np
from .space import Space
from unienv_interface.backends import ComputeBackend
import array_api_compat
import gymnasium as gym

BoxArrayT = TypeVar("BoxArrayT", covariant=True)
_BoxBDeviceT = TypeVar("_BoxBDeviceT", covariant=True)
_BoxBDTypeT = TypeVar("_BoxBDTypeT", covariant=True)
_BoxBDRNGT = TypeVar("_BoxBDRNGT", covariant=True)

class Box(Space[BoxArrayT, np.ndarray, _BoxBDeviceT, _BoxBDTypeT, _BoxBDRNGT]):
    def __init__(
        self,
        backend : Type[ComputeBackend[BoxArrayT, _BoxBDeviceT, _BoxBDTypeT, _BoxBDRNGT]],
        low: SupportsFloat | BoxArrayT,
        high: SupportsFloat | BoxArrayT,
        dtype: _BoxBDTypeT,
        device : Optional[_BoxBDeviceT] = None,
        shape: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ):
        assert (
            dtype is not None
        ), "Box dtype must be explicitly provided, cannot be None."
        assert (
            backend.dtype_is_real_floating(dtype) or backend.dtype_is_real_integer(dtype)
        ), f"Box dtype must be a real floating or integer type, actual dtype: {dtype}"
        self.dtype = dtype
        self._dtype_is_float = backend.dtype_is_real_floating(self.dtype)

        array_api_workspace = self.backend.array_api_namespace

        # determine shape if it isn't provided directly
        if shape is not None:
            assert all(
                np.issubdtype(type(dim), np.integer) for dim in shape
            ), f"Expect all shape elements to be an integer, actual type: {tuple(type(dim) for dim in shape)}"
            shape = tuple(int(dim) for dim in shape)  # This changes any np types to int
        elif backend.is_backendarray(low):
            shape = low.shape
        elif backend.is_backendarray(high):
            shape = high.shape
        elif isinstance(low, (int, float)) and isinstance(high, (int, float)):
            shape = (1,)
        else:
            raise ValueError(
                f"Box shape is inferred from low and high, expect their types to be backend array, an integer or a float, actual type low: {type(low)}, high: {type(high)}"
            )

        # Capture the boundedness information
        _low = array_api_workspace.full(shape, low, dtype=float, device=device) if isinstance(low, (int, float)) else low
        self.bounded_below = -array_api_workspace.inf < _low

        _high = array_api_workspace.full(shape, high, dtype=float, device=device) if isinstance(high, (int, float)) else high
        self.bounded_above = array_api_workspace.inf > _high

        assert not array_api_workspace.any(array_api_workspace.isnan(_low)), f"low contains NaN values: {_low}"
        assert not array_api_workspace.isnan(_high), f"high contains NaN values: {_high}"

        assert (
            _low.shape == shape
        ), f"_low.shape doesn't match provided shape, _low.shape: {_low.shape}, shape: {shape}"
        assert (
            _low.shape == shape
        ), f"_low.shape doesn't match provided shape, high.shape: {_low.shape}, shape: {shape}"

        self._shape: tuple[int, ...] = shape

        self.low = array_api_workspace.astype(_low, dtype=self.dtype, device=device)
        self.high = array_api_workspace.astype(_high, dtype=self.dtype, device=device)

        super().__init__(
            backend=backend,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def to_device(self, device : _BoxBDeviceT) -> "Box[BoxArrayT, _BoxBDeviceT, _BoxBDTypeT, _BoxBDRNGT]":
        return Box(
            backend=self.backend,
            low=self.low,
            high=self.high,
            dtype=self.dtype,
            shape=self.shape,
            device=device,
            seed=self.seed
        )

    def to_backend(self, backend : Type[ComputeBackend], device : Optional[Any]) -> "Box":
        new_low = backend.from_dlpack(self.low)
        new_high = backend.from_dlpack(self.high)

        return Box(
            backend=backend,
            low=new_low,
            high=new_high,
            dtype=new_low.dtype,
            shape=self.shape,
            device=device
        )

    @property
    def is_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True
    
    @property
    def flat_dim(self) -> int:
        """Return the shape of the space as an immutable property."""
        return int(np.prod(self.shape))
    
    def flatten(self, data : BoxArrayT) -> BoxArrayT:
        """Flatten the data."""
        return self.backend.array_api_namespace.reshape(data, (-1,))
    
    def unflatten(self, data : BoxArrayT) -> BoxArrayT:
        """Unflatten the data."""
        return self.backend.array_api_namespace.reshape(data, self.shape)

    def is_bounded(self, manner: Literal["both", "below", "above"] = "both") -> bool:
        below = bool(self.backend.array_api_namespace.all(self.bounded_below))
        above = bool(self.backend.array_api_namespace.all(self.bounded_above))
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError(
                f"manner is not in {{'below', 'above', 'both'}}, actual value: {manner}"
            )

    def sample(self, mask: None = None) -> BoxArrayT:
        r"""Generates a single random sample inside the Box.

        In creating a sample of the box, each coordinate is sampled (independently) from a distribution
        that is chosen according to the form of the interval:

        * :math:`[a, b]` : uniform distribution
        * :math:`[a, \infty)` : shifted exponential distribution
        * :math:`(-\infty, b]` : shifted negative exponential distribution
        * :math:`(-\infty, \infty)` : normal distribution

        Args:
            mask: A mask for sampling values from the Box space, currently unsupported.

        Returns:
            A sampled value from the Box
        """
        if mask is not None:
            raise ValueError(
                f"Box.sample cannot be provided a mask, actual value: {mask}"
            )

        high = self.high if self._dtype_is_float else self.high + 1
        sample = self.backend.array_api_namespace.empty(self.shape, dtype=self.dtype, device=self.device)

        # Masking arrays which classify the coordinates according to interval type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        self._rng, sample[unbounded] = self.backend.random_normal(self.rng, shape=unbounded[unbounded].shape, dtype=self.dtype, device=self.device)

        self._rng, exponential_generated = self.backend.random_exponential(self.rng, shape=low_bounded[low_bounded].shape, dtype=self.dtype, device=self.device)
        sample[low_bounded] = (
            exponential_generated
            + self.low[low_bounded]
        )

        self._rng, exponential_generated = self.backend.random_exponential(self.rng, shape=upp_bounded[upp_bounded].shape, dtype=self.dtype, device=self.device)
        sample[upp_bounded] = (
            -exponential_generated
            + high[upp_bounded]
        )

        if self._dtype_is_float:
            self._rng, sample[bounded] = self.backend.random_uniform(
                self.rng, shape=bounded[bounded].shape, lower_bound=0.0, upper_bound=1.0, 
                dtype=self.dtype, device=self.device
            ) * (high[bounded] - self.low[bounded]) + self.low[bounded]
        else:
            self._rng, sample[bounded] = self.backend.array_api_namespace.floor(
                self.backend.random_uniform(
                    self.rng, shape=bounded[bounded].shape, lower_bound=0.0, upper_bound=1.0,
                    dtype=self.dtype, device=self.device
                ) * (high[bounded] - self.low[bounded]) + self.low[bounded]
            )

        if not self._dtype_is_float:
            sample = self.backend.array_api_namespace.floor(sample)

        return self.backend.array_api_namespace.astype(sample, self.dtype)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not self.backend.is_backendarray(x):
            raise ValueError(
                f"Box.contains expects x to be a backend array, actual type: {type(x)}"
            )

        return bool(
            self.backend.array_api_namespace.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and self.backend.array_api_namespace.all(x >= self.low)
            and self.backend.array_api_namespace.all(x <= self.high)
        )

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include bounds, shape and dtype.
        If a bound is uniform, only the corresponding scalar will be given to avoid redundant and ugly strings.

        Returns:
            A representation of the space
        """
        return f"Box({self.backend.__name__}, {self.low}, {self.high}, {self.shape}, {self.dtype}, {self.device})"

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        return (
            isinstance(other, Box)
            and (self.shape == other.shape)
            # and (self.dtype == other.dtype)
            and self.backend.array_api_namespace.allclose(self.low, other.low)
            and self.backend.array_api_namespace.allclose(self.high, other.high)
        )
    
    def to_jsonable(self, sample_n: Sequence[BoxArrayT]) -> list[np.ndarray]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [self.backend.to_numpy(sample).tolist() for sample in sample_n]

    def from_jsonable(self, sample_n: Sequence[Sequence[float | int]]) -> list[BoxArrayT]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [self.backend.from_numpy(np.asarray(sample), dtype=self.dtype, device=self.device) for sample in sample_n]

    def from_gym_data(self, gym_data: np.ndarray) -> BoxArrayT:
        return self.backend.from_numpy(gym_data, dtype=self.dtype, device=self.device)
    
    def from_other_backend(self, other_data: Any) -> BoxArrayT:
        new_tensor = self.backend.from_dlpack(other_data)
        return self.from_same_backend(new_tensor)

    def from_same_backend(self, other_data: BoxArrayT) -> BoxArrayT:
        new_tensor = other_data
        if self.dtype is not None:
            new_tensor = self.backend.array_api_namespace.astype(new_tensor, dtype=self.dtype, device=self.device)
        elif self.device is not None:
            new_tensor = array_api_compat.to_device(new_tensor, device=self.device)
        
        return new_tensor

    def to_gym_data(self, data: BoxArrayT) -> np.ndarray:
        return self.backend.to_numpy(data)
    
    def to_gym_space(self) -> gym.spaces.Box:
        """Convert this space to a gym space."""
        new_low = self.backend.to_numpy(self.low)
        new_high = self.backend.to_numpy(self.high)
        return gym.spaces.Box(
            low=new_low,
            high=new_high,
            dtype=new_low.dtype,
            seed=self.np_rng.integers(0)
        )

