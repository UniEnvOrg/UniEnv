from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, List, Type, Literal, Union, Tuple
import numpy as np
from .space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
import gymnasium as gym

def abbreviate_array(backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType], array : BArrayType) -> Union[float, int, BArrayType]:
    first_idx = [0] * len(array.shape)
    first_elem = array[tuple(first_idx)]
    if backend.array_api_namespace.all(array == first_elem):
        return first_elem
    return array

class Box(Space[BArrayType, np.ndarray, BDeviceType, BDtypeType, BRNGType]):
    def __init__(
        self,
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        low: SupportsFloat | BArrayType,
        high: SupportsFloat | BArrayType,
        dtype: BDtypeType,
        device : Optional[BDeviceType] = None,
        shape: Optional[Sequence[int]] = None,
    ):
        assert (
            dtype is not None
        ), "Box dtype must be explicitly provided, cannot be None."
        assert (
            backend.dtype_is_real_floating(dtype) or backend.dtype_is_real_integer(dtype)
        ), f"Box dtype must be a real floating or integer type, actual dtype: {dtype}"
        self._dtype_is_float = backend.dtype_is_real_floating(dtype)

        array_api_workspace = backend.array_api_namespace

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
        
        super().__init__(
            backend=backend,
            shape=shape,
            device=device,
            dtype=dtype,
        )

        _low = array_api_workspace.full(shape, low, dtype=float, device=device) if isinstance(low, (int, float)) else low
        _high = array_api_workspace.full(shape, high, dtype=float, device=device) if isinstance(high, (int, float)) else high

        # Before performing any operation on low and high arrays, move to the device to avoid jax tiling problems
        if device is not None:
            _low = backend.to_device(_low, device)
            _high = backend.to_device(_high, device)

        assert not array_api_workspace.any(array_api_workspace.isnan(_low)), f"low contains NaN values: {_low}"
        assert not array_api_workspace.any(array_api_workspace.isnan(_high)), f"high contains NaN values: {_high}"
        
        assert (
            _low.shape == shape
        ), f"_low.shape doesn't match provided shape, _low.shape: {_low.shape}, shape: {shape}"
        assert (
            _low.shape == shape
        ), f"_low.shape doesn't match provided shape, high.shape: {_low.shape}, shape: {shape}"

        self._shape: tuple[int, ...] = shape

        self._low = array_api_workspace.astype(_low, self.dtype)
        self._high = array_api_workspace.astype(_high, self.dtype)
        assert array_api_workspace.all(self.low <= self.high), f"low is greater than high: low={self.low}, high={self.high}"
        
        self._recalculate_boundedness()

    @property
    def low(self) -> BArrayType:
        return self._low
    
    @property
    def high(self) -> BArrayType:
        return self._high

    def _recalculate_boundedness(self):
        self._bounded_below = -self.backend.array_api_namespace.inf < self.low
        self._bounded_above = self.backend.array_api_namespace.inf > self.high
        if not self._dtype_is_float:
            below = bool(self.backend.array_api_namespace.all(self._bounded_below))
            above = bool(self.backend.array_api_namespace.all(self._bounded_above))
            assert below and above, f"Box bounds must be finite for integer dtype, actual low: {self.low}, high: {self.high}"

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def to_device(self, device : BDeviceType) -> "Box[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        return Box(
            backend=self.backend,
            low=self.low,
            high=self.high,
            dtype=self.dtype,
            shape=self.shape,
            device=device
        )

    def to_backend(self, backend : ComputeBackend, device : Optional[Any]) -> "Box":
        new_low = backend.from_other_backend(self.low, self.backend)
        new_high = backend.from_other_backend(self.high, self.backend)

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
    def is_batch_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return len(self.shape) > 1

    @property
    def flat_dim(self) -> int:
        """Return the shape of the space as an immutable property."""
        return int(np.prod(self.shape))
    
    @property
    def batch_flat_dim(self) -> int:
        """Return the shape of the space as an immutable property."""
        assert len(self.shape) > 1, f"Batch flat dim is only defined for batched spaces, actual shape: {self.shape}"
        return int(np.prod(self.shape[1:]))

    def flatten(self, data : BArrayType) -> BArrayType:
        """Flatten the data."""
        return self.backend.array_api_namespace.reshape(data, (-1,))
    
    def flatten_batch(self, data : BArrayType) -> BArrayType:
        """Flatten the data."""
        return self.backend.array_api_namespace.reshape(data, (data.shape[0], -1))

    def unflatten(self, data : BArrayType) -> BArrayType:
        """Unflatten the data."""
        return self.backend.array_api_namespace.reshape(data, self.shape)

    def unflatten_batch(self, data : BArrayType) -> BArrayType:
        """Unflatten the data."""
        return self.backend.array_api_namespace.reshape(data, (data.shape[0],) + self.shape[1:])

    def is_bounded(self, manner: Literal["both", "below", "above"] = "both") -> bool:
        below = bool(self.backend.array_api_namespace.all(self._bounded_below))
        above = bool(self.backend.array_api_namespace.all(self._bounded_above))
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

    def sample(self, rng : BRNGType) -> Tuple[BRNGType, BArrayType]:
        r"""Generates a single random sample inside the Box.

        In creating a sample of the box, each coordinate is sampled (independently) from a distribution
        that is chosen according to the form of the interval:

        * :math:`[a, b]` : uniform distribution
        * :math:`[a, \infty)` : shifted exponential distribution
        * :math:`(-\infty, b]` : shifted negative exponential distribution
        * :math:`(-\infty, \infty)` : normal distribution

        Returns:
            A sampled value from the Box
        """

        if self._dtype_is_float:
            high = self.high
            sample = self.backend.array_api_namespace.empty(self.shape, dtype=self.dtype, device=self.device)

            # Masking arrays which classify the coordinates according to interval type
            unbounded = self.backend.array_api_namespace.logical_and(
                self.backend.array_api_namespace.logical_not(self._bounded_below), 
                self.backend.array_api_namespace.logical_not(self._bounded_above)
            )
            upp_bounded = self.backend.array_api_namespace.logical_and(
                self.backend.array_api_namespace.logical_not(self._bounded_below), 
                self._bounded_above
            )
            low_bounded = self.backend.array_api_namespace.logical_and(
                self._bounded_below, 
                self.backend.array_api_namespace.logical_not(self._bounded_above)
            )
            bounded = self.backend.array_api_namespace.logical_and(
                self._bounded_below, 
                self._bounded_above
            )

            # Vectorized sampling by interval type
            rng, unbounded_sample = self.backend.random_normal(rng, shape=unbounded[unbounded].shape, dtype=self.dtype, device=self.device) if self._dtype_is_float else self.backend.random_normal(rng, shape=unbounded[unbounded].shape, mean=0.0, std=1.0, device=self.device)
            sample = self.backend.replace_inplace(sample, unbounded, unbounded_sample)

            rng, exponential_generated = self.backend.random_exponential(rng, shape=low_bounded[low_bounded].shape, dtype=self.dtype, device=self.device)
            low_bounded_sample = exponential_generated + self.low[low_bounded]
            sample = self.backend.replace_inplace(sample, low_bounded, low_bounded_sample)

            rng, exponential_generated = self.backend.random_exponential(rng, shape=upp_bounded[upp_bounded].shape, dtype=self.dtype, device=self.device)
            upp_bounded_sample = high[upp_bounded] - exponential_generated
            sample = self.backend.replace_inplace(sample, upp_bounded, upp_bounded_sample)

            rng, bounded_sample = self.backend.random_uniform(
                rng, shape=bounded[bounded].shape, lower_bound=0.0, upper_bound=1.0,
                dtype=self.dtype, device=self.device
            )
            bounded_sample *= (high[bounded] - self.low[bounded])
            bounded_sample += self.low[bounded]
            sample = self.backend.replace_inplace(sample, bounded, bounded_sample)

        else:
            high=self.high + 1
            rng, sample = self.backend.random_uniform(
                rng, shape=self.shape, lower_bound=0.0, upper_bound=1.0, device=self.device
            )
            sample *= (high - self.low)
            sample = self.backend.array_api_namespace.floor(sample)

            # Fix for floating point errors
            floating_point_error_idx = sample >= high - self.low
            if self.backend.array_api_namespace.any(floating_point_error_idx):
                sample = self.backend.replace_inplace(
                    sample, floating_point_error_idx, self.backend.array_api_namespace.astype(high[floating_point_error_idx] - self.low[floating_point_error_idx] - 1, sample.dtype)
                )

            sample = self.backend.array_api_namespace.astype(sample, self.dtype)
            sample += self.low
        return rng, sample

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
    
    def clip(self, x: BArrayType) -> BArrayType:
        """Clip the values of x to be within the bounds of this space."""
        return self.backend.array_api_namespace.clip(x, self.low, self.high)

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include bounds, shape and dtype.
        If a bound is uniform, only the corresponding scalar will be given to avoid redundant and ugly strings.

        Returns:
            A representation of the space
        """
        return f"Box({self.backend.__name__}, {abbreviate_array(self.backend, self.low)}, {abbreviate_array(self.backend, self.high)}, {self.shape}, {self.dtype}, {self.device})"

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        return (
            isinstance(other, Box)
            and (self.shape == other.shape)
            # and (self.dtype == other.dtype)
            and self.backend.array_api_namespace.allclose(self.low, other.low)
            and self.backend.array_api_namespace.allclose(self.high, other.high)
        )
    
    def to_jsonable(self, sample_n: Sequence[BArrayType]) -> list[np.ndarray]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [self.backend.to_numpy(sample).tolist() for sample in sample_n]

    def from_jsonable(self, sample_n: Sequence[Sequence[float | int]]) -> list[BArrayType]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [self.backend.from_numpy(np.asarray(sample), dtype=self.dtype, device=self.device) for sample in sample_n]

    def from_gym_data(self, gym_data: np.ndarray) -> BArrayType:
        return self.backend.from_numpy(gym_data, dtype=self.dtype, device=self.device)
    
    def from_other_backend(self, other_data: Any, backend : ComputeBackend) -> BArrayType:
        new_tensor = self.backend.from_other_backend(other_data, backend)
        return self.from_same_backend(new_tensor)

    def from_same_backend(self, other_data: BArrayType) -> BArrayType:
        new_tensor = other_data
        
        if self.device is not None:
            new_tensor = self.backend.to_device(new_tensor, self.device)
            # For some reason jax doesn't have good support for dlpack converted back cpu arrays, so we need to move it to the device first
        if self.dtype is not None:
            new_tensor = self.backend.array_api_namespace.astype(new_tensor, self.dtype)
        
        return new_tensor

    def to_gym_data(self, data: BArrayType) -> np.ndarray:
        return self.backend.to_numpy(data)
    
    def to_gym_space(self) -> gym.spaces.Box:
        """Convert this space to a gym space."""
        new_low = self.backend.to_numpy(self.low)
        new_high = self.backend.to_numpy(self.high)
        return gym.spaces.Box(
            low=new_low,
            high=new_high,
            dtype=new_low.dtype
        )

