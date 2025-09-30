from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, List, Type, Literal, Union, Tuple
import numpy as np
from ..space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

class BoxSpace(Space[BArrayType, BDeviceType, BDtypeType, BRNGType]):
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

        # determine shape if it isn't provided directly
        if shape is not None:
            assert all(
                np.issubdtype(type(dim), np.integer) for dim in shape
            ), f"Expect all shape elements to be an integer, actual type: {tuple(type(dim) for dim in shape)}"
            shape = tuple(int(dim) for dim in shape)  # This changes any np types to int
        elif backend.is_backendarray(low) or backend.is_backendarray(high):
            shapes = []
            if backend.is_backendarray(low):
                shapes.append(low.shape)
            if backend.is_backendarray(high):
                shapes.append(high.shape)
            shape = backend.broadcast_shapes(*shapes) if len(shapes) > 1 else shapes[0]
        elif isinstance(low, (int, float)) and isinstance(high, (int, float)):
            shape = ()
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

        # Cast low and high to backend arrays
        if backend.dtype_is_real_integer(dtype):
            dttype_iinfo = backend.iinfo(dtype)
            dtype_min = dttype_iinfo.min
            dtype_max = dttype_iinfo.max

            if isinstance(low, (int, float)) and (low == backend.inf or low == -backend.inf):
                low = dtype_min if low == -backend.inf else dtype_max
            if isinstance(high, (int, float)) and (high == backend.inf or high == -backend.inf):
                high = dtype_max if high == backend.inf else dtype_min
            
        if isinstance(low, (int, float)):
            _low = backend.full([1] * len(shape), low, dtype=dtype, device=device)
        else:
            _low = backend.astype(low, dtype)
        if isinstance(high, (int, float)):
            _high = backend.full([1] * len(shape), high, dtype=dtype, device=device)
        else:
            _high = backend.astype(high, dtype)
        
        _low = backend.abbreviate_array(
            _low,
            try_cast_scalar=False
        )
        _high = backend.abbreviate_array(
            _high,
            try_cast_scalar=False
        )
        
        assert backend.all(_low <= _high), f"low is greater than high: low={_low}, high={_high}"

        if device is not None:
            _low = backend.to_device(_low, device)
            _high = backend.to_device(_high, device)

        assert not backend.any(backend.isnan(_low)), f"low contains NaN values: {_low}"
        assert not backend.any(backend.isnan(_high)), f"high contains NaN values: {_high}"
        
        assert len(_low.shape) == len(shape) and all(
            (_low.shape[i] == shape[i] or (_low.shape[i] == 1 and shape[i] >= 1)) for i in range(len(shape))
        ), f"_low.shape doesn't match provided shape and is not broadcastable, _low.shape: {_low.shape}, shape: {shape}"
        assert len(_high.shape) == len(shape) and all(
            (_high.shape[i] == shape[i] or (_high.shape[i] == 1 and shape[i] >= 1)) for i in range(len(shape))
        ), f"_low.shape doesn't match provided shape and is not broadcastable, high.shape: {_low.shape}, shape: {shape}"

        self._shape: Tuple[int, ...] = shape
        
        # Check bounded below and above for integer dtype
        if not self._dtype_is_float:
            bounded_below = -self.backend.inf < _low
            bounded_above = self.backend.inf > _high
            below = bool(self.backend.all(bounded_below))
            above = bool(self.backend.all(bounded_above))
            assert below and above, f"Box bounds must be finite for integer dtype, actual low: {self.low}, high: {self.high}"

        self._low = _low
        self._high = _high

    @property
    def low(self) -> BArrayType:
        return self.backend.broadcast_to(self._low, self.shape)
    
    @property
    def high(self) -> BArrayType:
        return self.backend.broadcast_to(self._high, self.shape)

    def to(
        self, 
        backend : Optional[ComputeBackend] = None,
        device : Optional[Union[BDeviceType, Any]] = None,
    ) -> Union["BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType]", "BoxSpace"]:
        if (backend is None or backend==self.backend) and (device is None or device==self.device):
            return self
        
        if backend is not None and backend != self.backend:
            new_low = backend.from_other_backend(self.backend, self._low)
            new_high = backend.from_other_backend(self.backend, self._high)
            new_dtype = new_low.dtype
            new_device = device
        else:
            new_low = self._low
            new_high = self._high
            new_dtype = self.dtype
            new_device = self.device if device is None else device
        
        return BoxSpace(
            backend=backend or self.backend,
            low=new_low,
            high=new_high,
            dtype=new_dtype,
            shape=self.shape,
            device=new_device
        )

    def is_bounded(self, manner = "both"):
        bounded_below = -self.backend.inf < self._low
        bounded_above = self.backend.inf > self._high
        below = bool(self.backend.all(bounded_below))
        above = bool(self.backend.all(bounded_above))
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError(
                f"Unsupported manner {manner}"
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

        low = self.low
        

        if self._dtype_is_float:
            high = self.high
            sample = self.backend.empty(self.shape, dtype=self.dtype, device=self.device)

            bounded_below = -self.backend.inf < low
            bounded_above = self.backend.inf > high

            # Masking arrays which classify the coordinates according to interval type
            unbounded = self.backend.logical_and(
                self.backend.logical_not(bounded_below), 
                self.backend.logical_not(bounded_above)
            )
            upp_bounded = self.backend.logical_and(
                self.backend.logical_not(bounded_below), 
                bounded_above
            )
            low_bounded = self.backend.logical_and(
                bounded_below, 
                self.backend.logical_not(bounded_above)
            )
            bounded = self.backend.logical_and(
                bounded_below, 
                bounded_above
            )

            # Vectorized sampling by interval type
            rng, unbounded_sample = self.backend.random.random_normal(unbounded[unbounded].shape, rng=rng, dtype=self.dtype, device=self.device)
            sample = self.backend.at(sample)[unbounded].set(unbounded_sample)

            rng, exponential_generated = self.backend.random.random_exponential(low_bounded[low_bounded].shape, rng=rng, dtype=self.dtype, device=self.device)
            low_bounded_sample = exponential_generated + low[low_bounded]
            sample = self.backend.at(sample)[low_bounded].set(low_bounded_sample)

            rng, exponential_generated = self.backend.random.random_exponential(upp_bounded[upp_bounded].shape, rng=rng, dtype=self.dtype, device=self.device)
            upp_bounded_sample = high[upp_bounded] - exponential_generated
            sample = self.backend.at(sample)[upp_bounded].set(upp_bounded_sample)

            rng, bounded_sample = self.backend.random.random_uniform(
                bounded[bounded].shape, rng=rng, low=0.0, high=1.0,
                dtype=self.dtype, device=self.device
            )
            bounded_sample *= (high[bounded] - low[bounded])
            bounded_sample += low[bounded]
            sample = self.backend.at(sample)[bounded].set(bounded_sample)
        else:
            high=self.high + 1
            rng, sample = self.backend.random.random_uniform(
                self.shape, rng=rng, low=0.0, high=1.0, device=self.device
            )
            sample *= (high - low)
            sample = self.backend.floor(sample)

            # Fix for floating point errors
            floating_point_error_idx = sample >= high - low
            if self.backend.any(floating_point_error_idx):
                sample = self.backend.at(sample)[floating_point_error_idx].set(
                    self.backend.astype(high[floating_point_error_idx] - low[floating_point_error_idx] - 1, sample.dtype)
                )

            sample = self.backend.astype(sample, self.dtype)
            sample += low
        return rng, sample

    def create_empty(self) -> BArrayType:
        return self.backend.empty(
            self.shape,
            dtype=self.dtype,
            device=self.device
        )

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not self.backend.is_backendarray(x):
            raise ValueError(
                f"Box.contains expects x to be a backend array, actual type: {type(x)}"
            )

        return bool(
            self.backend.can_cast(x, self.dtype)
            and x.shape == self.shape
            and self.backend.all(x >= self.low)
            and self.backend.all(x <= self.high)
        )
    
    def clip(self, x: BArrayType) -> BArrayType:
        """Clip the values of x to be within the bounds of this space."""
        return self.backend.clip(x, self._low, self._high)

    def get_repr(
        self,
        abbreviate : bool = False,
        include_backend : bool = True,
        include_device : bool = True,
        include_dtype : bool = True,
    ) -> str:
        if abbreviate:
            ret = f"[{self.backend.abbreviate_array(self._low, try_cast_scalar=True)}, {self.backend.abbreviate_array(self._high, try_cast_scalar=True)}) {self.shape}"
        else:
            ret = f"BoxSpace({self.backend.abbreviate_array(self._low, try_cast_scalar=True)}, {self.backend.abbreviate_array(self._high, try_cast_scalar=True)}, {self.shape}"
        if include_backend:
            ret += f", backend={self.backend}"
        if include_device:
            ret += f", device={self.device}"
        if include_dtype:
            ret += f", dtype={self.dtype}"
        if not abbreviate:
            ret += ")"
        return ret

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        try:
            return bool(
                isinstance(other, BoxSpace)
                and (self.backend == other.backend)
                and (self.shape == other.shape)
                and (self.dtype == other.dtype)
                and self.backend.all(self.backend.isclose(self.low, other.low))
                and self.backend.all(self.backend.isclose(self.high, other.high))
            )
        except:
            return False
    
    def data_to(
        self, 
        data : BArrayType, 
        backend : Optional[ComputeBackend] = None,
        device : Optional[Union[BDeviceType, Any]] = None
    ) -> Union[BArrayType, Any]:
        """Convert data to another backend."""
        if backend is not None and backend != self.backend:
            data = backend.from_other_backend(self.backend, data)
        if device is not None:
            data = (backend or self.backend).to_device(data, device)
        return data
