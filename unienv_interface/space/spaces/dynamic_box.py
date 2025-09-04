from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, List, Type, Literal, Union, Tuple
import numpy as np
from ..space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

class DynamicBoxSpace(Space[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    @staticmethod
    def pad_array_on_axis(
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        data : BArrayType, # (..., T, ...)
        axis : int,
        target_size : int,
        fill_value : Union[int, float] = 0,
    ) -> BArrayType:
        n_pad = target_size - data.shape[axis]
        if n_pad != 0:
            assert n_pad > 0
            return backend.concat([
                data,
                backend.full(data.shape[:axis] + (n_pad,) + data.shape[axis+1:], fill_value, device=backend.device(data), dtype=data.dtype)
            ], axis=axis)
        return data

    @staticmethod
    def get_array_axis_length(
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        data : BArrayType, # (..., T, ...)
        axis : int,
        fill_value : Union[int, float] = 0,
    ) -> int:
        if axis < 0:
            axis += len(data.shape)
        new_shape = (int(np.prod(data.shape[:axis])), data.shape[axis], int(np.prod(data.shape[axis+1:])))
        fill_mask = backend.reshape(backend.equal(
            data, fill_value
        ), new_shape) # (B, T, D)
        fill_mask = backend.all(fill_mask, axis=0) # (T, D)
        fill_mask = backend.all(fill_mask, axis=-1) # (T,)
        
        final_len = data.shape[axis]
        for i in reversed(range(data.shape[axis])):
            if fill_mask[i]:
                final_len -= 1
            else:
                break
        return final_len

    @staticmethod
    def unpad_array_on_axis(
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        data : BArrayType, # (..., T, ...)
        axis : int,
        fill_value : Union[int, float] = 0,
    ) -> BArrayType:
        axis_len = __class__.get_array_axis_length(backend, data, axis, fill_value)
        return backend.take(data, backend.arange(axis_len, dtype=backend.default_integer_dtype, device=backend.device(data)), axis=axis)

    def __init__(
        self,
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        low: Union[int, float, BArrayType],
        high: Union[int, float, BArrayType],
        shape_low : Sequence[int],
        shape_high : Sequence[int],
        dtype: BDtypeType,
        device : Optional[BDeviceType] = None,
        fill_value : Union[int, float] = 0,
    ):
        assert (
            dtype is not None
        ), "DynamicBox dtype must be explicitly provided, cannot be None."
        assert (
            backend.dtype_is_real_floating(dtype) or backend.dtype_is_real_integer(dtype)
        ), f"DynamicBox dtype must be a real floating or integer type, actual dtype: {dtype}"
        self._dtype_is_float = backend.dtype_is_real_floating(dtype)

        assert len(shape_low) == len(shape_high), \
            f"shape_low and shape_high must have the same length, actual lengths: {len(shape_low)} and {len(shape_high)}"
        assert all(
            np.issubdtype(type(dim), np.integer) for dim in shape_low
        ), f"Expect all shape_low elements to be an integer, actual type: {tuple(type(dim) for dim in shape_low)}"
        assert all(
            np.issubdtype(type(dim), np.integer) for dim in shape_high
        ), f"Expect all shape_high elements to be an integer, actual type: {tuple(type(dim) for dim in shape_high)}"
        shape_low = tuple(shape_low)
        shape_high = tuple(shape_high)
        assert all(
            dim_low >= 0 and dim_high >= 0 and dim_low <= dim_high
            for dim_low, dim_high in zip(shape_low, shape_high)
        ), f"shape_low and shape_high must be non-negative and shape_low[i] <= shape_high[i], actual: {shape_low}, {shape_high}"
        assert not all(
            dim_low == dim_high for dim_low, dim_high in zip(shape_low, shape_high)
        ), f"shape_low and shape_high must have one dimension that is different, otherwise you can use `BoxSpace`, actual: {shape_low}, {shape_high}"

        super().__init__(
            backend=backend,
            shape=None,
            device=device,
            dtype=dtype,
        )

        # Cast low and high to backend arrays
        if backend.dtype_is_real_integer(dtype):
            dttype_iinfo = backend.iinfo(dtype)
            dtype_min = dttype_iinfo.min
            dtype_max = dttype_iinfo.max
            if isinstance(low, int):
                if low == backend.inf or low == -backend.inf:
                    _low = dtype_min if low == -backend.inf else dtype_max
                else:
                    _low = low
                _low = backend.full([1] * len(shape_low), _low, dtype=dtype, device=device)
            else:
                _low = backend.astype(low, dtype)
            if isinstance(high, int):
                if high == backend.inf or high == -backend.inf:
                    _high = dtype_max if high == backend.inf else dtype_min
                else:
                    _high = high
                _high = backend.full([1] * len(shape_low), _high, dtype=dtype, device=device)
            else:
                _high = backend.astype(high, dtype)
        else:
            if isinstance(low, (int, float)):
                _low = backend.full([1] * len(shape_low), low, dtype=dtype, device=device)
            else:
                _low = backend.astype(low, dtype)
            if isinstance(high, (int, float)):
                _high = backend.full([1] * len(shape_low), high, dtype=dtype, device=device)
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
        
        assert len(_low.shape) == len(shape_low) and all(
            ((_low.shape[i] == shape_low[i] and shape_low[i] == shape_high[i]) or (_low.shape[i] == 1)) for i in range(len(shape_low))
        ), f"_low.shape doesn't match provided shape and is not broadcastable, _low.shape: {_low.shape}, shape_low: {shape_low}, shape_high: {shape_high}"
        assert len(_high.shape) == len(shape_low) and all(
            ((_high.shape[i] == shape_low[i] and shape_low[i] == shape_high[i]) or (_high.shape[i] == 1)) for i in range(len(shape_low))
        ), f"_low.shape doesn't match provided shape and is not broadcastable, high.shape: {_low.shape}, shape_low: {shape_low}, shape_high: {shape_high}"

        self._shape_low = shape_low
        self._shape_high = shape_high
        self._broadcast_shape = tuple(
            1 if dim_low != dim_high else dim_low
            for dim_low, dim_high in zip(shape_low, shape_high)
        )
        
        # Check bounded below and above for integer dtype
        if not self._dtype_is_float:
            bounded_below = -self.backend.inf < _low
            bounded_above = self.backend.inf > _high
            below = bool(self.backend.all(bounded_below))
            above = bool(self.backend.all(bounded_above))
            assert below and above, f"Box bounds must be finite for integer dtype, actual low: {self.low}, high: {self.high}"

        self._low = _low
        self._high = _high
        self.fill_value = fill_value

    def get_low(self, shape : Sequence[int]) -> BArrayType:
        return self.backend.broadcast_to(self._low, shape)
    
    def get_high(self, shape : Sequence[int]) -> BArrayType:
        return self.backend.broadcast_to(self._high, shape)

    @property
    def shape_low(self) -> Tuple[int, ...]:
        """Return the lower shape of the BoxSpace."""
        return tuple(self._shape_low)
    
    @property
    def shape_high(self) -> Tuple[int, ...]:
        """Return the upper shape of the BoxSpace."""
        return tuple(self._shape_high)

    def to(
        self, 
        backend : Optional[ComputeBackend] = None,
        device : Optional[Union[BDeviceType, Any]] = None,
    ) -> Union["DynamicBoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType]", "DynamicBoxSpace"]:
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
        
        return DynamicBoxSpace(
            backend=backend or self.backend,
            low=new_low,
            high=new_high,
            shape_low=self.shape_low,
            shape_high=self.shape_high,
            dtype=new_dtype,
            device=new_device,
            fill_value=self.fill_value,
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

        target_shape = self.backend.random.random_uniform(
            len(self.shape_low),
            rng=rng,
            dtype=self.backend.default_floating_dtype
        )
        target_shape_bounds = self.backend.asarray([float(dim_high + 1 - dim_low) for dim_low, dim_high in zip(self.shape_low, self.shape_high)], dtype=self.backend.default_floating_dtype)
        target_shape = self.backend.floor(target_shape * target_shape_bounds)
        target_shape = tuple(int(target_shape[i]) + self.shape_low[i] for i in range(len(target_shape)))

        
        low = self.get_low(target_shape)

        if self._dtype_is_float:
            high = self.get_high(target_shape)
            sample = self.backend.empty(target_shape, dtype=self.dtype, device=self.device)

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
            high=self.get_high(target_shape) + 1
            rng, sample = self.backend.random.random_uniform(
                target_shape, rng=rng, low=0.0, high=1.0, device=self.device
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

    def pad_data(
        self,
        data : BArrayType,
        start_axis : Optional[int] = None,
        end_axis : Optional[int] = None,
    ) -> BArrayType:
        """Pad the data to the maximum shape of this space."""
        assert self.backend.is_backendarray(data), f"DynamicBoxSpace.pad_data expects data to be a backend array, actual type: {type(data)}"
        axis_slice = slice(start_axis, end_axis)
        for axis_i in range(*axis_slice.indices(len(self.shape_low))):
            if self.shape_high[axis_i] == self.shape_low[axis_i]:
                continue
            data = self.pad_array_on_axis(
                self.backend,
                data,
                axis_i,
                self.shape_high[axis_i],
                fill_value=self.fill_value
            )
        return data
    
    def unpad_data(
        self,
        data : BArrayType,
        start_axis : Optional[int] = None,
        end_axis : Optional[int] = None,
    ) -> BArrayType:
        """Unpad the data to the minimum shape of this space."""
        assert self.backend.is_backendarray(data), f"DynamicBoxSpace.unpad_data expects data to be a backend array, actual type: {type(data)}"
        axis_slice = slice(start_axis, end_axis)
        for axis_i in range(*axis_slice.indices(len(self.shape_low))):
            if self.shape_high[axis_i] == self.shape_low[axis_i]:
                continue
            data = self.unpad_array_on_axis(
                self.backend,
                data,
                axis_i,
                fill_value=self.fill_value
            )
        return data

    def create_empty(self):
        # Create an empty data structure with maximum shape
        return self.backend.empty(
            self.shape_high,
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
            and self.shape_contains(x.shape)
            and self.backend.all(x >= self.low)
            and self.backend.all(x <= self.high)
        )
    
    def shape_contains(self, shape: Sequence[int]) -> bool:
        """Check if the shape is within the bounds of this space."""
        if len(shape) != len(self.shape_low):
            return False
        return all(
            self.shape_low[i] <= shape[i] <= self.shape_high[i]
            for i in range(len(self.shape_low))
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
            ret = f"[{self.backend.abbreviate_array(self._low, try_cast_scalar=True)}, {self.backend.abbreviate_array(self._high, try_cast_scalar=True)}) {self.shape_low}, {self.shape_high}, fill_value={self.fill_value}"
        else:
            ret = f"DynamicBoxSpace({self.backend.abbreviate_array(self._low, try_cast_scalar=True)}, {self.backend.abbreviate_array(self._high, try_cast_scalar=True)}, {self.shape_low}, {self.shape_high}, fill_value={self.fill_value}"
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
                isinstance(other, DynamicBoxSpace)
                and (self.backend == other.backend)
                and (self.shape_low == other.shape_low)
                and (self.shape_high == other.shape_high)
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
