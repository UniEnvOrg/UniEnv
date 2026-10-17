from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, List, Type, Literal, Union, Tuple
import math
import numpy as np
from ..space import Space, _is_jax_tracer
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

def _sample_integers_in_closed_interval(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    rng : BRNGType,
    low : BArrayType,
    high : BArrayType,
    shape : Sequence[int],
    dtype : BDtypeType,
    device : Optional[BDeviceType] = None,
) -> Tuple[BRNGType, BArrayType]:
    """Uniformly sample integers from ``[low, high]`` (elementwise, both ends included).

    The interval width is computed in floating point because ``high - low + 1``
    overflows the integer dtype for (near) full-range bounds such as the ones
    produced from ``-inf``/``inf``.
    """
    rng, uniform = backend.random.random_uniform(
        shape, rng=rng, low=0.0, high=1.0, device=device
    )
    float_dtype = uniform.dtype
    low_f = backend.astype(low, float_dtype)
    high_f = backend.astype(high, float_dtype)
    sample = low_f + backend.floor(uniform * (high_f - low_f + 1))
    sample = backend.minimum(sample, high_f) # Fix for floating point errors

    # Floats at or above ``max + 1`` cannot be cast to the integer dtype; such
    # values only arise from rounding ``high`` up, so they map back to ``high``.
    overflow = sample >= float(int(backend.iinfo(low.dtype).max) + 1)
    sample = backend.where(overflow, backend.zeros_like(sample), sample)
    sample = backend.astype(sample, dtype)
    sample = backend.where(overflow, high, sample)
    sample = backend.clip(sample, low, high)
    # numpy turns 0-d results of elementwise ops into scalars
    return rng, backend.asarray(sample, dtype=dtype, device=device)

def _bounds_traced(low : BArrayType, high : BArrayType) -> bool:
    """Whether the bounds only exist inside a jax trace (the space was built inside ``jax.jit``)."""
    return _is_jax_tracer(low) or _is_jax_tracer(high)

def _bounds_numpy(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    low : BArrayType,
    high : BArrayType,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The stored (abbreviated) bounds as numpy arrays, for building cached plans.

    Returns ``None`` if they cannot be represented in numpy (e.g. torch bfloat16).
    """
    try:
        return np.asarray(backend.to_numpy(low)), np.asarray(backend.to_numpy(high))
    except Exception:
        return None

def _scalar_bounds(
    bounds : Tuple[np.ndarray, np.ndarray],
    dtype_is_float : bool,
) -> Optional[Tuple[Union[int, float], Union[int, float]]]:
    """Python-scalar ``(low, high)`` if every coordinate shares the same bounds, else ``None``."""
    low, high = bounds
    if low.size != 1 or high.size != 1:
        return None
    cast = float if dtype_is_float else int
    return cast(low.reshape(-1)[0]), cast(high.reshape(-1)[0])

# ``random_discrete_uniform`` takes an exclusive upper bound, which jax (without x64)
# parses as int32: ``high + 1`` has to fit.
_RANDINT_LIMIT = 2 ** 31 - 1

def _build_sample_plan(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    dtype_is_float : bool,
    low : BArrayType,
    high : BArrayType,
) -> Tuple[tuple, bool]:
    """How to sample a box with stored bounds ``low``/``high``, and whether the plan may be cached.

    * ``("uniform", high - low)``: every coordinate is bounded (float): one uniform
      draw scaled to the bounds -- the same values the generic path draws for
      bounded coordinates.
    * ``("randint", low, high + 1)``: integer box whose coordinates share scalar
      bounds: one discrete uniform draw.
    * ``("generic",)``: per-coordinate sampling by interval type.
    """
    if _bounds_traced(low, high):
        return ("generic",), False
    bounds = _bounds_numpy(backend, low, high)
    if bounds is None:
        return ("generic",), True
    if dtype_is_float:
        if np.all(bounds[0] > -np.inf) and np.all(bounds[1] < np.inf):
            span = high - low
            # inside a trace the span is a tracer: use it for this call only
            return ("uniform", span), not _is_jax_tracer(span)
        return ("generic",), True
    scalar_bounds = _scalar_bounds(bounds, dtype_is_float)
    if scalar_bounds is not None:
        lo, hi = scalar_bounds
        if -_RANDINT_LIMIT - 1 <= lo and hi + 1 <= _RANDINT_LIMIT:
            return ("randint", lo, hi + 1), True
    return ("generic",), True

def _sample_from_plan(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    plan : tuple,
    rng : BRNGType,
    shape : Sequence[int],
    dtype : BDtypeType,
    device : Optional[BDeviceType],
    low : BArrayType,
) -> Tuple[BRNGType, BArrayType]:
    """Draw a sample of ``shape`` following a non-generic plan from ``_build_sample_plan``."""
    if plan[0] == "uniform":
        rng, sample = backend.random.random_uniform(
            shape, rng=rng, low=0.0, high=1.0, dtype=dtype, device=device
        )
        sample = sample * plan[1] + low
    else:
        rng, sample = backend.random.random_discrete_uniform(
            shape, plan[1], plan[2], rng=rng, dtype=dtype, device=device
        )
    if not backend.is_backendarray(sample):
        # numpy turns 0-d results of elementwise ops into scalars
        sample = backend.asarray(sample, dtype=dtype, device=device)
    return rng, sample

class BoxSpace(Space[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """Continuous or integer hyper-rectangle defined by elementwise bounds.

    A box is immutable after construction (``_low``/``_high`` are never
    reassigned), so values derived from the bounds are computed lazily and
    cached on the instance; caches are dropped when pickling.
    """
    _CACHE_ATTRS = ("_low_b", "_high_b", "_contains_plan", "_sample_plan")
    # Immutable: skip ``Space``'s mutation tracking, which would slow down construction
    __setattr__ = object.__setattr__

    def __init__(
        self,
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        low: SupportsFloat | BArrayType,
        high: SupportsFloat | BArrayType,
        dtype: BDtypeType,
        device : Optional[BDeviceType] = None,
        shape: Optional[Sequence[int]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        """Create a box with broadcastable ``low`` and ``high`` bounds.

        ``metadata`` is an optional JSON-serializable description of the data (see ``Space``).
        """
        assert (
            dtype is not None
        ), "Box dtype must be explicitly provided, cannot be None."
        assert (
            backend.dtype_is_real_floating(dtype) or backend.dtype_is_real_integer(dtype)
        ), f"Box dtype must be a real floating or integer type, actual dtype: {dtype}"
        self._dtype_is_float = backend.dtype_is_real_floating(dtype)
        if dtype is int or dtype is float:
            # Python scalar types (jax's default dtypes) -> the concrete dtype arrays get,
            # so equality, serialization round-trips and data dtype checks agree
            dtype = backend.empty((0,), dtype=dtype).dtype

        # determine shape if it isn't provided directly
        if shape is not None:
            assert all(
                type(dim) is int or np.issubdtype(type(dim), np.integer) for dim in shape
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
            metadata=metadata,
        )

        # Cast low and high to backend arrays
        if backend.dtype_is_real_integer(dtype):
            # Query the materialised dtype: e.g. jax without x64 stores ``int`` as int32
            dttype_iinfo = backend.iinfo(backend.empty((0,), dtype=dtype, device=device).dtype)
            dtype_min = dttype_iinfo.min
            dtype_max = dttype_iinfo.max

            if isinstance(low, (int, float)) and (low == backend.inf or low == -backend.inf):
                low = dtype_min if low == -backend.inf else dtype_max
            if isinstance(high, (int, float)) and (high == backend.inf or high == -backend.inf):
                high = dtype_max if high == backend.inf else dtype_min

        if (
            type(low) in (int, float) and type(high) in (int, float)
            and self._scalar_bounds_ok(backend, low, high, dtype, device)
        ):
            # Fast path: both bounds are Python scalars that passed every check below
            # in Python (see ``_scalar_bounds_ok``), and a ``[1] * ndim`` array is
            # already abbreviated, so the backend-side validation is redundant.
            self._low = backend.full([1] * len(shape), low, dtype=dtype, device=device)
            self._high = backend.full([1] * len(shape), high, dtype=dtype, device=device)
            self._init_caches()
            return

        if isinstance(low, (int, float)):
            _low = backend.full([1] * len(shape), low, dtype=dtype, device=device)
        else:
            _low = backend.astype(low, dtype)
        if isinstance(high, (int, float)):
            _high = backend.full([1] * len(shape), high, dtype=dtype, device=device)
        else:
            _high = backend.astype(high, dtype)
        
        # Empty arrays have nothing to abbreviate (and xbarray cannot abbreviate them)
        if 0 not in tuple(_low.shape):
            _low = backend.abbreviate_array(
                _low,
                try_cast_scalar=False
            )
        if 0 not in tuple(_high.shape):
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
        
        # Size-1 axes broadcast to any size, including 0
        assert len(_low.shape) == len(shape) and all(
            (_low.shape[i] == shape[i] or _low.shape[i] == 1) for i in range(len(shape))
        ), f"_low.shape doesn't match provided shape and is not broadcastable, _low.shape: {_low.shape}, shape: {shape}"
        assert len(_high.shape) == len(shape) and all(
            (_high.shape[i] == shape[i] or _high.shape[i] == 1) for i in range(len(shape))
        ), f"_high.shape doesn't match provided shape and is not broadcastable, _high.shape: {_high.shape}, shape: {shape}"

        self._shape: Tuple[int, ...] = shape
        
        # Check bounded below and above for integer dtype
        if not self._dtype_is_float:
            bounded_below = -self.backend.inf < _low
            bounded_above = self.backend.inf > _high
            below = bool(self.backend.all(bounded_below))
            above = bool(self.backend.all(bounded_above))
            assert below and above, f"Box bounds must be finite for integer dtype, actual low: {_low}, high: {_high}"

        self._low = _low
        self._high = _high
        self._init_caches()

    @staticmethod
    def _scalar_bounds_ok(backend, low, high, dtype, device) -> bool:
        """Validate Python-scalar bounds without touching the backend.

        Returns ``True`` only if the array-based checks in ``__init__`` are
        guaranteed to pass on the bounds produced by ``backend.full``; otherwise
        (NaN, casts that overflow or truncate into ``low > high``, ...) returns
        ``False`` so the generic path runs and raises exactly as before.
        """
        if low != low or high != high:  # NaN
            return False
        if backend.dtype_is_real_integer(dtype):
            if not (math.isfinite(low) and math.isfinite(high)):
                return False
            # Materialised dtype: e.g. jax without x64 stores int64 as int32
            info = backend.iinfo(backend.empty((0,), dtype=dtype, device=device).dtype)
            low_c, high_c = int(low), int(high)  # float -> int truncates like ``full``/``astype``
            return info.min <= low_c <= high_c <= info.max
        # Rounding to a narrower float dtype is monotonic: low <= high is preserved
        return low <= high

    @classmethod
    def _from_validated(
        cls,
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        low : BArrayType,
        high : BArrayType,
        dtype : BDtypeType,
        shape : Sequence[int],
        device : Optional[BDeviceType] = None,
        metadata : Optional[Mapping[str, Any]] = None,
    ) -> "BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        """Internal constructor that skips validation.

        Only for bounds derived from an already validated ``BoxSpace`` (e.g. by
        batching / unbatching), i.e. ``low``/``high`` already have ``dtype``,
        live on ``device``, have ``len(shape)`` dims that are either 1 or equal
        to ``shape`` and satisfy ``low <= high``.
        """
        self = cls.__new__(cls)
        Space.__init__(self, backend=backend, shape=shape, device=device, dtype=dtype, metadata=metadata)
        self._dtype_is_float = backend.dtype_is_real_floating(dtype)
        self._low = low
        self._high = high
        self._init_caches()
        return self

    def _init_caches(self) -> None:
        for name in self._CACHE_ATTRS:
            setattr(self, name, None)

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._CACHE_ATTRS:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_caches()

    @property
    def low(self) -> BArrayType:
        low = self._low_b
        if low is None:
            low = self.backend.broadcast_to(self._low, self.shape)
            if not _is_jax_tracer(low):
                self._low_b = low
        return low
    
    @property
    def high(self) -> BArrayType:
        high = self._high_b
        if high is None:
            high = self.backend.broadcast_to(self._high, self.shape)
            if not _is_jax_tracer(high):
                self._high_b = high
        return high

    def to(
        self, 
        backend : Optional[ComputeBackend] = None,
        device : Optional[Union[BDeviceType, Any]] = None,
    ) -> Union["BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType]", "BoxSpace"]:
        """Return an equivalent box on another backend and/or device."""
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
            device=new_device,
            metadata=self._metadata,
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

        plan = self._sample_plan
        if plan is None:
            plan, cacheable = _build_sample_plan(self.backend, self._dtype_is_float, self._low, self._high)
            if cacheable:
                self._sample_plan = plan
        if plan[0] != "generic":
            return _sample_from_plan(self.backend, plan, rng, self.shape, self.dtype, self.device, self._low)

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
            rng, sample = _sample_integers_in_closed_interval(
                self.backend, rng, low, self.high, self.shape, self.dtype, device=self.device
            )
        return rng, sample

    def create_empty(self) -> BArrayType:
        """Allocate an uninitialized array with the box shape and dtype."""
        return self.backend.empty(
            self.shape,
            dtype=self.dtype,
            device=self.device
        )

    def _build_contains_plan(self) -> Optional[tuple]:
        """How ``contains`` checks the values (``None``: generic check, don't cache).

        * ``("minmax", low, high, full_range)``: all coordinates share the Python-scalar
          bounds ``low``/``high``; ``full_range`` marks an integer box spanning its
          whole dtype.
        * ``("generic",)``: elementwise comparison against the bounds.
        """
        if _bounds_traced(self._low, self._high):
            return None
        bounds = _bounds_numpy(self.backend, self._low, self._high)
        if bounds is None:
            return ("generic",)
        scalar_bounds = _scalar_bounds(bounds, self._dtype_is_float)
        if scalar_bounds is None:
            return ("generic",)
        low, high = scalar_bounds
        full_range = False
        if not self._dtype_is_float:
            info = np.iinfo(bounds[0].dtype)
            full_range = low == int(info.min) and high == int(info.max)
        return ("minmax", low, high, full_range)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not self.backend.is_backendarray(x):
            return False
        if not (self.backend.can_cast(x, self.dtype) and tuple(x.shape) == self.shape):
            return False

        plan = self._contains_plan
        if plan is None:
            plan = self._build_contains_plan()
            if plan is not None:
                self._contains_plan = plan
        if plan is not None and plan[0] == "minmax":
            _, low, high, full_range = plan
            if full_range and x.dtype == self._low.dtype:
                # Every value of the (materialised) box dtype lies inside a full-range box
                return True
            if 0 in self.shape:
                return True
            # One reduction each instead of two elementwise compares + all(); the
            # extremes are compared as exact Python scalars, and NaN fails both.
            cast = float if self._dtype_is_float else int
            return cast(self.backend.min(x)) >= low and cast(self.backend.max(x)) <= high
        return bool(
            self.backend.all(x >= self._low)
            and self.backend.all(x <= self._high)
        )
    
    def clip(self, x: BArrayType) -> BArrayType:
        """Clamp ``x`` elementwise to the bounds of this box (see ``Space.clip``).

        ``x`` is cast to the box dtype first (members of a superset box are
        losslessly castable), so the result always has the box dtype; ``±inf``
        bounds leave the corresponding coordinates untouched. Never raises
        for data of the box shape. Returns a new array.
        """
        backend = self.backend
        dtype = self._low.dtype  # materialised dtype (jax without x64 stores int64 as int32)
        if not backend.is_backendarray(x):
            x = backend.asarray(x, dtype=dtype, device=self.device)
        elif x.dtype != dtype:
            x = backend.astype(x, dtype)

        plan = self._contains_plan
        if plan is None:
            plan = self._build_contains_plan()
            if plan is not None:
                self._contains_plan = plan
        if plan is not None and plan[0] == "minmax":
            # Shared scalar bounds: the backends' scalar clip is much faster than array bounds
            ret = backend.clip(x, plan[1], plan[2])
        else:
            ret = backend.clip(x, self._low, self._high)
        if not backend.is_backendarray(ret):
            # numpy turns 0-d results of elementwise ops into scalars
            ret = backend.asarray(ret, dtype=dtype, device=self.device)
        return ret

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

    def is_subspaceeq(self, other: Any) -> bool:
        """Return whether this box is a non-strict subspace of ``other`` (⊆).

        True iff ``other`` is a ``BoxSpace`` on the same backend, with equal
        shape and dtype, and ``other``'s bounds contain ``self``'s bounds
        elementwise (``other.low <= self.low`` and ``other.high >= self.high``).
        Infinite bounds are handled correctly via direct comparison since
        ``±inf`` compares as expected against finite values and itself.
        ``device`` is ignored.
        """
        try:
            if not (isinstance(other, BoxSpace) and self.backend == other.backend):
                return False
            if self.shape != other.shape or self.dtype != other.dtype:
                return False
            return bool(
                self.backend.all(self.backend.broadcast_to(other.low, self.shape) <= self.low)
                and self.backend.all(self.backend.broadcast_to(other.high, self.shape) >= self.high)
            )
        except Exception:
            return False

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
