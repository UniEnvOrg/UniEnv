"""Implementation of a space that represents the cartesian product of `Discrete` spaces."""
"""Implementation of a space consisting of finitely many elements."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal
import numpy as np
from .space import Space, register_space_to_gym_mapping
from unienv_interface.backends.base import ComputeBackend
import array_api_compat
import gymnasium as gym
from .discrete import Discrete

MultiDArrayT = TypeVar("MultiBArrayT", covariant=True)
_MultiDDeviceT = TypeVar("_MultiBDeviceT", covariant=True)
_MultiDDTypeT = TypeVar("_MultiBDTypeT", covariant=True)
_MultiDDRNGT = TypeVar("_MultiBDRNGT", covariant=True)

@register_space_to_gym_mapping(gym.spaces.MultiDiscrete)
class MultiDiscrete(Space[MultiDArrayT, np.ndarray, _MultiDDeviceT, _MultiDDTypeT, _MultiDDRNGT]):
    """This represents the cartesian product of arbitrary :class:`Discrete` spaces.

    It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space.

    Note:
        Some environment wrappers assume a value of 0 always represents the NOOP action.

    e.g. Nintendo Game Controller - Can be conceptualized as 3 discrete action spaces:

    1. Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
    2. Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    3. Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1

    It can be initialized as ``MultiDiscrete([ 5, 2, 2 ])`` such that a sample might be ``array([3, 1, 0])``.

    Although this feature is rarely used, :class:`MultiDiscrete` spaces may also have several axes
    if ``nvec`` has several axes:

    Example:
        >>> from gymnasium.spaces import MultiDiscrete
        >>> import numpy as np
        >>> observation_space = MultiDiscrete(np.array([[1, 2], [3, 4]]), seed=42)
        >>> observation_space.sample()
        array([[0, 0],
               [2, 2]])
    """

    def __init__(
        self,
        backend: Type[ComputeBackend[MultiDArrayT, Any, _MultiDDeviceT, _MultiDDTypeT, _MultiDDRNGT]],
        nvec: MultiDArrayT,
        start: Optional[MultiDArrayT] = None,
        dtype: Optional[_MultiDDTypeT] = None,
        device : Optional[_MultiDDeviceT] = None,
        seed: Optional[int] = None,
    ):
        """Constructor of :class:`MultiDiscrete` space.

        The argument ``nvec`` will determine the number of values each categorical variable can take. If
        ``start`` is provided, it will define the minimal values corresponding to each categorical variable.

        Args:
            nvec: vector of counts of each categorical variable. This will usually be a list of integers. However,
                you may also pass a more complicated numpy array if you'd like the space to have several axes.
            dtype: This should be some kind of integer type.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
            start: Optionally, the starting value the element of each class will take (defaults to 0).
        """
        if dtype is not None:
            assert backend.dtype_is_real_integer(dtype), f"Invalid dtype {dtype}"
            self.nvec = backend.array_api_namespace.astype(nvec, dtype, device=device)
            self.start = backend.array_api_namespace.astype(start, dtype, device=device) if start is not None else None
        elif device is not None:
            self.nvec = array_api_compat.to_device(nvec, device)
            self.start = array_api_compat.to_device(start, device) if start is not None else None
        else:
            self.nvec = nvec
        
        if self.start is None:
            self.start = backend.array_api_namespace.zeros_like(self.nvec, dtype=dtype, device=device)
        
        assert self.start.shape == self.nvec.shape, "start and nvec (counts) should have the same shape"
        assert backend.array_api_namespace.all(self.nvec > 0), "nvec (counts) have to be positive"

        super().__init__(
            backend=backend,
            shape=self.nvec.shape,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Has stricter type than :class:`gym.Space` - never None."""
        return self._shape  # type: ignore

    def to_device(self, device: _MultiDDeviceT) -> "MultiDiscrete[MultiDArrayT, _MultiDDeviceT, _MultiDDTypeT, _MultiDDRNGT]":
        return MultiDiscrete(
            backend=self.backend,
            nvec=self.nvec,
            start=self.start,
            dtype=self.dtype,
            device=device
        )

    def to_backend(self, backend: ComputeBackend, device: Optional[Any]) -> "MultiDiscrete":
        return MultiDiscrete(
            backend=backend,
            nvec=backend.from_dlpack(self.nvec),
            start=backend.from_dlpack(self.start),
            device=device
        )

    @property
    def is_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def sample(
        self
    ) -> MultiDArrayT:
        uniform_sample = self.backend.random_uniform(self.rng, self.shape, device=self.device)
        scaled_sample = uniform_sample * self.nvec + self.start
        floored_sample = self.backend.array_api_namespace.floor(scaled_sample)
        if self.dtype is not None:
            return self.backend.array_api_namespace.astype(floored_sample, self.dtype, device=self.device)
        else:
            return self.backend.array_api_namespace.astype(floored_sample, self.nvec.dtype, device=self.device)

    def contains(self, x: MultiDArrayT) -> bool:
        return bool(
            self.backend.is_backendarray(x)
            and x.shape == self.shape
            and (x.dtype == self.dtype or self.dtype is None)
            and self.backend.array_api_namespace.all(self.start <= x)
            and self.backend.array_api_namespace.all(x - self.start < self.nvec)
        )

    def __repr__(self):
        """Gives a string representation of this space."""
        if np.any(self.start != 0):
            return f"MultiDiscrete({self.nvec}, start={self.start})"
        return f"MultiDiscrete({self.nvec})"

    def __getitem__(self, index: int | tuple[int, ...]):
        """Extract a subspace from this ``MultiDiscrete`` space."""
        nvec = self.nvec[index]
        start = self.start[index]
        if len(nvec.shape) == 0:
            subspace = Discrete(
                self.backend,
                n=int(nvec),
                start=int(start),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            subspace = MultiDiscrete(
                self.backend,
                nvec=nvec,
                start=start,
                dtype=self.dtype,
                device=self.device,
            )
        
        return subspace

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return bool(
            isinstance(other, MultiDiscrete)
            and self.dtype == other.dtype
            and self.shape == other.shape
            and np.all(self.nvec == other.nvec)
            and np.all(self.start == other.start)
        )

    def to_jsonable(
        self, sample_n: Sequence[MultiDArrayT]
    ) -> list[Sequence[int]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [self.backend.to_numpy(sample).tolist() for sample in sample_n]

    def from_jsonable(
        self, sample_n: list[Sequence[int]]
    ) -> list[MultiDArrayT]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [self.backend.from_numpy(np.asarray(sample, dtype=np.int64), dtype=self.dtype, device=self.device) for sample in sample_n]

    def from_gym_data(self, gym_data : np.ndarray) -> MultiDArrayT:
        return self.backend.from_numpy(gym_data, dtype=self.dtype, device=self.device)
    
    def to_gym_data(self, data : MultiDArrayT) -> np.ndarray:
        return self.backend.to_numpy(data).astype(int)
    
    def from_other_backend(self, other_data : Any) -> MultiDArrayT:
        new_tensor = self.backend.from_dlpack(other_data)
        return self.from_same_backend(new_tensor)
    
    def from_same_backend(self, other_data : MultiDArrayT) -> MultiDArrayT:
        new_tensor = other_data
        if self.dtype is not None:
            new_tensor = self.backend.array_api_namespace.astype(new_tensor, dtype=self.dtype, device=self.device)
        elif self.device is not None:
            new_tensor = array_api_compat.to_device(new_tensor, device=self.device)
        
        return new_tensor

    def to_gym_space(self) -> gym.Space:
        return gym.spaces.MultiDiscrete(self.backend.to_numpy(self.nvec), start=self.backend.to_numpy(self.start))
    
    @staticmethod
    def from_gym_space(
        gym_space : gym.spaces.MultiDiscrete,
        backend : Type[ComputeBackend[MultiDArrayT, Any, _MultiDDeviceT, _MultiDDTypeT, _MultiDDRNGT]],
        dtype : Optional[_MultiDDTypeT] = None,
        device : Optional[_MultiDDeviceT] = None,
    ) -> "MultiDiscrete[MultiDArrayT, _MultiDDeviceT, _MultiDDTypeT, _MultiDDRNGT]":
        nvec = backend.from_numpy(gym_space.nvec, dtype=dtype, device=device)
        start = backend.from_numpy(gym_space.start, dtype=dtype, device=device)
        return MultiDiscrete(
            backend=backend,
            nvec=nvec,
            start=start,
            dtype=dtype,
            device=device
        )