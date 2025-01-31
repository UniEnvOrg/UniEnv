"""Implementation of a space consisting of finitely many elements."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal
import numpy as np
from .space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
import gymnasium as gym

class Discrete(Space[BArrayType, np.ndarray, BDeviceType, BDtypeType, BRNGType]):
    r"""A space consisting of finitely many elements.

    This class represents a finite subset of integers, more specifically a set of the form :math:`\{ a, a+1, \dots, a+n-1 \}`.

    Example:
        >>> from gymnasium.spaces import Discrete
        >>> observation_space = Discrete(2, seed=42) # {0, 1}
        >>> observation_space.sample()
        np.int64(0)
        >>> observation_space = Discrete(3, start=-1, seed=42)  # {-1, 0, 1}
        >>> observation_space.sample()
        np.int64(-1)
    """

    def __init__(
        self,
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        n: int,
        start: int = 0,
        device : Optional[BDeviceType] = None,
        dtype: Optional[BDtypeType] = None,
    ):
        assert isinstance(n, int), f"Expects `n` to be an integer, actual dtype: {type(n)}"
        assert n > 0, "n (counts) have to be positive"
        assert isinstance(start, int), f"Expects `start` to be an integer, actual type: {type(start)}"
        assert dtype is None or backend.dtype_is_real_integer(dtype), f"Expects dtype to be an integer, actual dtype: {dtype}"

        self.n = int(n)
        self.start = int(start)
        super().__init__(
            backend=backend,
            shape=(1,),
            device=device,
            dtype=dtype,
        )

    def to_device(self, device: BDeviceType) -> "Discrete[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        return Discrete(
            backend=self.backend,
            n=self.n,
            start=self.start,
            device=device,
            dtype=self.dtype
        )
    
    def to_backend(self, backend: ComputeBackend, device : Optional[Any]) -> "Discrete":
        return Discrete(
            backend=backend,
            n=self.n,
            start=self.start,
            device=device
        )

    @property
    def is_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    @property
    def flat_dim(self) -> int:
        return 1
    
    def flatten(self, data : BArrayType) -> BArrayType:
        """Flatten the data."""
        return data
    
    def unflatten(self, data : BArrayType) -> BArrayType:
        """Unflatten the data."""
        return data

    def sample(self, rng : BRNGType) -> Tuple[BRNGType, BArrayType]:
        """Generates a single random sample from this space.

        A sample will be chosen uniformly at random with the mask if provided

        Args:
            mask: An optional mask for if an action can be selected.
                Expected `np.ndarray` of shape ``(n,)`` and dtype ``np.int8`` where ``1`` represents valid actions and ``0`` invalid / infeasible actions.
                If there are no possible actions (i.e. ``np.all(mask == 0)``) then ``space.start`` will be returned.

        Returns:
            A sampled integer from the space
        """
        rng, sample = self.backend.random_discrete_uniform(
            rng,
            shape=(1,),
            from_num=self.start,
            to_num=self.start + self.n,
            dtype=self.dtype,
            device=self.device,
        )
        sample = sample + self.start
        return rng, sample

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if self.backend.is_backendarray(x) and x.shape == (1,):
            as_int = int(x[0])
        else:
            return False

        return bool(self.start <= as_int < self.start + self.n)

    def get_repr(
        self, 
        include_backend = True, 
        include_device = True, 
        include_dtype = True
    ):
        ret = f"Discrete({self.n}"
        if self.start != 0:
            ret += f", start={self.start}"
        
        if include_backend:
            ret += f", backend={self.backend}"
        if include_device:
            ret += f", device={self.device}"
        if include_dtype:
            ret += f", dtype={self.dtype}"
        ret += ")"
        return ret

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Discrete)
            and self.n == other.n
            and self.start == other.start
        )

    def to_jsonable(self, sample_n: Sequence[BArrayType]) -> list[int]:
        """Converts a list of samples to a list of ints."""
        return [int(x[0]) for x in sample_n]

    def from_jsonable(self, sample_n: list[int]) -> list[BArrayType]:
        """Converts a list of json samples to a list of np.int64."""
        return [self.backend.array_api_namespace.asarray([x], dtype=self.dtype, device=self.device) for x in sample_n]

    def from_gym_data(self, gym_data : np.int64) -> BArrayType:
        """Convert a gym space to this space."""
        return self.backend.array_api_namespace.asarray([gym_data], dtype=self.dtype, device=self.device)

    def from_other_backend(self, other_data : Any, backend : ComputeBackend) -> BArrayType:
        new_tensor = self.backend.from_other_backend(other_data, backend)
        return self.from_same_backend(new_tensor)

    def from_same_backend(self, other_data : BArrayType) -> BArrayType:
        new_tensor = other_data
        
        if self.device is not None:
            new_tensor = self.backend.to_device(new_tensor, self.device)
        if self.dtype is not None:
            new_tensor = self.backend.array_api_namespace.astype(new_tensor, self.dtype)
        return new_tensor

    def to_gym_data(self, data : BArrayType) -> np.int64:
        """Convert this space to a gym space."""
        return np.int64(data[0])
    
    def to_gym_space(self) -> gym.Space:
        """Convert this space to a gym space."""
        return gym.spaces.Discrete(
            self.n,
            start=self.start
        )
