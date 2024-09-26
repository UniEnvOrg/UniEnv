"""Implementation of a space consisting of finitely many elements."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal
import numpy as np
from .space import Space, register_space_to_gym_mapping
from unienv_interface.backends.base import ComputeBackend
import array_api_compat
import gymnasium as gym

DiscreteArrayT = TypeVar("BoxArrayT", covariant=True)
_DiscreteBDeviceT = TypeVar("_BoxBDeviceT", covariant=True)
_DiscreteBDTypeT = TypeVar("_BoxBDTypeT", covariant=True)
_DiscreteBDRNGT = TypeVar("_BoxBDRNGT", covariant=True)

@register_space_to_gym_mapping(gym.spaces.Discrete)
class Discrete(Space[DiscreteArrayT, _DiscreteBDeviceT, _DiscreteBDTypeT, _DiscreteBDRNGT]):
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
        backend : Type[ComputeBackend[DiscreteArrayT, Any, _DiscreteBDeviceT, _DiscreteBDTypeT, _DiscreteBDRNGT]],
        n: int,
        start: int = 0,
        device : Optional[_DiscreteBDeviceT] = None,
        dtype: Optional[_DiscreteBDTypeT] = None,
        seed: Optional[int] = None,
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
            seed=seed,
        )

    def to_device(self, device: _DiscreteBDeviceT) -> "Discrete[DiscreteArrayT, _DiscreteBDeviceT, _DiscreteBDTypeT, _DiscreteBDRNGT]":
        return Discrete(
            backend=self.backend,
            n=self.n,
            start=self.start,
            device=device,
            dtype=self.dtype
        )
    
    def to_backend(self, backend: Type[ComputeBackend], device : Optional[Any]) -> "Discrete":
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

    def sample(self) -> DiscreteArrayT:
        """Generates a single random sample from this space.

        A sample will be chosen uniformly at random with the mask if provided

        Args:
            mask: An optional mask for if an action can be selected.
                Expected `np.ndarray` of shape ``(n,)`` and dtype ``np.int8`` where ``1`` represents valid actions and ``0`` invalid / infeasible actions.
                If there are no possible actions (i.e. ``np.all(mask == 0)``) then ``space.start`` will be returned.

        Returns:
            A sampled integer from the space
        """
        
        return self.start + self.backend.random_discrete_uniform(
            self.rng,
            shape=(1,),
            from_num=self.start,
            to_num=self.start + self.n,
            dtype=self.dtype,
            device=self.device,
        )

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if self.backend.is_backendarray(x) and x.shape == (1,):
            as_int = int(x[0])
        else:
            return False

        return bool(self.start <= as_int < self.start + self.n)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        if self.start != 0:
            return f"Discrete({self.backend.__name__}, {self.n}, start={self.start}, {self.dtype}, {self.device})"
        return f"Discrete({self.backend.__name__}, {self.n}, {self.dtype}, {self.device})"

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Discrete)
            and self.n == other.n
            and self.start == other.start
        )

    def __setstate__(self, state: Iterable[tuple[str, Any]] | Mapping[str, Any]):
        """Used when loading a pickled space.

        This method has to be implemented explicitly to allow for loading of legacy states.

        Args:
            state: The new state
        """
        # Don't mutate the original state
        state = dict(state)

        # Allow for loading of legacy states.
        # See https://github.com/openai/gym/pull/2470
        if "start" not in state:
            state["start"] = np.int64(0)

        super().__setstate__(state)

    def to_jsonable(self, sample_n: Sequence[DiscreteArrayT]) -> list[int]:
        """Converts a list of samples to a list of ints."""
        return [int(x[0]) for x in sample_n]

    def from_jsonable(self, sample_n: list[int]) -> list[DiscreteArrayT]:
        """Converts a list of json samples to a list of np.int64."""
        return [self.backend.array_api_namespace.asarray([x], dtype=self.dtype, device=self.device) for x in sample_n]

    def from_gym_data(self, gym_data : np.int64) -> DiscreteArrayT:
        """Convert a gym space to this space."""
        return self.backend.array_api_namespace.asarray([gym_data], dtype=self.dtype, device=self.device)

    def from_other_backend(self, other_data : Any) -> DiscreteArrayT:
        new_tensor = self.backend.from_dlpack(other_data)
        return self.from_same_backend(new_tensor)

    def from_same_backend(self, other_data : DiscreteArrayT) -> DiscreteArrayT:
        new_tensor = other_data
        if self.dtype is not None:
            new_tensor = self.backend.array_api_namespace.astype(new_tensor, dtype=self.dtype, device=self.device)
        elif self.device is not None:
            new_tensor = array_api_compat.to_device(new_tensor, device=self.device)
        return new_tensor

    def to_gym_data(self, data : DiscreteArrayT) -> np.int64:
        """Convert this space to a gym space."""
        return np.int64(data[0])
    
    def to_gym_space(self) -> gym.Space:
        """Convert this space to a gym space."""
        return gym.spaces.Discrete(
            self.n,
            start=self.start
        )
    
    @staticmethod
    def from_gym_space(
        gym_space : gym.spaces.Discrete,
        backend : Type[ComputeBackend[DiscreteArrayT, Any, _DiscreteBDeviceT, _DiscreteBDTypeT, _DiscreteBDRNGT]],
        dtype : Optional[_DiscreteBDTypeT] = None,
        device : Optional[_DiscreteBDeviceT] = None,
    ) -> "Discrete[DiscreteArrayT, _DiscreteBDeviceT, _DiscreteBDTypeT, _DiscreteBDRNGT]":
        assert isinstance(gym_space, gym.spaces.Discrete), f"Expected gym_space to be of type gym.spaces.Discrete, got {type(gym_space)}"
        return Discrete(
            backend=backend,
            n=int(gym_space.n),
            start=int(gym_space.start),
            dtype=dtype,
            device=device
        )