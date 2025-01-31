"""Implementation of a space that represents the cartesian product of `Discrete` spaces."""
"""Implementation of a space consisting of finitely many elements."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal, List
import numpy as np
from .space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils import seed_util
import gymnasium as gym
from .discrete import Discrete

alphanumeric: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

class Text(Space[str, str, BDeviceType, BDtypeType, BRNGType]):
    def __init__(
        self,
        backend : ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType],
        max_length: int,
        *,
        min_length: int = 1,
        charset: frozenset[str] | str = alphanumeric,
        device : Optional[BDeviceType] = None,
    ):
        self._gym_space = gym.spaces.Text(max_length=max_length, min_length=min_length, charset=charset)
        super().__init__(
            backend=backend,
            shape=None,
            dtype=str,
            device=device,
        )

    def to_device(self, device : BDeviceType) -> "Text[BDeviceType, BDtypeType, BRNGType]":
        return self

    def to_backend(self, backend : ComputeBackend, device : Optional[Any]) -> "Text":
        return Text(
            backend=backend,
            max_length=self.max_length,
            min_length=self.min_length,
            charset=self.character_set,
            device=device
        )

    @property
    def min_length(self) -> int:
        return self._gym_space.min_length

    @min_length.setter
    def min_length(self, value: int) -> None:
        self._gym_space.min_length = value
    
    @property
    def max_length(self) -> int:
        return self._gym_space.max_length
    
    @max_length.setter
    def max_length(self, value: int) -> None:
        self._gym_space.max_length = value
    
    @property
    def character_set(self) -> frozenset[str]:
        """Returns the character set for the space."""
        return self._gym_space.character_set

    @property
    def character_list(self) -> tuple[str, ...]:
        """Returns a tuple of characters in the space."""
        return self._gym_space.character_list

    def character_index(self, char: str) -> np.int32:
        """Returns a unique index for each character in the space's character set."""
        return self._gym_space.character_index(char)

    @property
    def characters(self) -> str:
        """Returns a string with all Text characters."""
        return self._gym_space.characters

    @property
    def is_flattenable(self) -> bool:
        """The flattened version is an integer array for each character, padded to the max character length."""
        return True

    @property
    def flat_dim(self) -> int:
        return self.max_length
    
    def flatten(self, data : str) -> Any:
        dat = self.backend.array_api_namespace.full(
            (self.max_length,),
            len(self.character_set),
            device=self.device,
        )
        for i, char in enumerate(data):
            dat = self.backend.replace_inplace(dat, i, self.character_index(char))
        return dat

    def unflatten(self, data : Any) -> str:
        return "".join([
            self.character_list[i] for i in data if i < len(self.character_list)
        ])

    def sample(
        self,
        rng: BRNGType,
    ) -> Tuple[BRNGType, str]:
        rng, next_seed = seed_util.next_seed_rng(rng, self.backend)
        self._gym_space.seed(next_seed)
        return rng, self._gym_space.sample()

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return self._gym_space.contains(x)

    def get_repr(
        self, 
        include_backend = True, 
        include_device = True, 
        include_dtype = True
    ):
        """Get a string representation of this space."""
        return f"Text({self._gym_space.min_length}, {self._gym_space.max_length}, charset={self.characters})"

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Text)
            and self.min_length == other.min_length
            and self.max_length == other.max_length
            and self.character_set == other.character_set
        )
    
    def to_jsonable(self, sample_n: Sequence[str]) -> List[str]:
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n: List[str]) -> List[str]:
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return list(sample_n)
    
    def from_gym_data(self, gym_data : str) -> str:
        """Convert a gym space to this space."""
        return gym_data
    
    def to_gym_data(self, data : str) -> str:
        """Convert this space to a gym space."""
        return data
    
    def from_other_backend(self, other_data : str, backend : ComputeBackend) -> str:
        """Convert data from another backend to this backend."""
        return other_data
    
    def from_same_backend(self, other_data : str) -> str:
        """Convert data from another device to this device."""
        return other_data

    def to_gym_space(self) -> gym.spaces.Text:
        return self._gym_space
