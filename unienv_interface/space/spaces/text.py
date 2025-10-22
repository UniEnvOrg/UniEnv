"""Implementation of a space that represents the cartesian product of `Discrete` spaces."""
"""Implementation of a space consisting of finitely many elements."""
from typing import Any, Generic, Iterable, FrozenSet, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal, List, Dict
import numpy as np
from ..space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils import seed_util
import string

alphanumeric: FrozenSet[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


class TextSpace(Space[str, BDeviceType, BDtypeType, BRNGType]):
    def __init__(
        self,
        backend : ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType],
        max_length: int,
        *,
        min_length: int = 0,
        charset: Optional[FrozenSet[str] | str] = None,
        device : Optional[BDeviceType] = None,
    ):
        assert min_length >= 0 and min_length <= max_length
        self.min_length: int = min_length
        self.max_length: int = max_length
        
        self._char_set: Optional[FrozenSet[str]] = frozenset(charset) if charset is not None else None
        self._char_list: Optional[Tuple[str, ...]] = tuple(sorted(tuple(charset))) if charset is not None else None
        self._char_index: Optional[Dict[str, int]] = {
            val: i for i, val in enumerate(self._char_list)
        } if self._char_list is not None else None
        super().__init__(
            backend,
            shape=None,
            device=device,
            dtype=None
        )

    def to(self, backend = None, device = None):
        if (backend is None or backend==self.backend) and (device is None or device==self.device):
            return self
        new_device = device if backend is not None else (device or self.device)
        return TextSpace(
            backend or self.backend,
            max_length=self.max_length,
            min_length=self.min_length,
            charset=self.charset,
            device=new_device
        )

    @property
    def charset(self) -> Optional[FrozenSet[str]]:
        return self._char_set
    
    @property
    def charset_index(self) -> Optional[Mapping[str, int]]:
        return self._char_index
    
    @property
    def charset_list(self) -> Optional[Tuple[str, ...]]:
        return self._char_list

    def character_index(self, char: str) -> Optional[int]:
        return self._char_index[char] if self._char_index is not None and char in self._char_index else None

    def sample(
        self,
        rng: BRNGType,
    ) -> Tuple[BRNGType, str]:
        charset_list = self._char_list or tuple(sorted(alphanumeric))
        rng, length = self.backend.random.random_discrete_uniform(
            1,
            self.min_length,
            self.max_length + 1,
            rng=rng,
            dtype=self.backend.default_integer_dtype,
        )
        length = int(length[0])
        index = self.backend.random.random_permutation(
            length,
            rng=rng,
        )
        sample = "".join(
            charset_list[index[i]] for i in range(length)
        ) if length > 0 else ""
        return rng, sample

    def create_empty(self) -> str:
        return ""

    def is_bounded(self, manner = "both"):
        return manner == "below" or (
            self.charset is not None
        )

    def contains(self, x: Any) -> bool:
        if not isinstance(x, str):
            return False
        return (
            (self.min_length <= len(x) <= self.max_length)
            and (
                self.charset is None 
                or all(c in self.charset for c in x)                         
            )
        )

    def get_repr(
        self, 
        abbreviate = False,
        include_backend = True, 
        include_device = True, 
        include_dtype = True
    ):
        return f"TextSpace({self.min_length}, {self.max_length}, charset={self.charset!r})"

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, TextSpace)
            and self.min_length == other.min_length
            and self.max_length == other.max_length
            and self.charset == other.charset
        )
    
    def data_to(self, data, backend = None, device = None):
        return data