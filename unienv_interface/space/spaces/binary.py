"""Implementation of a space consisting of finitely many elements."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal
import numpy as np
from ..space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

class BinarySpace(Space[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    def __init__(
        self, 
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        shape: Sequence[int],
        dtype: Optional[BDtypeType] = None,
        device : Optional[BDeviceType] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        assert dtype is None or backend.dtype_is_boolean(dtype), f"Invalid dtype {dtype}"
        
        assert all(
            np.issubdtype(type(dim), np.integer) for dim in shape
        ), f"Expect all shape elements to be an integer, actual type: {tuple(type(dim) for dim in shape)}"
        shape = tuple(int(dim) for dim in shape)  # This changes any np types to int
        
        super().__init__(
            backend=backend,
            shape=shape,
            device=device,
            dtype=dtype or backend.default_boolean_dtype,
            metadata=metadata,
        )

    def to(self, backend = None, device = None):
        if (backend is None or backend==self.backend) and (device is None or device==self.device):
            return self
        
        if backend is not None and backend != self.backend:
            new_dtype = backend.default_boolean_dtype
            new_device = device
        else:
            new_dtype = self.dtype
            new_device = self.device if device is None else device
        
        return BinarySpace(
            backend or self.backend,
            self.shape,
            dtype=new_dtype,
            device=new_device,
            metadata=self._metadata,
        )
    
    def sample(self, rng : BRNGType) -> Tuple[
        BRNGType, BArrayType
    ]:
        rng, uniform_sample = self.backend.random.random_uniform(
            self.shape, rng=rng, device=self.device
        )
        # ``asarray``: numpy turns the 0-d comparison result into a scalar
        return rng, self.backend.asarray(
            uniform_sample > 0.5,
            dtype=self.dtype,
            device=self.device
        )
    
    def create_empty(self):
        return self.backend.empty(
            self.shape,
            dtype=self.dtype,
            device=self.device
        )

    def is_bounded(self, manner = "both"):
        return True

    def contains(self, x: BArrayType) -> bool:
        return bool(
            self.backend.is_backendarray(x)
            and self.backend.dtype_is_boolean(x.dtype)
            and self.shape == x.shape
        )

    def clip(self, x: BArrayType) -> BArrayType:
        """Return ``x`` (cast to this space's boolean dtype if needed); see ``Space.clip``.

        A binary space is only a subspace of an identical binary space, so any
        member of a superset is already a member: there is nothing to project.
        Raises ``ValueError`` if ``x`` does not have this space's shape.
        """
        if not self.backend.is_backendarray(x):
            x = self.backend.asarray(x, dtype=self.dtype, device=self.device)
        if tuple(x.shape) != self.shape:
            raise ValueError(f"BinarySpace.clip: expected data of shape {self.shape}, got {tuple(x.shape)}")
        if x.dtype != self.dtype:
            x = self.backend.astype(x, self.dtype)
        return x

    def get_repr(
        self, 
        abbreviate = False,
        include_backend = True, 
        include_device = True, 
        include_dtype = True
    ):
        ret = f"BinarySpace({self.shape}"
        if include_backend:
            ret += f", {self.backend}"
        if include_device:
            ret += f", {self.device}"
        if include_dtype:
            ret += f", {self.dtype}"
        ret += ")"
        return ret

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return isinstance(other, BinarySpace) and self.backend == other.backend and self.shape == other.shape and self.dtype == other.dtype and self.device == other.device

    def is_subspaceeq(self, other: Any) -> bool:
        """Return whether this binary space is a non-strict subspace of ``other`` (⊆).

        True iff ``other`` is a ``BinarySpace`` on the same backend with equal
        shape and dtype. Since a binary space contains exactly the boolean
        tensors of its shape, two such spaces are in a subspace relation iff
        they are structurally identical (modulo ``device``).
        """
        return (
            isinstance(other, BinarySpace)
            and self.backend == other.backend
            and self.shape == other.shape
            and self.dtype == other.dtype
        )
    
    def data_to(self, data, backend = None, device = None):
        if backend is not None and backend != self.backend:
            data = backend.from_other_backend(self.backend, data)
        if device is not None:
            data = (backend or self.backend).to_device(data,device)
        return data
