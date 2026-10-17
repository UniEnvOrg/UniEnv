"""Implementation of a space consisting of finitely many elements."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal, Union, Callable
import numpy as np
from ..space import Space, SpaceDataT
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

class BatchedSpace(Space[np.ndarray, BDeviceType, BDtypeType, BRNGType]):
    """
    This space represents a batch of 
    """
    def __init__(
        self, 
        single_space : Space[SpaceDataT, BDeviceType, BDtypeType, BRNGType],
        batch_shape: Sequence[int],
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        assert len(batch_shape) > 0, "Batch shape must be a non-empty sequence"
        batch_shape = tuple(int(dim) for dim in batch_shape)  # This changes any np types to int
        super().__init__(
            backend=single_space.backend,
            shape=batch_shape if single_space.shape is None else batch_shape + single_space.shape,
            device=single_space.device,
            dtype=single_space.dtype,
            metadata=metadata,
        )

        self.batch_shape = batch_shape
        self.single_space = single_space

    def _make_object_array(self, items : Sequence[Any]) -> np.ndarray:
        # Fill element-wise so that sequence-like items (arrays, tuples) are kept
        # as single objects instead of being expanded into extra dimensions.
        ret = np.empty(self.batch_shape, dtype=object)
        for i, item in enumerate(items):
            ret.flat[i] = item
        return ret

    def to(self, backend = None, device = None):
        return BatchedSpace(
            self.single_space.to(backend=backend, device=device),
            self.batch_shape,
            metadata=self._metadata,
        )
    
    def sample(self, rng : BRNGType) -> Tuple[
        BRNGType, BArrayType
    ]:
        flat_shape = np.prod(self.batch_shape)
        samples = []
        for i in range(flat_shape):
            rng, single_sample = self.single_space.sample(rng)
            samples.append(single_sample)
        return rng, self._make_object_array(samples)
    
    def create_empty(self):
        flat_shape = np.prod(self.batch_shape)
        empties = [self.single_space.create_empty() for _ in range(flat_shape)]
        return self._make_object_array(empties)
    
    def is_bounded(self, manner = "both"):
        return self.single_space.is_bounded(manner=manner)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray) or x.dtype != object or x.shape != self.batch_shape:
            return False
        return all(self.single_space.contains(xi) for xi in x.flat)

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Clip every element of the object array ``x`` with ``single_space`` (see ``Space.clip``).

        Returns a new object array of ``batch_shape``. Raises ``ValueError`` if
        ``x`` is not an object array of ``batch_shape``, or if any element
        cannot be clipped.
        """
        if not isinstance(x, np.ndarray) or x.dtype != object or x.shape != self.batch_shape:
            raise ValueError(
                f"BatchedSpace.clip: expected an object array of shape {self.batch_shape}, got "
                + (f"{x.dtype} array of shape {x.shape}" if isinstance(x, np.ndarray) else f"{type(x)}")
            )
        return self._make_object_array([self.single_space.clip(xi) for xi in x.flat])

    def get_repr(
        self, 
        abbreviate = False,
        include_backend = True, 
        include_device = True, 
        include_dtype = True
    ):
        ret = f"BatchedSpace({self.single_space}, batch_shape={self.batch_shape}"
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
        return isinstance(other, BatchedSpace) and self.backend == other.backend and self.batch_shape == other.batch_shape and self.single_space == other.single_space

    def is_subspaceeq(self, other: Any) -> bool:
        """Return whether this batched space is a subspace of ``other``.

        True iff ``other`` is a ``BatchedSpace`` on the same backend with equal
        ``batch_shape`` and ``self.single_space.is_subspaceeq(other.single_space)``
        holds recursively (non-strict ⊆). ``device`` is ignored.
        """
        if not (isinstance(other, BatchedSpace) and self.backend == other.backend):
            return False
        if self.batch_shape != other.batch_shape:
            return False
        return self.single_space.is_subspaceeq(other.single_space)
    
    def data_to(self, data, backend = None, device = None):
        ret = np.empty(data.shape, dtype=object)
        for i, d in enumerate(data.flat):
            ret.flat[i] = self.single_space.data_to(d, backend=backend, device=device)
        return ret