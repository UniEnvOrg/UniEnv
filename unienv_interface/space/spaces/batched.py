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
    ):
        assert len(batch_shape) > 0, "Batch shape must be a non-empty sequence"
        batch_shape = tuple(int(dim) for dim in batch_shape)  # This changes any np types to int
        super().__init__(
            backend=single_space.backend,
            shape=batch_shape + ((),) if single_space.shape is None else batch_shape + single_space.shape,
            device=single_space.device,
            dtype=single_space.dtype,
        )

        self.batch_shape = batch_shape
        self.single_space = single_space

    def to(self, backend = None, device = None):
        return BatchedSpace(
            self.single_space.to(backend=backend, device=device),
            self.batch_shape
        )
    
    def sample(self, rng : BRNGType) -> Tuple[
        BRNGType, BArrayType
    ]:
        flat_shape = np.prod(self.batch_shape)
        samples = []
        for i in range(flat_shape):
            rng, single_sample = self.single_space.sample(rng)
            samples.append(single_sample)
        return rng, np.asarray(samples, dtype=object).reshape(self.batch_shape)
    
    def create_empty(self):
        flat_shape = np.prod(self.batch_shape)
        empties = [self.single_space.create_empty() for _ in range(flat_shape)]
        return np.asarray(empties, dtype=object).reshape(self.batch_shape)
    
    def is_bounded(self, manner = "both"):
        return self.single_space.is_bounded(manner=manner)

    def contains(self, x: BArrayType) -> bool:
        def is_contained_func(x):
            return self.single_space.contains(x)
        for _dim in reversed(self.batch_shape):
            def new_is_contained_func(x):
                if (not isinstance(x, np.ndarray)) or (not x.dtype == object):
                    return False
                if len(x) != _dim:
                    return False
                return all(is_contained_func(xi) for xi in x)
            is_contained_func = new_is_contained_func
        return is_contained_func(x)

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
    
    def data_to(self, data, backend = None, device = None):
        if isinstance(data, np.ndarray) and data.dtype == object:
            return tuple(self.data_to(d, backend=backend, device=device) for d in data)
        else:
            return self.single_space.data_to(data, backend=backend, device=device)