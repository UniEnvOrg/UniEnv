from typing import Optional, Generic, TypeVar, Dict, Union, Any, Sequence, SupportsFloat, Tuple
import array_api_compat.numpy
from .base import ComputeBackend
import numpy as np
import dlpack

class NumpyComputeBackend(ComputeBackend[np.ndarray, Any, np.dtype, np.random.Generator]):
    array_api_namespace = array_api_compat.numpy

    @classmethod
    def is_backendarray(cls, data : Any) -> bool:
        return array_api_compat.is_torch_array(data)
    
    @classmethod
    def is_backenddict(cls, data : Any) -> bool:
        return isinstance(data, dict)

    @classmethod
    def from_numpy(cls, data : np.ndarray, dtype : Optional[np.dtype] = None, device : Optional[Any] = None) -> np.ndarray:
        if dtype is not None:
            data = data.astype(dtype)
        return data

    @classmethod
    def to_numpy(cls, data : np.ndarray) -> np.ndarray:
        return data

    @classmethod
    def from_dlpack(cls, data : dlpack.DLPackObject) -> np.ndarray:
        return np.from_dlpack(data)

    @classmethod
    def random_number_generator(cls, seed : Optional[int] = None, device : Optional[Any] = None) -> np.random.Generator:
        rng = np.random.default_rng(seed=seed)
        return rng
    
    @classmethod
    def random_discrete_uniform(
        cls, 
        rng : np.random.Generator, 
        shape : Sequence[int], 
        from_num : int, 
        to_num : int, 
        dtype : Optional[np.dtype] = None, 
        device : Optional[Any] = None
    ) -> Tuple[np.random.Generator, np.ndarray]:
        t = rng.uniform(int(from_num), int(to_num), size=shape)
        if dtype is not None:
            t = t.astype(dtype)
        return rng, t

    @classmethod
    def random_uniform(
        cls, 
        rng : np.random.Generator, 
        shape : Sequence[int], 
        lower_bound : float = 0.0, 
        upper_bound : float = 1.0, 
        dtype : Optional[np.dtype] = None, 
        device : Optional[Any] = None
    ) -> Tuple[np.random.Generator, np.ndarray]:
        t = rng.uniform(float(lower_bound), float(upper_bound), size=shape)
        if dtype is not None:
            t = t.astype(dtype)
        return rng, t

    @classmethod
    def random_exponential(
        cls, 
        rng : np.random.Generator, 
        shape : Sequence[int], 
        lambd : float = 1.0, 
        dtype : Optional[np.dtype] = None, 
        device : Optional[Any] = None
    ) -> Tuple[np.random.Generator, np.ndarray]:
        t = rng.exponential(1.0 / lambd, size=shape)
        if dtype is not None:
            t = t.astype(dtype)
        return rng, t

    @classmethod
    def random_normal(
        cls, 
        rng : np.random.Generator, 
        shape : Sequence[int], 
        mean : float = 0.0, 
        std : float = 1.0, 
        dtype : Optional[np.dtype] = None, 
        device : Optional[Any] = None
    ) -> Tuple[np.random.Generator, np.ndarray]:
        t = rng.normal(mean, std, size=shape)
        if dtype is not None:
            t = t.astype(dtype)
        return rng, t

    @classmethod
    def dtype_is_real_integer(cls, dtype : np.dtype) -> bool:
        return dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, int)
    
    @classmethod
    def dtype_is_real_floating(cls, dtype : np.dtype) -> bool:
        return dtype in (np.float16, np.float32, np.float64, np.float128, float)