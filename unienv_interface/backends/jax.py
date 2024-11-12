from typing import Optional, Generic, TypeVar, Dict, Union, Any, Sequence, SupportsFloat, Tuple, Type

from .base import ComputeBackend
import numpy as np
import dlpack
import jax
from packaging.version import Version

jax_version = Version(jax.__version__)
if jax_version < Version("0.4.32"):
    import jax.experimental.array_api
    jax_array_api = jax.experimental.array_api
else:
    jax_array_api = jax.numpy

JaxDevice = Union[jax.Device, jax.sharding.Sharding]
JaxRNG = jax.Array

class JaxComputeBackend(ComputeBackend[jax.Array, JaxDevice, np.dtype, JaxRNG]):
    ARRAY_TYPE = jax.Array
    DEVICE_TYPE = JaxDevice
    DTYPE_TYPE = np.dtype
    RNG_TYPE = JaxRNG

    array_api_namespace = jax_array_api
    default_integer_dtype = int
    default_floating_dtype = float
    default_boolean_dtype = bool

    @classmethod
    def is_backendarray(cls, data : Any) -> bool:
        return isinstance(data, jax.Array)

    @classmethod
    def is_device_tpu(cls, device: JaxDevice) -> bool:
        return device.platform == "tpu"

    @classmethod
    def is_device_gpu(cls, device: JaxDevice) -> bool:
        return device.platform == "gpu"

    @classmethod
    def is_device_cpu(cls, device: JaxDevice) -> bool:
        return device.platform == "cpu"

    @classmethod
    def from_numpy(cls, data : np.ndarray, dtype : Optional[np.dtype] = None, device : Optional[JaxDevice] = None) -> jax.Array:
        return jax.numpy.asarray(data, dtype=dtype, device=device)

    @classmethod
    def to_numpy(cls, data : jax.Array) -> np.ndarray:
        if data.dtype == jax.dtypes.bfloat16:
            data = data.astype(np.float32)
        return np.asarray(data)

    @classmethod
    def from_other_backend(cls, data : dlpack.DLPackObject, backend : Optional[ComputeBackend] = None) -> jax.Array:
        try:
            # if hasattr(data, "contiguous"): # Fix for torch non-standard strides
            #     data = data.contiguous()
            return jax.dlpack.from_dlpack(data)
        except Exception as e:
            if backend is None:
                raise e
            # jax sometimes has tiling issues with dlpack converted data
            np = backend.to_numpy(data)
            return cls.from_numpy(np)

    @classmethod
    def replace_inplace(cls, data: jax.Array, index: jax.Array, value: jax.Array) -> jax.Array:
        return data.at[index].set(value)

    @classmethod
    def random_number_generator(cls, seed : Optional[int] = None, device : Optional[JaxDevice] = None) -> JaxRNG:
        rng_seed = np.random.randint(0) if seed is None else seed
        rng = jax.random.key(
            seed=rng_seed
        )
        return rng
    
    @classmethod
    def random_discrete_uniform(
        cls, 
        rng : JaxRNG, 
        shape : Sequence[int], 
        from_num : int, 
        to_num : int, 
        dtype : Optional[np.dtype] = None, 
        device : Optional[JaxDevice] = None
    ) -> Tuple[JaxRNG, jax.Array]:
        new_rng, rng = jax.random.split(rng)
        data = jax.random.randint(rng, shape, minval=from_num, maxval=to_num, dtype=dtype or int)
        if device is not None:
            data = jax.device_put(data, device)
        return new_rng, data

    @classmethod
    def random_uniform(
        cls, 
        rng : JaxRNG, 
        shape : Sequence[int], 
        lower_bound : float = 0.0, 
        upper_bound : float = 1.0, 
        dtype : Optional[np.dtype] = None, 
        device : Optional[JaxDevice] = None
    ) -> Tuple[JaxRNG, jax.Array]:
        new_rng, rng = jax.random.split(rng)
        data = jax.random.uniform(rng, shape, dtype=dtype or float, minval=lower_bound, maxval=upper_bound)
        if device is not None:
            data = jax.device_put(data, device)
        return new_rng, data

    @classmethod
    def random_exponential(
        cls, 
        rng : JaxRNG, 
        shape : Sequence[int], 
        lambd : float = 1.0, 
        dtype : Optional[np.dtype] = None, 
        device : Optional[JaxDevice] = None
    ) -> Tuple[JaxRNG, jax.Array]:
        new_rng, rng = jax.random.split(rng)
        data = jax.random.exponential(rng, shape, dtype=dtype or float) / lambd
        if device is not None:
            data = jax.device_put(data, device)
        return new_rng, data

    @classmethod
    def random_normal(
        cls, 
        rng : JaxRNG, 
        shape : Sequence[int], 
        mean : float = 0.0, 
        std : float = 1.0, 
        dtype : Optional[np.dtype] = None, 
        device : Optional[JaxDevice] = None
    ) -> Tuple[JaxRNG, np.ndarray]:
        new_rng, rng = jax.random.split(rng)
        data = jax.random.normal(rng, shape, dtype=dtype or float) * std + mean
        if device is not None:
            data = jax.device_put(data, device)
        return new_rng, data

    @classmethod
    def random_geometric(
        cls, 
        rng : JaxRNG, 
        shape : Sequence[int], 
        p : float, 
        dtype : Optional[np.dtype] = None, 
        device : Optional[JaxDevice] = None
    ) -> Tuple[JaxRNG, jax.Array]:
        new_rng, rng = jax.random.split(rng)
        data = jax.random.geometric(rng, p=p, shape=shape, dtype=dtype or int)
        if device is not None:
            data = jax.device_put(data, device)
        return new_rng, data

    @classmethod
    def dtype_is_real_integer(cls, dtype : np.dtype) -> bool:
        return dtype in cls.list_real_integer_dtypes()
    
    @classmethod
    def list_real_integer_dtypes(cls) -> Sequence[np.dtype]:
        return (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, int)

    @classmethod
    def dtype_is_real_floating(cls, dtype : np.dtype) -> bool:
        return dtype in cls.list_real_floating_dtypes()
    
    @classmethod
    def list_real_floating_dtypes(cls) -> Sequence[np.dtype]:
        return (jax.dtypes.bfloat16, np.float16, np.float32, np.float64, float)
    
    @classmethod
    def dtype_is_boolean(cls, dtype : np.dtype) -> bool:
        return dtype in cls.list_real_boolean_dtypes()
    
    @classmethod
    def list_real_boolean_dtypes(cls) -> Sequence[np.dtype]:
        return (np.bool_, bool)