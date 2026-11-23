from typing import Optional, Union
from .base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

__all__ = [
    "serialize_backend",
    "deserialize_backend",
    "serialize_dtype",
    "deserialize_dtype",
]

# Names of the real / complex numeric dtypes that may be serialized even when a backend's
# array-API dtype table does not list them.
_NUMERIC_DTYPE_NAMES = (
    "bool",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "bfloat16", "float32", "float64",
    "complex64", "complex128",
)

def _native_dtype(backend : ComputeBackend, name : str, canonical : bool) -> Optional[BDtypeType]:
    """The dtype of ``backend`` called ``name`` (``None`` if the backend has none).

    With ``canonical=True`` the dtype arrays actually get is returned (JAX without x64
    stores float64 / int64 / uint64 / complex128 as their 32-bit counterparts).
    """
    try:
        kind = backend.simplified_name
        if kind == "pytorch":
            import torch
            dtype = getattr(torch, name, None)
            return dtype if isinstance(dtype, torch.dtype) else None
        import numpy as np
        if kind == "jax":
            import jax
            import jax.numpy as jnp
            dtype = jnp.dtype(name)
            return jax.dtypes.canonicalize_dtype(dtype) if canonical else dtype
        if name == "bfloat16":
            import ml_dtypes
            return np.dtype(ml_dtypes.bfloat16)
        return np.dtype(name)
    except Exception:
        return None

def serialize_backend(backend: ComputeBackend) -> str:
    """
    Serialize a ComputeBackend to a string representation.
    
    Args:
        backend (ComputeBackend): The backend to serialize.
    
    Returns:
        str: A string representation of the backend.
    """
    return backend.simplified_name

def deserialize_backend(backend_str: str) -> ComputeBackend:
    if backend_str == "numpy":
        from .numpy import NumpyComputeBackend
        return NumpyComputeBackend
    elif backend_str == "pytorch":
        from .pytorch import PyTorchComputeBackend
        return PyTorchComputeBackend
    elif backend_str == "jax":
        from .jax import JaxComputeBackend
        return JaxComputeBackend
    else:
        raise ValueError(f"Unknown backend: {backend_str}")

def serialize_dtype(backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType], dtype: Optional[BDtypeType]) -> Optional[str]:
    """
    Serialize a data type to a string representation.
    
    Args:
        dtype (BDtypeType): The data type to serialize.
    
    Returns:
        str: A string representation of the data type.
    """
    if dtype is None:
        return None

    all_dtypes_mapping = backend.__array_namespace_info__().dtypes()
    for key, value in all_dtypes_mapping.items():
        if value == dtype:
            return key
    # Some backends (e.g. JAX) use Python scalar types such as ``float`` as their
    # default dtypes; resolve them to the concrete dtype of an array.
    try:
        concrete_dtype = backend.zeros((1,), dtype=dtype).dtype
    except Exception:
        concrete_dtype = None
    if concrete_dtype is not None:
        for key, value in all_dtypes_mapping.items():
            if value == concrete_dtype:
                return key
    # Real dtypes that are missing from the backend's array-API table (e.g. float16,
    # bfloat16, or torch's unsigned types) are serialized by their name.
    for name in _NUMERIC_DTYPE_NAMES:
        native = _native_dtype(backend, name, canonical=False)
        if native is not None and (native == dtype or (concrete_dtype is not None and native == concrete_dtype)):
            return name
    raise ValueError(f"Unknown dtype: {dtype}")

def deserialize_dtype(backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType], dtype_str: Optional[str]) -> Optional[BDtypeType]:
    """
    Deserialize a string representation of a data type to its corresponding type.
    
    Args:
        dtype_str (str): The string representation of the data type.
    
    Returns:
        BDtypeType: The deserialized data type.
    """
    if dtype_str is None:
        return None
    all_dtypes_mapping = backend.__array_namespace_info__().dtypes()
    if dtype_str in all_dtypes_mapping:
        return all_dtypes_mapping[dtype_str]
    # Names missing from the array-API table: dtypes the backend has outside of it
    # (float16, bfloat16, ...), or ones it stores narrower (JAX without x64 keeps
    # float64 / int64 data as float32 / int32, which is what it gets here).
    native = _native_dtype(backend, dtype_str, canonical=True) if dtype_str in _NUMERIC_DTYPE_NAMES else None
    if native is None:
        raise ValueError(f"Unknown dtype string: {dtype_str}")
    return native