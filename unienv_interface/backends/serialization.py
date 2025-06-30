from typing import Optional, Union
from .base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

__all__ = [
    "serialize_backend",
    "deserialize_backend",
    "serialize_dtype",
    "deserialize_dtype",
]

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
    if dtype_str not in all_dtypes_mapping:
        raise ValueError(f"Unknown dtype string: {dtype_str}")
    return all_dtypes_mapping[dtype_str]