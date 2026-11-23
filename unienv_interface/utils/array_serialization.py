from typing import Any, Dict, Optional, Tuple

import numpy as np

from unienv_interface.backends import ComputeBackend, BDeviceType, get_backend_from_tensor
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.serialization import deserialize_backend, serialize_backend, serialize_dtype, deserialize_dtype

__all__ = [
    "serialize_scalar_or_array_value",
    "deserialize_scalar_or_array_value",
]


def serialize_scalar_or_array_value(
    value: Any,
) -> Tuple[Optional[ComputeBackend], Optional[Dict[str, Any]]]:
    """
    Serialize a scalar or backend array value to a JSON-compatible payload.

    Returns:
        (backend, serialized_data)
        - backend is only set for backend-array values.
        - serialized_data is None when the input value is None.
    """
    if value is None:
        return None, None
    if isinstance(value, np.generic):
        # numpy scalars (e.g. np.float32(1.0)) are not backend arrays
        value = value.item()
    if isinstance(value, (int, float)):
        return None, {"scalar": True, "value": value, "backend": None}

    backend = get_backend_from_tensor(value)
    try:
        numpy_value = backend.to_numpy(value)
    except Exception:
        # dtypes numpy lacks (e.g. torch bfloat16): float32 holds every value exactly
        numpy_value = backend.to_numpy(backend.astype(value, backend.__array_namespace_info__().dtypes()["float32"]))
    return backend, {
        "scalar": False,
        "value": numpy_value.tolist(),
        "backend": serialize_backend(backend),
        "dtype": serialize_dtype(backend, value.dtype),
    }


def deserialize_scalar_or_array_value(
    value_data: Optional[Dict[str, Any]],
    override_backend: Optional[ComputeBackend] = None,
    override_device: Optional[BDeviceType] = None,
) -> Tuple[Optional[ComputeBackend], Any]:
    """
    Deserialize a scalar or backend array value from serialized payload.

    Returns:
        (backend, value)
        - backend is only set for backend-array values.
        - value is None when input value_data is None.
    """
    if value_data is None:
        return None, None
    if value_data.get("scalar", False):
        return None, value_data["value"]

    value_backend = override_backend or deserialize_backend(value_data["backend"])
    # Payloads written before the dtype was recorded fall back to numpy's inference.
    dtype_name = value_data.get("dtype", None)
    try:
        np_dtype = None if dtype_name is None else deserialize_dtype(NumpyComputeBackend, dtype_name)
    except ValueError:
        np_dtype = None  # e.g. bfloat16 without ml_dtypes: cast on the target backend below
    if np_dtype is not None and np_dtype.name == "bfloat16":
        np_dtype = np.dtype(np.float32)  # backends cannot import numpy's (ml_dtypes) bfloat16
    value = value_backend.from_numpy(np.array(value_data["value"], dtype=np_dtype))
    if dtype_name is not None and (np_dtype is None or np_dtype.name != dtype_name):
        value = value_backend.astype(value, deserialize_dtype(value_backend, dtype_name))
    if override_device is not None:
        value = value_backend.to_device(value, override_device)
    return value_backend, value
