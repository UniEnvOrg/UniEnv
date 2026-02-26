from typing import Any, Dict, Optional, Tuple

import numpy as np

from unienv_interface.backends import ComputeBackend, BDeviceType, get_backend_from_tensor
from unienv_interface.backends.serialization import deserialize_backend, serialize_backend

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
    if isinstance(value, (int, float)):
        return None, {"scalar": True, "value": value, "backend": None}

    backend = get_backend_from_tensor(value)
    return backend, {
        "scalar": False,
        "value": backend.to_numpy(value).tolist(),
        "backend": serialize_backend(backend),
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
    value = value_backend.from_numpy(np.array(value_data["value"]))
    if override_device is not None:
        value = value_backend.to_device(value, override_device)
    return value_backend, value
