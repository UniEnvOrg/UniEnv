"""Tests for ``unienv_interface.utils.array_serialization``."""

import json

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.utils.array_serialization import (
    deserialize_scalar_or_array_value,
    serialize_scalar_or_array_value,
)
from tests_common.backends import backend_params, cpu_backend_params
from tests_common.trees import to_numpy

DTYPE_NAMES = ["bool", "int8", "int32", "int64", "uint8", "float32", "float64"]


def _round_trip(value, **kwargs):
    backend, payload = serialize_scalar_or_array_value(value)
    payload = json.loads(json.dumps(payload))
    return backend, deserialize_scalar_or_array_value(payload, **kwargs)


@pytest.mark.parametrize("backend,device", backend_params())
@pytest.mark.parametrize("dtype_name", DTYPE_NAMES)
def test_array_round_trip_preserves_dtype_values_and_backend(backend, device, dtype_name):
    dtypes = backend.__array_namespace_info__().dtypes()
    if dtype_name not in dtypes:
        pytest.skip(f"{backend.simplified_name} has no {dtype_name}")
    dtype = dtypes[dtype_name]
    original_np = (np.arange(6).reshape(2, 3) % 2 if dtype_name == "bool" else np.arange(6).reshape(2, 3) * 1.5)
    value = backend.asarray(original_np.astype(np.dtype(dtype_name)), dtype=dtype, device=device)
    ser_backend, (out_backend, out) = _round_trip(value)
    assert ser_backend is backend and out_backend is backend
    assert backend.is_backendarray(out)
    assert out.dtype == value.dtype
    np.testing.assert_array_equal(to_numpy(out), to_numpy(value))


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_override_backend(backend, device):
    value = np.array([1.0, 2.0], np.float32)
    _, (out_backend, out) = _round_trip(value, override_backend=backend, override_device=device)
    assert out_backend is backend
    assert backend.is_backendarray(out)
    assert out.dtype == backend.__array_namespace_info__().dtypes()["float32"]


@pytest.mark.parametrize("value", [3, -2.5, True])
def test_python_scalars(value):
    backend, payload = serialize_scalar_or_array_value(value)
    assert backend is None and payload["scalar"] is True
    out_backend, out = deserialize_scalar_or_array_value(payload)
    assert out_backend is None and out == value and type(out) is type(value)


def test_none():
    assert serialize_scalar_or_array_value(None) == (None, None)
    assert deserialize_scalar_or_array_value(None) == (None, None)


def test_legacy_payload_without_dtype_still_loads():
    payload = {"scalar": False, "value": [1.0, 2.0], "backend": "numpy"}
    backend, out = deserialize_scalar_or_array_value(payload)
    assert backend is NumpyComputeBackend
    np.testing.assert_array_equal(out, [1.0, 2.0])
