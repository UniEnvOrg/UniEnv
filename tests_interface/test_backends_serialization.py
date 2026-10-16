"""Tests for ``unienv_interface.backends.serialization``."""

import pytest

from unienv_interface.backends.serialization import (
    deserialize_backend,
    deserialize_dtype,
    serialize_backend,
    serialize_dtype,
)
from tests_common.backends import cpu_backend_params


def _dtype_params():
    params = []
    for p in cpu_backend_params():
        backend = p.values[0]
        for name in backend.__array_namespace_info__().dtypes():
            params.append(pytest.param(backend, name, id=f"{backend.simplified_name}-{name}"))
    return params


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_backend_round_trip(backend, device):
    name = serialize_backend(backend)
    assert name in ("numpy", "pytorch", "jax")
    assert deserialize_backend(name) is backend


def test_unknown_backend_name():
    with pytest.raises(ValueError):
        deserialize_backend("tensorflow")


@pytest.mark.parametrize("backend,name", _dtype_params())
def test_dtype_round_trip(backend, name):
    dtype = backend.__array_namespace_info__().dtypes()[name]
    assert serialize_dtype(backend, dtype) == name
    assert deserialize_dtype(backend, name) == dtype


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_dtype_none_and_unknown(backend, device):
    assert serialize_dtype(backend, None) is None
    assert deserialize_dtype(backend, None) is None
    with pytest.raises(ValueError):
        deserialize_dtype(backend, "float128x")
    with pytest.raises(ValueError):
        serialize_dtype(backend, object())


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_default_dtypes_serialize(backend, device):
    # JAX's defaults are Python scalar types (float/int/bool); they must still serialize
    # to the concrete dtype arrays created with them get.
    for dtype in (backend.default_floating_dtype, backend.default_integer_dtype, backend.default_boolean_dtype):
        name = serialize_dtype(backend, dtype)
        assert deserialize_dtype(backend, name) == backend.zeros((1,), dtype=dtype).dtype


def test_jax_box_space_with_default_dtype_serializes():
    jax_backend = pytest.importorskip("unienv_interface.backends.jax").JaxComputeBackend
    from unienv_interface.space import BoxSpace
    from unienv_interface.space.space_utils.serialization_utils import json_to_space, space_to_json

    space = BoxSpace(jax_backend, -1.0, 1.0, jax_backend.default_floating_dtype, shape=(2,))
    restored = json_to_space(space_to_json(space), jax_backend, None)
    assert restored.shape == (2,)
