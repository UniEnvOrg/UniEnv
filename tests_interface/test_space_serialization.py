"""Builtin space serialization coverage, independent of remote execution."""

from __future__ import annotations

import json
from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pytest

from unienv_interface.backends import NumpyComputeBackend as B
from unienv_interface.space import spaces
from unienv_interface.space.spaces import (
    BatchedSpace, BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, GraphSpace,
    TextSpace, TupleSpace, UnionSpace,
)
from unienv_interface.space.space_utils.serialization_utils import (
    SPACE_TYPES, json_to_space, space_to_json,
)

if TYPE_CHECKING:
    from unienv_interface.backends import ComputeBackend


@pytest.fixture(params=["numpy", "pytorch", "jax"])
def backend_device(request: pytest.FixtureRequest) -> Tuple[ComputeBackend, Optional[object]]:
    if request.param == "pytorch":
        pytest.importorskip("torch")
        from unienv_interface.backends.pytorch import PyTorchComputeBackend
        return PyTorchComputeBackend, "cpu"
    if request.param == "jax":
        jax = pytest.importorskip("jax")
        from unienv_interface.backends.jax import JaxComputeBackend
        return JaxComputeBackend, jax.devices("cpu")[0]
    return B, None


def builtin_spaces(backend: ComputeBackend, device: Optional[object]) -> Dict[str, spaces.Space]:
    def box(shape: Sequence[int]) -> BoxSpace:
        return BoxSpace(backend, -np.inf, np.inf, backend.float32, shape=shape, device=device)

    return {
        "BoxSpace": box((2,)),
        "BinarySpace": BinarySpace(backend, (2,), device=device),
        "TextSpace": TextSpace(backend, 16, min_length=2, charset="abc", device=device),
        "DynamicBoxSpace": DynamicBoxSpace(
            backend, -2, backend.from_numpy(np.array([[3, 4]], np.float32), device=device),
            (1, 2), (4, 2), backend.float32, device=device, fill_value=-1),
        "TupleSpace": TupleSpace(backend, [box((1,)), TextSpace(backend, 4, device=device)], device=device),
        "DictSpace": DictSpace(backend, {"value": box((2,))}, device=device),
        "UnionSpace": UnionSpace(backend, [box((2,)), TextSpace(backend, 4, device=device)], device=device),
        "BatchedSpace": BatchedSpace(DictSpace(backend, {"value": box((2,))}, device=device), (3,)),
        "GraphSpace": GraphSpace(backend, box((2,)), edge_feature_space=box((1,)), is_edge=True,
                                 min_nodes=1, max_nodes=4, min_edges=1, max_edges=6,
                                 batch_shape=(2,), device=device),
    }


def test_serialization_registries_cover_all_builtin_spaces() -> None:
    builtin_types = {value for name in spaces.__all__ if isinstance(value := getattr(spaces, name), type)
                     and issubclass(value, spaces.Space) and value is not spaces.Space}
    assert set(space_to_json.registry) - {object} == builtin_types
    assert set(json_to_space.registry) - {object} == builtin_types
    assert set(SPACE_TYPES.values()) == builtin_types
    examples = builtin_spaces(B, None)
    assert set(examples) == set(SPACE_TYPES)
    for name, cls in SPACE_TYPES.items():
        assert type(examples[name]) is cls
        assert space_to_json(examples[name])["type"] == name


@pytest.mark.parametrize("space_name", list(SPACE_TYPES))
def test_builtin_space_roundtrip(backend_device: Tuple[ComputeBackend, Optional[object]], space_name: str) -> None:
    backend, device = backend_device
    source = builtin_spaces(backend, device)[space_name]
    descriptor = space_to_json(source)
    numpy_descriptor = space_to_json(builtin_spaces(B, None)[space_name])
    assert descriptor == numpy_descriptor
    if isinstance(source, TextSpace):
        assert descriptor["charset"] == "abc"
    for payload in (descriptor, json.loads(json.dumps(descriptor))):
        restored = json_to_space(payload, backend, device)
        assert type(restored) is type(source)
        assert restored.backend is backend
        assert restored.device == source.device
        assert space_to_json(restored) == descriptor
        if isinstance(restored, DynamicBoxSpace):
            assert restored.shape_low == (1, 2)
            assert restored.shape_high == (4, 2)
            assert backend.to_numpy(restored._high).dtype == np.float32
            np.testing.assert_array_equal(backend.to_numpy(restored._high), [[3, 4]])
        if isinstance(restored, BatchedSpace):
            assert restored.batch_shape == (3,)
            assert restored.single_space["value"].shape == (2,)
