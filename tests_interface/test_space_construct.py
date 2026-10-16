"""Tests for ``construct_utils`` (inferring spaces from data)."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import (
    BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, TextSpace, TupleSpace,
)
from unienv_interface.space.space_utils import construct_utils as scu
from tests_common.backends import cpu_backend_params


def _arrays(seed, shape, n=4):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=shape).astype(np.float32) for _ in range(n)]


@pytest.mark.parametrize("add_bounds", [True, False])
def test_same_shape_stream_gives_box(add_bounds):
    data = _arrays(0, (3,))
    space = scu.construct_space_from_data_stream(data, B, add_bounds=add_bounds)
    assert isinstance(space, BoxSpace) and space.shape == (3,)
    assert all(space.contains(d) for d in data)
    if add_bounds:
        np.testing.assert_array_equal(space.low, np.min(data, axis=0))
        np.testing.assert_array_equal(space.high, np.max(data, axis=0))
    else:
        assert not space.is_bounded("below") and not space.is_bounded("above")


def test_varying_shape_stream_gives_dynamic_box():
    data = [np.full((n, 2), float(n), dtype=np.float32) for n in (2, 4, 3, 2)]
    space = scu.construct_space_from_data_stream(data, B)
    assert isinstance(space, DynamicBoxSpace)
    assert space.shape_low == (2, 2) and space.shape_high == (4, 2)
    assert all(space.contains(d) for d in data)


def test_nested_stream_forwards_add_bounds():
    data = [
        {"a": a, "t": (b, np.array([True, False])), "s": "hey"}
        for a, b in zip(_arrays(1, (2,)), _arrays(2, (1,)))
    ]
    space = scu.construct_space_from_data_stream(data, B, add_bounds=True)
    assert isinstance(space, DictSpace)
    assert isinstance(space["t"], TupleSpace) and isinstance(space["t"][1], BinarySpace)
    assert isinstance(space["s"], TextSpace) and space["s"].max_length == 3
    assert all(space.contains(d) for d in data)
    np.testing.assert_array_equal(space["a"].low, np.min([d["a"] for d in data], axis=0))
    np.testing.assert_array_equal(space["t"][0].high, np.max([d["t"][0] for d in data], axis=0))


def test_single_sample_without_bounds_is_unbounded():
    space = scu.construct_space_from_data({"a": np.zeros(2, dtype=np.float64), "s": "x"}, B)
    assert not space["a"].is_bounded("both")
    assert space["s"].max_length >= 4096


def test_unsupported_data_raises():
    with pytest.raises(ValueError):
        scu.construct_space_from_data(3.5, B)


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_other_backends(backend, device):
    data = [backend.from_numpy(d, device=device) for d in _arrays(3, (2, 2))]
    space = scu.construct_space_from_data_stream(data, backend)
    assert isinstance(space, BoxSpace) and space.backend is backend
    assert all(space.contains(d) for d in data)
