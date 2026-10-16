"""Targeted tests for ``flatten_utils`` (flat_dim / flatten_space / flatten_data / unflatten_data)."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import (
    BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, TextSpace, TupleSpace, UnionSpace,
)
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.space.space_utils import flatten_utils as sfu
from tests_common.backends import cpu_backend_params
from tests_common.trees import assert_tree_equal


def _rng(seed=0):
    return B.random.random_number_generator(seed)


def _check_roundtrip(space, x, start_dim=0):
    flat_space = sfu.flatten_space(space, start_dim)
    flat = sfu.flatten_data(space, x, start_dim)
    assert flat.shape[-1] == sfu.flat_dim(space, start_dim)
    assert flat_space.contains(flat), (flat_space, flat)
    assert_tree_equal(sfu.unflatten_data(space, flat, start_dim), x)


def test_text_roundtrip():
    space = TextSpace(B, 6, min_length=0, charset="abc")
    for text in ["", "a", "abcabc", "cab"]:
        _check_roundtrip(space, text)


def test_text_without_charset_is_not_flattenable():
    space = TextSpace(B, 4)
    assert not sfu.is_flattenable(space)
    with pytest.raises(ValueError):
        sfu.flatten_space(space)


@pytest.mark.parametrize("seed", range(6))
def test_union_roundtrip(seed):
    space = UnionSpace(B, [
        BoxSpace(B, 0.0, 1.0, np.float32, shape=(3,)),
        BoxSpace(B, 2, 5, np.int64, shape=(2,)),
        BinarySpace(B, (1,)),
    ])
    _, x = space.sample(_rng(seed))
    assert sfu.flat_dim(space) == 4
    _check_roundtrip(space, x)
    idx, _ = sfu.unflatten_data(space, sfu.flatten_data(space, x))
    assert type(idx) is int


def test_union_padding_is_inside_flat_space():
    # Every alternative is bounded away from zero, yet unused slots are padded with zeros.
    space = UnionSpace(B, [
        BoxSpace(B, 5.0, 6.0, np.float32, shape=(1,)),
        BoxSpace(B, 5.0, 6.0, np.float32, shape=(3,)),
    ])
    _check_roundtrip(space, (0, np.array([5.5], dtype=np.float32)))


def test_dict_order_after_setitem():
    space = DictSpace(B, {"b": BoxSpace(B, 0.0, 1.0, np.float32, shape=(3,))})
    space["a"] = BoxSpace(B, 5.0, 6.0, np.float32, shape=(1,))
    _, x = space.sample(_rng())
    _check_roundtrip(space, x)


def test_dict_with_unsortable_keys():
    space = DictSpace(B, {1: BoxSpace(B, 0.0, 1.0, np.float32, shape=(2,)), "a": BoxSpace(B, 5.0, 6.0, np.float32, shape=(1,))})
    _, x = space.sample(_rng())
    _check_roundtrip(space, x)


@pytest.mark.parametrize("fill_value", [0, -3])
def test_dynamic_box_padding_is_inside_flat_space(fill_value):
    space = DynamicBoxSpace(B, 1.0, 2.0, (1, 2), (3, 2), np.float32, fill_value=fill_value)
    flat_space = sfu.flatten_space(space)
    for seed in range(5):
        _, x = space.sample(_rng(seed))
        _check_roundtrip(space, x)
    # Positions that can never be padding keep the tight bounds.
    assert np.all(flat_space.low[:2] == 1.0) and np.all(flat_space.high[:2] == 2.0)


def test_batched_start_dim_one():
    single = TupleSpace(B, [
        BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,)),
        DictSpace(B, {"z": BinarySpace(B, (3,)), "y": BoxSpace(B, 0, 4, np.int64, shape=(2,))}),
    ])
    batched = sbu.batch_space(single, 4)
    _, x = batched.sample(_rng())
    flat = sfu.flatten_data(batched, x, start_dim=1)
    assert flat.shape == (4, sfu.flat_dim(single))
    _check_roundtrip(batched, x, start_dim=1)


def test_zero_length_text_flattens_to_empty_box():
    space = TextSpace(B, 0, charset="ab")
    assert sfu.flat_dim(space) == 0
    _check_roundtrip(space, "")


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_dynamic_box_in_mixed_dict_keeps_dtype(dtype):
    # The flat array is promoted to a common dtype; unflattening must restore the leaf dtype.
    space = DictSpace(B, {
        "a": BinarySpace(B, ()),
        "b": DynamicBoxSpace(B, 0, 3, (1,), (3,), dtype, fill_value=-1),
        "c": BoxSpace(B, 0.0, 1.0, np.float64, shape=(1,)),
    })
    for seed in range(3):
        _, x = space.sample(_rng(seed))
        _check_roundtrip(space, x)


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_union_flatten_on_every_backend(backend, device):
    float32 = backend.__array_namespace_info__().dtypes()["float32"]
    int32 = backend.__array_namespace_info__().dtypes()["int32"]
    space = UnionSpace(backend, [
        BoxSpace(backend, 1.0, 2.0, float32, shape=(2,), device=device),
        BoxSpace(backend, 0, 3, int32, shape=(1,), device=device),
    ], device=device)
    rng = backend.random.random_number_generator(0, device=device)
    for _ in range(4):
        rng, x = space.sample(rng)
        flat = sfu.flatten_data(space, x)
        assert sfu.flatten_space(space).contains(flat)
        idx, value = sfu.unflatten_data(space, flat)
        assert idx == x[0]
        assert_tree_equal(value, x[1])
