"""Unit tests for ``BatchedSpace`` (object-array batches of non-stackable spaces)."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BatchedSpace, BoxSpace, TextSpace, UnionSpace
from tests_common.backends import cpu_backend_params
from tests_common.trees import assert_tree_equal


def _text():
    return TextSpace(B, 5, min_length=1, charset="abc")


def _rng(seed=0):
    return B.random.random_number_generator(seed)


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_contains_own_sample(batch_shape):
    space = BatchedSpace(_text(), batch_shape)
    _, x = space.sample(_rng())
    assert isinstance(x, np.ndarray) and x.dtype == object and x.shape == batch_shape
    assert space.contains(x)


def test_contains_rejects_wrong_shape_type_and_element():
    space = BatchedSpace(_text(), (2,))
    good = np.empty((2,), dtype=object)
    good[0], good[1] = "ab", "c"
    assert space.contains(good)
    assert not space.contains(["ab", "c"])
    assert not space.contains(np.array(["ab", "c"]))  # not an object array
    wrong_len = np.empty((3,), dtype=object)
    wrong_len[:] = "a"
    assert not space.contains(wrong_len)
    bad_elem = good.copy()
    bad_elem[1] = "xyz"  # outside the charset
    assert not space.contains(bad_elem)
    # The per-dimension check must not ignore the inner batch dimension.
    nested = BatchedSpace(_text(), (2, 3))
    flat = np.empty((2, 2), dtype=object)
    flat[:] = "a"
    assert not nested.contains(flat)


def test_shape_for_shapeless_single_space():
    assert BatchedSpace(_text(), (2,)).shape == (2,)
    assert BatchedSpace(_text(), (2, 3)).shape == (2, 3)
    box = BoxSpace(B, 0.0, 1.0, np.float32, shape=(3,))
    assert BatchedSpace(box, (2,)).shape == (2, 3)


@pytest.mark.parametrize("single", [
    BoxSpace(B, 0.0, 1.0, np.float32, shape=(3,)),
    UnionSpace(B, [BoxSpace(B, 0.0, 1.0, np.float32, shape=(2,)), TextSpace(B, 3, charset="ab")]),
], ids=["box", "union"])
def test_sample_and_create_empty_with_sequence_like_elements(single):
    space = BatchedSpace(single, (2,))
    _, x = space.sample(_rng())
    assert x.shape == (2,) and x.dtype == object
    assert space.contains(x)
    empty = space.create_empty()
    assert empty.shape == (2,) and empty.dtype == object


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_data_to_returns_object_array(backend, device):
    space = BatchedSpace(_text(), (2, 2))
    _, x = space.sample(_rng())
    converted = space.data_to(x, backend, device)
    assert isinstance(converted, np.ndarray) and converted.dtype == object
    assert converted.shape == (2, 2)
    assert space.to(backend, device).contains(converted)

    box_space = BatchedSpace(BoxSpace(B, 0.0, 1.0, np.float32, shape=(3,)), (2,))
    _, bx = box_space.sample(_rng(1))
    moved = box_space.data_to(bx, backend, device)
    assert moved.shape == (2,) and box_space.to(backend, device).contains(moved)
    back = box_space.to(backend, device).data_to(moved, B)
    assert_tree_equal(back, bx)
