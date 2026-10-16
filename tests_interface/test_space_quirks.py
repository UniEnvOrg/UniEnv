"""Documented, intentional behaviours of the space layer that differ from a naive contract."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import (
    BatchedSpace, BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, GraphSpace, TextSpace,
)
from unienv_interface.space.space_utils import flatten_utils as sfu, gym_utils as sgu


def test_box_contains_is_strict_about_float_width():
    space = BoxSpace(B, 0.0, 1.0, np.float32, shape=(1,))
    assert not space.contains(np.array([0.5], dtype=np.float64))  # no implicit narrowing
    assert BoxSpace(B, 0.0, 1.0, np.float64, shape=(1,)).contains(np.array([0.5], dtype=np.float32))


def test_dynamic_box_unflatten_trims_trailing_fill_values():
    # Data that ends in fill_value is indistinguishable from padding.
    space = DynamicBoxSpace(B, 0, 5, (1,), (4,), np.int64, fill_value=0)
    x = np.array([3, 0], dtype=np.int64)
    restored = sfu.unflatten_data(space, sfu.flatten_data(space, x))
    np.testing.assert_array_equal(restored, [3])
    # A fill_value outside the bounds avoids the ambiguity
    space = DynamicBoxSpace(B, 0, 5, (1,), (4,), np.int64, fill_value=-1)
    np.testing.assert_array_equal(sfu.unflatten_data(space, sfu.flatten_data(space, x)), x)


def test_text_without_charset_is_not_flattenable():
    assert sfu.flat_dim(TextSpace(B, 3)) is None
    assert sfu.flat_dim(TextSpace(B, 3, charset="ab")) == 3


def test_graph_create_empty_is_unsupported():
    space = GraphSpace(B, BoxSpace(B, 0.0, 1.0, np.float32, shape=(1,)), max_nodes=3)
    with pytest.raises(NotImplementedError):
        space.create_empty()


def test_binary_equality_compares_device_unlike_other_spaces():
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as T
    assert BinarySpace(T, (2,), device="cpu") != BinarySpace(T, (2,), device=None)
    assert BinarySpace(T, (2,), device="cpu").is_subspaceeq(BinarySpace(T, (2,), device=None))
    import torch
    box = BoxSpace(T, 0.0, 1.0, torch.float32, shape=(2,))
    assert box == box.to(device="cpu")


def test_dict_contains_requires_exact_keys_but_subspace_allows_extra():
    small = DictSpace(B, {"a": BinarySpace(B, (1,))})
    large = DictSpace(B, {"a": BinarySpace(B, (1,)), "b": BinarySpace(B, (1,))})
    assert small.is_subspaceeq(large) and small.is_subspace(large)
    _, x = small.sample(B.random.random_number_generator(0))
    assert not large.contains(x)


@pytest.mark.parametrize("space", [
    DynamicBoxSpace(B, 0.0, 1.0, (1,), (3,), np.float32),
    BatchedSpace(TextSpace(B, 3, charset="ab"), (2,)),
], ids=["dynamic_box", "batched"])
def test_no_gym_equivalent(space):
    with pytest.raises(TypeError):
        sgu.to_gym_space(space)
