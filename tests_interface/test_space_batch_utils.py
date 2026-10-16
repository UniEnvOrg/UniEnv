"""Targeted tests for ``batch_utils``."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import (
    BatchedSpace, BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, GraphSpace, TextSpace,
    TupleSpace, UnionSpace,
)
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.transformations import BatchifyTransformation, UnBatchifyTransformation
from tests_common.trees import assert_tree_equal


def _rng(seed=0):
    return B.random.random_number_generator(seed)


def _text():
    return TextSpace(B, 4, min_length=1, charset="ab")


def _union():
    return UnionSpace(B, [BoxSpace(B, 0.0, 1.0, np.float32, shape=(2,)), _text()])


def test_batch_differing_box_spaces_with_scalar_bounds():
    spaces = [BoxSpace(B, 0.0, 1.0, np.float32, shape=(3,)), BoxSpace(B, -1.0, 2.0, np.float32, shape=(3,))]
    batched = sbu.batch_differing_spaces(spaces)
    assert batched.shape == (2, 3)
    np.testing.assert_array_equal(batched.low, [[0, 0, 0], [-1, -1, -1]])
    unbatched = list(sbu.unbatch_spaces(batched))
    assert unbatched == spaces


def test_batch_differing_nested_spaces():
    a = DictSpace(B, {"x": BoxSpace(B, 0.0, 1.0, np.float32, shape=(2,)), "y": BinarySpace(B, (1,))})
    b = DictSpace(B, {"x": BoxSpace(B, 0.0, 3.0, np.float32, shape=(2,)), "y": BinarySpace(B, (1,))})
    batched = sbu.batch_differing_spaces([TupleSpace(B, [a]), TupleSpace(B, [b])])
    assert batched[0]["x"].shape == (2, 2) and batched[0]["y"].shape == (2, 1)
    dyn = DynamicBoxSpace(B, 0.0, 1.0, (1,), (3,), np.float32)
    assert sbu.batch_differing_spaces([dyn, dyn]).shape_high == (2, 3)


@pytest.mark.parametrize("make_single", [_text, _union], ids=["text", "union"])
def test_concatenate_text_and_union(make_single):
    single = make_single()
    rng = _rng()
    items = []
    for _ in range(3):
        rng, x = single.sample(rng)
        items.append(x)
    batched_space = sbu.batch_space(single, 3)
    for space_arg in (single, batched_space):
        batch = sbu.concatenate(space_arg, items)
        assert isinstance(batch, np.ndarray) and batch.dtype == object and batch.shape == (3,)
        assert batched_space.contains(batch)
        for i, x in enumerate(items):
            assert_tree_equal(sbu.get_at(batched_space, batch, i), x)


def test_concatenate_nested_batched_space_stacks():
    inner = BatchedSpace(_text(), (2,))
    rng = _rng()
    items = []
    for _ in range(3):
        rng, x = inner.sample(rng)
        items.append(x)
    outer = sbu.batch_space(inner, 3)
    batch = sbu.concatenate(outer, items)
    assert batch.shape == (3, 2)
    assert outer.contains(batch)
    assert_tree_equal(sbu.get_at(outer, batch, 1), items[1])


def test_batchify_text_roundtrip():
    single = _text()
    t = BatchifyTransformation()
    target = t.get_target_space_from_source(single)
    batch = t.transform(single, "ab")
    assert target.contains(batch)
    inv = t.direction_inverse(single)
    assert inv.get_target_space_from_source(target) == single
    assert inv.transform(target, batch) == "ab"


def test_reshape_and_swap_batch_dims_roundtrip():
    single = DictSpace(B, {
        "box": BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,)),
        "bin": BinarySpace(B, (1,)),
        "tup": TupleSpace(B, [BoxSpace(B, 0, 3, np.int64, shape=())]),
    })
    space = sbu.batch_space(sbu.batch_space(single, 3), 2)  # batch dims (2, 3)
    _, x = space.sample(_rng())
    flat_space = sbu.reshape_batch_size(space, (2, 3), (6,))
    flat_x = sbu.reshape_batch_size_in_data(B, x, (2, 3), (6,))
    assert flat_space.contains(flat_x)
    assert_tree_equal(sbu.reshape_batch_size_in_data(B, flat_x, (6,), (2, 3)), x)
    assert sbu.reshape_batch_size(flat_space, (6,), (2, 3)) == space

    swapped = sbu.swap_batch_dims(space, 0, 1)
    swapped_x = sbu.swap_batch_dims_in_data(B, x, 0, 1)
    assert swapped.contains(swapped_x)
    assert_tree_equal(sbu.swap_batch_dims_in_data(B, swapped_x, 0, 1), x)
    assert sbu.swap_batch_dims(swapped, 0, 1) == space


def test_set_at_and_iterate():
    single = TupleSpace(B, [BoxSpace(B, 0.0, 1.0, np.float32, shape=(2,)), BinarySpace(B, (2,))])
    space = sbu.batch_space(single, 4)
    rng, x = space.sample(_rng())
    rng, v = single.sample(rng)
    y = sbu.set_at(space, x, 2, v)
    assert space.contains(y)
    assert_tree_equal(sbu.get_at(space, y, 2), v)
    assert_tree_equal(sbu.get_at(space, y, 0), sbu.get_at(space, x, 0))
    items = list(sbu.iterate(space, y))
    assert len(items) == 4 and all(single.contains(i) for i in items)

    text_space = sbu.batch_space(_text(), 3)
    _, tx = text_space.sample(_rng(1))
    ty = sbu.set_at(text_space, tx, 1, "bb")
    assert text_space.contains(ty) and ty[1] == "bb" and ty is not tx
    assert list(sbu.iterate(text_space, ty))[1] == "bb"


def test_batch_size():
    assert sbu.batch_size(BoxSpace(B, 0.0, 1.0, np.float32, shape=(5, 2))) == 5
    assert sbu.batch_size(BoxSpace(B, 0.0, 1.0, np.float32, shape=())) is None
    assert sbu.batch_size(sbu.batch_space(_text(), 7)) == 7
    assert sbu.batch_size(DictSpace(B, {"a": BinarySpace(B, (4,))})) == 4
    graph = GraphSpace(B, BoxSpace(B, 0.0, 1.0, np.float32, shape=(1,)), max_nodes=3, batch_shape=(2,))
    assert sbu.batch_size(graph) == 2
