"""Tests for the ``is_subspaceeq`` (non-strict ⊆) and ``is_subspace`` (strict ⊂)
capabilities on spaces.

The non-strict ``is_subspaceeq`` (a ⊆ b, equality allowed) is the primary
controller-required-space check. The strict ``is_subspace`` (a ⊂ b) is defined
on the base ``Space`` class as ``self.is_subspaceeq(other) and not
other.is_subspaceeq(self)``.
"""
import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import (
    BoxSpace,
    DictSpace,
    TupleSpace,
    BinarySpace,
    BatchedSpace,
    TextSpace,
    UnionSpace,
    DynamicBoxSpace,
    GraphSpace,
)

ALL_BACKENDS = [NumpyComputeBackend, JaxComputeBackend, PyTorchComputeBackend]
SEEDS = [0, 1, 42]


def _rng(backend, seed):
    return backend.random.random_number_generator(seed)


def make_box(backend, low=0.0, high=1.0, shape=(3,), dtype=None):
    if dtype is None:
        dtype = backend.default_floating_dtype
    return BoxSpace(backend, low=low, high=high, dtype=dtype, shape=shape)


# ---------------------------------------------------------------------------
# BoxSpace
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_box_equal_is_subspace(backend):
    a = make_box(backend, 0.0, 1.0, shape=(4,))
    b = make_box(backend, 0.0, 1.0, shape=(4,))
    assert a.is_subspaceeq(b)
    assert b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_box_tighter_is_subspace(backend):
    # a has tighter bounds than b -> a is a subspace of b.
    a = make_box(backend, 0.1, 0.9, shape=(4,))
    b = make_box(backend, 0.0, 1.0, shape=(4,))
    assert a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_box_wider_is_not_subspace(backend):
    a = make_box(backend, -1.0, 2.0, shape=(4,))
    b = make_box(backend, 0.0, 1.0, shape=(4,))
    assert not a.is_subspaceeq(b)
    assert b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_box_shape_mismatch(backend):
    a = make_box(backend, 0.0, 1.0, shape=(3,))
    b = make_box(backend, 0.0, 1.0, shape=(4,))
    assert not a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_box_dtype_mismatch(backend):
    a = make_box(backend, 0.0, 1.0, shape=(3,), dtype=backend.default_floating_dtype)
    b = make_box(backend, 0.0, 1.0, shape=(3,), dtype=backend.default_integer_dtype)
    assert not a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_box_inf_bounds(backend):
    # b is unbounded; a is bounded -> a is a subspace of b.
    a = make_box(backend, 0.0, 1.0, shape=(3,))
    b = BoxSpace(
        backend,
        low=-backend.inf,
        high=backend.inf,
        dtype=backend.default_floating_dtype,
        shape=(3,),
    )
    assert a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)
    # Two unbounded boxes are mutual subspaces.
    c = BoxSpace(
        backend,
        low=-backend.inf,
        high=backend.inf,
        dtype=backend.default_floating_dtype,
        shape=(3,),
    )
    assert b.is_subspaceeq(c) and c.is_subspaceeq(b)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_box_broadcast_bounds(backend):
    # b's bounds are scalar-broadcast; a's are per-element but within b.
    a = BoxSpace(
        backend,
        low=backend.asarray([0.0, 0.1, 0.2], dtype=backend.default_floating_dtype),
        high=backend.asarray([0.8, 0.9, 1.0], dtype=backend.default_floating_dtype),
        dtype=backend.default_floating_dtype,
        shape=(3,),
    )
    b = make_box(backend, 0.0, 1.0, shape=(3,))
    assert a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


# ---------------------------------------------------------------------------
# DictSpace
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dict_subset_keys_true(backend):
    """Controller use case: env has extra keys beyond required."""
    required = DictSpace(backend, {"obs": make_box(backend, 0.0, 1.0, shape=(2,))})
    env = DictSpace(
        backend,
        {
            "obs": make_box(backend, 0.0, 1.0, shape=(2,)),
            "extra": make_box(backend, -1.0, 1.0, shape=(5,)),
        },
    )
    assert required.is_subspaceeq(env)
    assert not env.is_subspaceeq(required)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dict_missing_key_false(backend):
    a = DictSpace(backend, {"obs": make_box(backend, 0.0, 1.0, shape=(2,))})
    b = DictSpace(backend, {"other": make_box(backend, 0.0, 1.0, shape=(2,))})
    assert not a.is_subspaceeq(b)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dict_extra_key_on_self_false(backend):
    a = DictSpace(
        backend,
        {
            "obs": make_box(backend, 0.0, 1.0, shape=(2,)),
            "extra": make_box(backend, 0.0, 1.0, shape=(2,)),
        },
    )
    b = DictSpace(backend, {"obs": make_box(backend, 0.0, 1.0, shape=(2,))})
    assert not a.is_subspaceeq(b)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dict_nested_recursion(backend):
    inner_a = DictSpace(backend, {"x": make_box(backend, 0.1, 0.9, shape=(2,))})
    inner_b = DictSpace(
        backend,
        {
            "x": make_box(backend, 0.0, 1.0, shape=(2,)),
            "y": make_box(backend, 0.0, 1.0, shape=(3,)),
        },
    )
    a = DictSpace(backend, {"nested": inner_a})
    b = DictSpace(backend, {"nested": inner_b})
    assert a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dict_child_bounds_violation_false(backend):
    a = DictSpace(backend, {"obs": make_box(backend, -1.0, 2.0, shape=(2,))})
    b = DictSpace(
        backend,
        {
            "obs": make_box(backend, 0.0, 1.0, shape=(2,)),
            "extra": make_box(backend, 0.0, 1.0, shape=(2,)),
        },
    )
    assert not a.is_subspaceeq(b)


# ---------------------------------------------------------------------------
# TupleSpace
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_tuple_basic(backend):
    a = TupleSpace(backend, [make_box(backend, 0.1, 0.9, shape=(2,)), make_box(backend, 0.0, 1.0, shape=(3,))])
    b = TupleSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,)), make_box(backend, 0.0, 1.0, shape=(3,))])
    assert a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_tuple_arity_mismatch(backend):
    a = TupleSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,))])
    b = TupleSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,)), make_box(backend, 0.0, 1.0, shape=(3,))])
    assert not a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


# ---------------------------------------------------------------------------
# BinarySpace
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_binary_basic(backend):
    a = BinarySpace(backend, shape=(4,))
    b = BinarySpace(backend, shape=(4,))
    assert a.is_subspaceeq(b)
    assert b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_binary_shape_mismatch(backend):
    a = BinarySpace(backend, shape=(4,))
    b = BinarySpace(backend, shape=(5,))
    assert not a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


# ---------------------------------------------------------------------------
# BatchedSpace
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_batched_basic(backend):
    a = BatchedSpace(make_box(backend, 0.1, 0.9, shape=(2,)), batch_shape=(3,))
    b = BatchedSpace(make_box(backend, 0.0, 1.0, shape=(2,)), batch_shape=(3,))
    assert a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_batched_batch_shape_mismatch(backend):
    a = BatchedSpace(make_box(backend, 0.0, 1.0, shape=(2,)), batch_shape=(3,))
    b = BatchedSpace(make_box(backend, 0.0, 1.0, shape=(2,)), batch_shape=(4,))
    assert not a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


# ---------------------------------------------------------------------------
# TextSpace
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_text_basic(backend):
    a = TextSpace(backend, max_length=5, min_length=1, charset="abc")
    b = TextSpace(backend, max_length=10, min_length=0, charset="abcdef")
    assert a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_text_charset_not_subset(backend):
    a = TextSpace(backend, max_length=5, min_length=0, charset="abcz")
    b = TextSpace(backend, max_length=10, min_length=0, charset="abcdef")
    assert not a.is_subspaceeq(b)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_text_length_violation(backend):
    a = TextSpace(backend, max_length=20, min_length=0, charset="abc")
    b = TextSpace(backend, max_length=10, min_length=0, charset="abcdef")
    assert not a.is_subspaceeq(b)


# ---------------------------------------------------------------------------
# UnionSpace
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_union_basic(backend):
    a = UnionSpace(backend, [make_box(backend, 0.1, 0.9, shape=(2,)), make_box(backend, 0.0, 0.5, shape=(3,))])
    b = UnionSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,)), make_box(backend, 0.0, 1.0, shape=(3,))])
    assert a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_union_count_mismatch(backend):
    a = UnionSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,))])
    b = UnionSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,)), make_box(backend, 0.0, 1.0, shape=(3,))])
    assert not a.is_subspaceeq(b)


# ---------------------------------------------------------------------------
# DynamicBoxSpace
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dynamic_box_basic(backend):
    a = DynamicBoxSpace(
        backend,
        low=0.1,
        high=0.9,
        shape_low=(2, 3),
        shape_high=(2, 5),
        dtype=backend.default_floating_dtype,
    )
    b = DynamicBoxSpace(
        backend,
        low=0.0,
        high=1.0,
        shape_low=(1, 3),
        shape_high=(3, 6),
        dtype=backend.default_floating_dtype,
    )
    assert a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dynamic_box_shape_range_violation(backend):
    a = DynamicBoxSpace(
        backend,
        low=0.0,
        high=1.0,
        shape_low=(2, 3),
        shape_high=(2, 7),
        dtype=backend.default_floating_dtype,
    )
    b = DynamicBoxSpace(
        backend,
        low=0.0,
        high=1.0,
        shape_low=(1, 3),
        shape_high=(3, 6),
        dtype=backend.default_floating_dtype,
    )
    # a.shape_high[1] = 7 > b.shape_high[1] = 6 -> not a subspace.
    assert not a.is_subspaceeq(b)


# ---------------------------------------------------------------------------
# GraphSpace
# ---------------------------------------------------------------------------
def _graph(backend, min_n=1, max_n=5, min_e=1, max_e=10, is_edge=False,
           node_low=0.0, node_high=1.0, node_shape=(2,)):
    node_space = make_box(backend, node_low, node_high, shape=node_shape)
    edge_space = make_box(backend, 0.0, 1.0, shape=(2,)) if is_edge else None
    return GraphSpace(
        backend,
        node_feature_space=node_space,
        edge_feature_space=edge_space,
        is_edge=is_edge,
        min_nodes=min_n,
        max_nodes=max_n,
        min_edges=min_e,
        max_edges=max_e,
    )


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_graph_basic(backend):
    a = _graph(backend, min_n=2, max_n=4, node_low=0.1, node_high=0.9)
    b = _graph(backend, min_n=1, max_n=5, node_low=0.0, node_high=1.0)
    assert a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_graph_count_range_violation(backend):
    a = _graph(backend, min_n=1, max_n=6, node_low=0.0, node_high=1.0)
    b = _graph(backend, min_n=1, max_n=5, node_low=0.0, node_high=1.0)
    assert not a.is_subspaceeq(b)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_graph_is_edge_mismatch(backend):
    a = _graph(backend, is_edge=False)
    b = _graph(backend, is_edge=True)
    assert not a.is_subspaceeq(b)
    assert not b.is_subspaceeq(a)


# ---------------------------------------------------------------------------
# Cross-type comparisons
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_cross_type_returns_false(backend):
    box = make_box(backend, 0.0, 1.0, shape=(2,))
    d = DictSpace(backend, {"a": make_box(backend, 0.0, 1.0, shape=(2,))})
    t = TupleSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,))])
    bin_ = BinarySpace(backend, shape=(2,))
    txt = TextSpace(backend, max_length=5)
    spaces = [box, d, t, bin_, txt]
    for i, s1 in enumerate(spaces):
        for j, s2 in enumerate(spaces):
            if i == j:
                continue
            assert not s1.is_subspaceeq(s2), f"{type(s1)} unexpectedly subspace of {type(s2)}"


# ---------------------------------------------------------------------------
# Consistency property: a.is_subspaceeq(b) => b.contains(a.sample())
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_consistency_box(backend, seed):
    rng = _rng(backend, seed)
    a = make_box(backend, 0.1, 0.9, shape=(4,))
    b = make_box(backend, 0.0, 1.0, shape=(4,))
    assert a.is_subspaceeq(b)
    for _ in range(5):
        rng, sample = a.sample(rng)
        assert b.contains(sample)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_consistency_dict(backend, seed):
    rng = _rng(backend, seed)
    a = DictSpace(backend, {"obs": make_box(backend, 0.1, 0.9, shape=(2,))})
    b = DictSpace(
        backend,
        {
            "obs": make_box(backend, 0.0, 1.0, shape=(2,)),
            "extra": make_box(backend, -1.0, 1.0, shape=(3,)),
        },
    )
    assert a.is_subspaceeq(b)
    for _ in range(5):
        rng, sample = a.sample(rng)
        # b.contains demands exact key equality, so check the shared keys
        # directly: every key of `a` is in `b` and the value is contained.
        assert all(k in b.spaces and b.spaces[k].contains(v) for k, v in sample.items())


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_consistency_tuple(backend, seed):
    rng = _rng(backend, seed)
    a = TupleSpace(backend, [make_box(backend, 0.1, 0.9, shape=(2,)), make_box(backend, 0.0, 0.5, shape=(3,))])
    b = TupleSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,)), make_box(backend, 0.0, 1.0, shape=(3,))])
    assert a.is_subspaceeq(b)
    for _ in range(5):
        rng, sample = a.sample(rng)
        assert b.contains(sample)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_consistency_binary(backend, seed):
    rng = _rng(backend, seed)
    a = BinarySpace(backend, shape=(4,))
    b = BinarySpace(backend, shape=(4,))
    assert a.is_subspaceeq(b)
    for _ in range(5):
        rng, sample = a.sample(rng)
        assert b.contains(sample)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_consistency_text(backend, seed):
    rng = _rng(backend, seed)
    a = TextSpace(backend, max_length=5, min_length=1, charset="abc")
    b = TextSpace(backend, max_length=10, min_length=0, charset="abcdef")
    assert a.is_subspaceeq(b)
    for _ in range(5):
        rng, sample = a.sample(rng)
        assert b.contains(sample)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_consistency_dynamic_box(backend, seed):
    rng = _rng(backend, seed)
    a = DynamicBoxSpace(
        backend,
        low=0.1,
        high=0.9,
        shape_low=(2, 3),
        shape_high=(2, 5),
        dtype=backend.default_floating_dtype,
    )
    b = DynamicBoxSpace(
        backend,
        low=0.0,
        high=1.0,
        shape_low=(1, 3),
        shape_high=(3, 6),
        dtype=backend.default_floating_dtype,
    )
    assert a.is_subspaceeq(b)
    for _ in range(5):
        rng, sample = a.sample(rng)
        assert b.contains(sample)


# ---------------------------------------------------------------------------
# Base Space raises NotImplementedError
# ---------------------------------------------------------------------------
def test_base_space_raises():
    from unienv_interface.space.space import Space

    # Space is abstract; instantiate a minimal concrete subclass to reach the
    # base is_subspaceeq implementation.
    class DummySpace(Space):
        def to(self, backend=None, device=None):
            return self

        def sample(self, rng, **kwargs):
            return rng, None

        def create_empty(self):
            return None

        def is_bounded(self, manner="both"):
            return True

        def contains(self, x):
            return False

        def get_repr(self, abbreviate=False, include_backend=True,
                     include_device=True, include_dtype=True):
            return "DummySpace"

        def data_to(self, data, backend=None, device=None):
            return data

    dummy = DummySpace(backend=NumpyComputeBackend)
    # Base is_subspaceeq raises NotImplementedError (mirrors abstract style).
    with pytest.raises(NotImplementedError):
        dummy.is_subspaceeq(dummy)
    # Strict is_subspace is defined on the base as
    # ``self.is_subspaceeq(other) and not other.is_subspaceeq(self)``; since the
    # base is_subspaceeq raises, the strict check must propagate the
    # NotImplementedError rather than swallow it into False.
    with pytest.raises(NotImplementedError):
        dummy.is_subspace(dummy)


# ===========================================================================
# Strict is_subspace (⊂) tests
#
# Strict semantics: a.is_subspace(b) ⟺ a.is_subspaceeq(b) and not
# b.is_subspaceeq(a). Equal spaces → False; strict subset → True;
# non-subspace → False; cross-type → False (NOT raise, since cross-type
# is_subspaceeq returns False).
# ===========================================================================


# ---------------------------------------------------------------------------
# BoxSpace — strict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_box_strict_equal_is_false(backend):
    a = make_box(backend, 0.0, 1.0, shape=(4,))
    b = make_box(backend, 0.0, 1.0, shape=(4,))
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_box_strict_tighter_is_true(backend):
    a = make_box(backend, 0.1, 0.9, shape=(4,))
    b = make_box(backend, 0.0, 1.0, shape=(4,))
    assert a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_box_strict_non_subspace_is_false(backend):
    a = make_box(backend, -1.0, 2.0, shape=(4,))
    b = make_box(backend, 0.0, 1.0, shape=(4,))
    # a is wider than b: a ⊄ b, so a ⊄ b (strict) is False.
    assert not a.is_subspace(b)
    # b ⊆ a (True) and a ⊆ b (False) -> b ⊂ a is True.
    assert b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_box_strict_shape_mismatch_is_false(backend):
    a = make_box(backend, 0.0, 1.0, shape=(3,))
    b = make_box(backend, 0.0, 1.0, shape=(4,))
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


# ---------------------------------------------------------------------------
# DictSpace — strict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dict_strict_equal_is_false(backend):
    a = DictSpace(backend, {"obs": make_box(backend, 0.0, 1.0, shape=(2,))})
    b = DictSpace(backend, {"obs": make_box(backend, 0.0, 1.0, shape=(2,))})
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dict_strict_fewer_keys_is_true(backend):
    required = DictSpace(backend, {"obs": make_box(backend, 0.0, 1.0, shape=(2,))})
    env = DictSpace(
        backend,
        {
            "obs": make_box(backend, 0.0, 1.0, shape=(2,)),
            "extra": make_box(backend, -1.0, 1.0, shape=(5,)),
        },
    )
    assert required.is_subspace(env)
    assert not env.is_subspace(required)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dict_strict_non_subspace_is_false(backend):
    a = DictSpace(backend, {"obs": make_box(backend, 0.0, 1.0, shape=(2,))})
    b = DictSpace(backend, {"other": make_box(backend, 0.0, 1.0, shape=(2,))})
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


# ---------------------------------------------------------------------------
# TupleSpace — strict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_tuple_strict_equal_is_false(backend):
    a = TupleSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,))])
    b = TupleSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,))])
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_tuple_strict_tighter_child_is_true(backend):
    a = TupleSpace(backend, [make_box(backend, 0.1, 0.9, shape=(2,)), make_box(backend, 0.0, 1.0, shape=(3,))])
    b = TupleSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,)), make_box(backend, 0.0, 1.0, shape=(3,))])
    assert a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_tuple_strict_arity_mismatch_is_false(backend):
    a = TupleSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,))])
    b = TupleSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,)), make_box(backend, 0.0, 1.0, shape=(3,))])
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


# ---------------------------------------------------------------------------
# BinarySpace — strict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_binary_strict_equal_is_false(backend):
    a = BinarySpace(backend, shape=(4,))
    b = BinarySpace(backend, shape=(4,))
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_binary_strict_shape_mismatch_is_false(backend):
    a = BinarySpace(backend, shape=(4,))
    b = BinarySpace(backend, shape=(5,))
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


# ---------------------------------------------------------------------------
# BatchedSpace — strict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_batched_strict_equal_is_false(backend):
    a = BatchedSpace(make_box(backend, 0.0, 1.0, shape=(2,)), batch_shape=(3,))
    b = BatchedSpace(make_box(backend, 0.0, 1.0, shape=(2,)), batch_shape=(3,))
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_batched_strict_tighter_child_is_true(backend):
    a = BatchedSpace(make_box(backend, 0.1, 0.9, shape=(2,)), batch_shape=(3,))
    b = BatchedSpace(make_box(backend, 0.0, 1.0, shape=(2,)), batch_shape=(3,))
    assert a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_batched_strict_batch_shape_mismatch_is_false(backend):
    a = BatchedSpace(make_box(backend, 0.0, 1.0, shape=(2,)), batch_shape=(3,))
    b = BatchedSpace(make_box(backend, 0.0, 1.0, shape=(2,)), batch_shape=(4,))
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


# ---------------------------------------------------------------------------
# TextSpace — strict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_text_strict_equal_is_false(backend):
    a = TextSpace(backend, max_length=5, min_length=1, charset="abc")
    b = TextSpace(backend, max_length=5, min_length=1, charset="abc")
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_text_strict_narrower_is_true(backend):
    a = TextSpace(backend, max_length=5, min_length=1, charset="abc")
    b = TextSpace(backend, max_length=10, min_length=0, charset="abcdef")
    assert a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_text_strict_non_subspace_is_false(backend):
    a = TextSpace(backend, max_length=20, min_length=0, charset="abc")
    b = TextSpace(backend, max_length=10, min_length=0, charset="abcdef")
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


# ---------------------------------------------------------------------------
# UnionSpace — strict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_union_strict_equal_is_false(backend):
    a = UnionSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,))])
    b = UnionSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,))])
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_union_strict_tighter_alternatives_is_true(backend):
    a = UnionSpace(backend, [make_box(backend, 0.1, 0.9, shape=(2,)), make_box(backend, 0.0, 0.5, shape=(3,))])
    b = UnionSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,)), make_box(backend, 0.0, 1.0, shape=(3,))])
    assert a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_union_strict_count_mismatch_is_false(backend):
    a = UnionSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,))])
    b = UnionSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,)), make_box(backend, 0.0, 1.0, shape=(3,))])
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


# ---------------------------------------------------------------------------
# DynamicBoxSpace — strict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dynamic_box_strict_equal_is_false(backend):
    a = DynamicBoxSpace(
        backend, low=0.0, high=1.0, shape_low=(2, 3), shape_high=(2, 5),
        dtype=backend.default_floating_dtype,
    )
    b = DynamicBoxSpace(
        backend, low=0.0, high=1.0, shape_low=(2, 3), shape_high=(2, 5),
        dtype=backend.default_floating_dtype,
    )
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dynamic_box_strict_contained_shape_range_is_true(backend):
    a = DynamicBoxSpace(
        backend, low=0.1, high=0.9, shape_low=(2, 3), shape_high=(2, 5),
        dtype=backend.default_floating_dtype,
    )
    b = DynamicBoxSpace(
        backend, low=0.0, high=1.0, shape_low=(1, 3), shape_high=(3, 6),
        dtype=backend.default_floating_dtype,
    )
    assert a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dynamic_box_strict_shape_range_violation_is_false(backend):
    a = DynamicBoxSpace(
        backend, low=0.0, high=1.0, shape_low=(2, 3), shape_high=(2, 7),
        dtype=backend.default_floating_dtype,
    )
    b = DynamicBoxSpace(
        backend, low=0.0, high=1.0, shape_low=(1, 3), shape_high=(3, 6),
        dtype=backend.default_floating_dtype,
    )
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


# ---------------------------------------------------------------------------
# GraphSpace — strict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_graph_strict_equal_is_false(backend):
    a = _graph(backend, min_n=1, max_n=5, node_low=0.0, node_high=1.0)
    b = _graph(backend, min_n=1, max_n=5, node_low=0.0, node_high=1.0)
    assert not a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_graph_strict_tighter_is_true(backend):
    a = _graph(backend, min_n=2, max_n=4, node_low=0.1, node_high=0.9)
    b = _graph(backend, min_n=1, max_n=5, node_low=0.0, node_high=1.0)
    assert a.is_subspace(b)
    assert not b.is_subspace(a)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_graph_strict_count_range_violation_is_false(backend):
    a = _graph(backend, min_n=1, max_n=6, node_low=0.0, node_high=1.0)
    b = _graph(backend, min_n=1, max_n=5, node_low=0.0, node_high=1.0)
    # a's max_nodes=6 exceeds b's max_nodes=5: a ⊄ b, so a ⊂ b is False.
    assert not a.is_subspace(b)
    # b ⊆ a (True) and a ⊆ b (False) -> b ⊂ a is True.
    assert b.is_subspace(a)


# ---------------------------------------------------------------------------
# Cross-type comparisons — strict must return False (NOT raise)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_cross_type_strict_returns_false(backend):
    box = make_box(backend, 0.0, 1.0, shape=(2,))
    d = DictSpace(backend, {"a": make_box(backend, 0.0, 1.0, shape=(2,))})
    t = TupleSpace(backend, [make_box(backend, 0.0, 1.0, shape=(2,))])
    bin_ = BinarySpace(backend, shape=(2,))
    txt = TextSpace(backend, max_length=5)
    spaces = [box, d, t, bin_, txt]
    for i, s1 in enumerate(spaces):
        for j, s2 in enumerate(spaces):
            if i == j:
                continue
            assert not s1.is_subspace(s2), (
                f"{type(s1)} unexpectedly strict-subspace of {type(s2)}"
            )


# ---------------------------------------------------------------------------
# Composition: a.is_subspace(b) implies a.is_subspaceeq(b)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_strict_implies_nonstrict_box(backend):
    a = make_box(backend, 0.1, 0.9, shape=(4,))
    b = make_box(backend, 0.0, 1.0, shape=(4,))
    assert a.is_subspace(b)
    assert a.is_subspaceeq(b)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_strict_implies_nonstrict_dict(backend):
    a = DictSpace(backend, {"obs": make_box(backend, 0.0, 1.0, shape=(2,))})
    b = DictSpace(
        backend,
        {
            "obs": make_box(backend, 0.0, 1.0, shape=(2,)),
            "extra": make_box(backend, -1.0, 1.0, shape=(3,)),
        },
    )
    assert a.is_subspace(b)
    assert a.is_subspaceeq(b)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_strict_implies_nonstrict_text(backend):
    a = TextSpace(backend, max_length=5, min_length=1, charset="abc")
    b = TextSpace(backend, max_length=10, min_length=0, charset="abcdef")
    assert a.is_subspace(b)
    assert a.is_subspaceeq(b)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_strict_implies_nonstrict_dynamic_box(backend):
    a = DynamicBoxSpace(
        backend, low=0.1, high=0.9, shape_low=(2, 3), shape_high=(2, 5),
        dtype=backend.default_floating_dtype,
    )
    b = DynamicBoxSpace(
        backend, low=0.0, high=1.0, shape_low=(1, 3), shape_high=(3, 6),
        dtype=backend.default_floating_dtype,
    )
    assert a.is_subspace(b)
    assert a.is_subspaceeq(b)


# ---------------------------------------------------------------------------
# Consistency property for strict: a.is_subspace(b) => b.contains(a.sample())
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_strict_consistency_box(backend, seed):
    rng = _rng(backend, seed)
    a = make_box(backend, 0.1, 0.9, shape=(4,))
    b = make_box(backend, 0.0, 1.0, shape=(4,))
    assert a.is_subspace(b)
    for _ in range(5):
        rng, sample = a.sample(rng)
        assert b.contains(sample)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_strict_consistency_dict(backend, seed):
    rng = _rng(backend, seed)
    a = DictSpace(backend, {"obs": make_box(backend, 0.1, 0.9, shape=(2,))})
    b = DictSpace(
        backend,
        {
            "obs": make_box(backend, 0.0, 1.0, shape=(2,)),
            "extra": make_box(backend, -1.0, 1.0, shape=(3,)),
        },
    )
    assert a.is_subspace(b)
    for _ in range(5):
        rng, sample = a.sample(rng)
        # b.contains demands exact key equality, so check the shared keys
        # directly: every key of `a` is in `b` and the value is contained.
        assert all(k in b.spaces and b.spaces[k].contains(v) for k, v in sample.items())


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_strict_consistency_text(backend, seed):
    rng = _rng(backend, seed)
    a = TextSpace(backend, max_length=5, min_length=1, charset="abc")
    b = TextSpace(backend, max_length=10, min_length=0, charset="abcdef")
    assert a.is_subspace(b)
    for _ in range(5):
        rng, sample = a.sample(rng)
        assert b.contains(sample)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_strict_consistency_dynamic_box(backend, seed):
    rng = _rng(backend, seed)
    a = DynamicBoxSpace(
        backend, low=0.1, high=0.9, shape_low=(2, 3), shape_high=(2, 5),
        dtype=backend.default_floating_dtype,
    )
    b = DynamicBoxSpace(
        backend, low=0.0, high=1.0, shape_low=(1, 3), shape_high=(3, 6),
        dtype=backend.default_floating_dtype,
    )
    assert a.is_subspace(b)
    for _ in range(5):
        rng, sample = a.sample(rng)
        assert b.contains(sample)