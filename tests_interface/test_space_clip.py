"""Tests for ``Space.clip``: projecting members of a superspace onto a subspace.

Contract: if ``target.is_subspaceeq(source)`` then for every ``x`` with
``source.contains(x)``, ``target.contains(target.clip(x))`` -- unless no
meaningful projection exists, in which case ``clip`` raises ``ValueError``.
``clip`` is idempotent on members of ``target`` and never mutates its input.
"""

import copy

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import (
    BatchedSpace, BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, GraphInstance, GraphSpace, Space,
    TextSpace, TupleSpace, UnionSpace,
)
from unienv_interface.space.space_utils import batch_utils as sbu
from tests_common.backends import cpu_backend_params
from tests_common.trees import assert_tree_equal, to_numpy


def _rng(space, seed):
    return space.backend.random.random_number_generator(seed, device=space.device)


def _samples(space, seed, n):
    rng = _rng(space, seed)
    out = []
    for _ in range(n):
        rng, x = space.sample(rng)
        out.append(x)
    return out


def _clip_fails(space, x):
    """Independent oracle: whether ``space.clip(x)`` has no projection (documented ``ValueError`` cases)."""
    if isinstance(space, DynamicBoxSpace):
        return any(dim < low for dim, low in zip(x.shape, space.shape_low))
    if isinstance(space, TextSpace):
        kept = x if space.charset is None else [c for c in x if c in space.charset]
        return len(kept[:space.max_length]) < space.min_length
    if isinstance(space, GraphSpace):
        n_nodes = to_numpy(x.n_nodes).reshape(-1)
        if np.any(n_nodes < space.min_nodes):
            return True
        if not space.is_edge:
            return False
        n_edges = to_numpy(x.n_edges).reshape(-1)
        edges = to_numpy(x.edges).reshape(n_nodes.size, -1, 2)
        for b in range(n_nodes.size):
            kept_nodes = n_nodes[b] if space.max_nodes is None else min(n_nodes[b], space.max_nodes)
            remaining = sum(
                1 for edge in edges[b, :n_edges[b]] if not any(kept_nodes <= v < n_nodes[b] for v in edge)
            )
            if space.max_edges is not None:
                remaining = min(remaining, space.max_edges)
            if remaining < space.min_edges:
                return True
        return False
    if isinstance(space, DictSpace):
        return any(_clip_fails(sub, x[key]) for key, sub in space.spaces.items())
    if isinstance(space, TupleSpace):
        return any(_clip_fails(sub, part) for sub, part in zip(space.spaces, x))
    if isinstance(space, UnionSpace):
        return _clip_fails(space.spaces[x[0]], x[1])
    if isinstance(space, BatchedSpace):
        return any(_clip_fails(space.single_space, xi) for xi in x.flat)
    return False


def _check_clip_contract(target, source, seed, n=3):
    assert target.is_subspaceeq(source)
    for x in _samples(source, seed, n):
        assert source.contains(x)
        snapshot = copy.deepcopy(x)
        if _clip_fails(target, x):
            with pytest.raises(ValueError):
                target.clip(x)
        else:
            y = target.clip(x)
            assert target.contains(y), (target, x, y)
            assert_tree_equal(target.clip(y), y)
        assert_tree_equal(x, snapshot)  # never mutated
    for y in _samples(target, seed, n):
        snapshot = copy.deepcopy(y)
        assert_tree_equal(target.clip(y), y)  # identity on members
        assert_tree_equal(y, snapshot)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from tests_common.space_strategies import build_space, narrowed_specs, seeds, space_specs  # noqa: E402


@given(st.data())
def test_clip_projects_superspace_members(data):
    spec = data.draw(space_specs())
    target_spec = data.draw(narrowed_specs(spec))
    _check_clip_contract(build_space(target_spec, NB), build_space(spec, NB), data.draw(seeds))


@given(st.data())
def test_clip_onto_itself_is_identity(data):
    spec = data.draw(space_specs())
    space = build_space(spec, NB)
    for x in _samples(space, data.draw(seeds), 2):
        assert_tree_equal(space.clip(x), x)


_cross_kinds = ("box", "binary", "dynbox", "text", "graph", "dict", "tuple", "union", "batched")


@pytest.mark.parametrize("backend,device", cpu_backend_params())
@settings(max_examples=10)
@given(data=st.data())
def test_clip_contract_on_every_backend(backend, device, data):
    spec = data.draw(space_specs(kinds=_cross_kinds, max_leaves=3))
    target_spec = data.draw(narrowed_specs(spec))
    _check_clip_contract(
        build_space(target_spec, backend, device), build_space(spec, backend, device), data.draw(seeds), n=2
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

def test_base_space_clip_raises_not_implemented():
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

        def get_repr(self, abbreviate=False, include_backend=True, include_device=True, include_dtype=True):
            return "DummySpace"

        def data_to(self, data, backend=None, device=None):
            return data

    with pytest.raises(NotImplementedError):
        DummySpace(backend=NB).clip(None)


# ---------------------------------------------------------------------------
# BoxSpace
# ---------------------------------------------------------------------------

def _arr(backend, device, values, dtype):
    return backend.from_numpy(np.asarray(values, dtype=dtype), device=device)


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_box_clip_float_with_infinite_bounds(backend, device):
    target = BoxSpace(backend, _arr(backend, device, [-np.inf, 0.0, -1.0], np.float32),
                      _arr(backend, device, [1.0, np.inf, np.inf], np.float32), backend.float32, device=device)
    x = _arr(backend, device, [-1e30, -5.0, 1e30], np.float32)
    y = target.clip(x)
    np.testing.assert_array_equal(to_numpy(y), np.array([-1e30, 0.0, 1e30], np.float32))
    assert target.contains(y)
    np.testing.assert_array_equal(to_numpy(x), np.array([-1e30, -5.0, 1e30], np.float32))


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_box_clip_integer_dtype(backend, device):
    target = BoxSpace(backend, 0, 10, backend.int32, device=device, shape=(4,))
    source = BoxSpace(backend, -100, 100, backend.int32, device=device, shape=(4,))
    x = _arr(backend, device, [-5, 3, 10, 99], np.int32)
    assert source.contains(x) and target.is_subspaceeq(source)
    y = target.clip(x)
    np.testing.assert_array_equal(to_numpy(y), [0, 3, 10, 10])
    assert y.dtype == x.dtype and target.contains(y)
    # Full-range integer boxes (built from +-inf) never change members
    full = BoxSpace(backend, -np.inf, np.inf, backend.int32, device=device, shape=(4,))
    np.testing.assert_array_equal(to_numpy(full.clip(x)), to_numpy(x))


def test_box_clip_casts_to_the_box_dtype():
    target = BoxSpace(NB, 0, 10, np.int64, shape=(3,))
    x = np.array([-3, 5, 100], dtype=np.int8)
    y = target.clip(x)
    assert y.dtype == np.int64
    np.testing.assert_array_equal(y, [0, 5, 10])
    flags = np.array([True, False, True])
    assert target.contains(flags)
    assert target.clip(flags).dtype == np.int64


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_box_clip_zero_dim_returns_array(backend, device):
    target = BoxSpace(backend, 0.0, 1.0, backend.float32, device=device, shape=())
    y = target.clip(_arr(backend, device, 2.0, np.float32))
    assert backend.is_backendarray(y) and tuple(y.shape) == ()
    assert target.contains(y) and float(y) == 1.0


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_box_clip_elementwise_and_batched_bounds(backend, device):
    single = BoxSpace(backend, _arr(backend, device, [0.0, -1.0], np.float32),
                      _arr(backend, device, [1.0, 0.0], np.float32), backend.float32, device=device)
    batched = sbu.batch_space(single, 3)
    x = _arr(backend, device, [[2.0, 2.0], [-2.0, -2.0], [0.5, -0.5]], np.float32)
    y = batched.clip(x)
    np.testing.assert_array_equal(to_numpy(y), [[1.0, 0.0], [0.0, -1.0], [0.5, -0.5]])
    assert batched.contains(y)


def test_box_clip_returns_a_new_array():
    space = BoxSpace(NB, -1.0, 1.0, np.float32, shape=(3,))
    x = np.zeros(3, np.float32)
    y = space.clip(x)
    y[0] = 5.0
    assert x[0] == 0.0


# ---------------------------------------------------------------------------
# BinarySpace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_binary_clip_is_identity(backend, device):
    space = BinarySpace(backend, (2, 3), device=device)
    _, x = space.sample(_rng(space, 0))
    assert space.clip(x) is x
    with pytest.raises(ValueError):
        space.clip(backend.from_numpy(np.zeros((3,), bool), device=device))


# ---------------------------------------------------------------------------
# DynamicBoxSpace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_dynamic_box_clip_trims_and_clamps(backend, device):
    source = DynamicBoxSpace(backend, -10.0, 10.0, (0, 3), (6, 3), backend.float32, device=device)
    target = DynamicBoxSpace(backend, 0.0, 1.0, (1, 3), (2, 3), backend.float32, device=device)
    assert target.is_subspaceeq(source)
    x = _arr(backend, device, np.arange(15).reshape(5, 3) - 7.0, np.float32)
    y = target.clip(x)
    assert tuple(y.shape) == (2, 3) and target.contains(y)
    np.testing.assert_array_equal(to_numpy(y), np.clip(np.arange(6).reshape(2, 3) - 7.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="shorter"):
        target.clip(_arr(backend, device, np.zeros((0, 3)), np.float32))
    with pytest.raises(ValueError, match="dimensions"):
        target.clip(_arr(backend, device, np.zeros((2,)), np.float32))


def test_dynamic_box_clip_zero_length_axis_and_integers():
    target = DynamicBoxSpace(NB, 0, 5, (0,), (3,), np.int32)
    empty = np.zeros((0,), np.int32)
    assert target.contains(empty)
    y = target.clip(empty)
    assert y.shape == (0,) and target.contains(y)
    y = target.clip(np.array([-1, 9, 2, 7], np.int16))
    assert y.dtype == np.int32
    np.testing.assert_array_equal(y, [0, 5, 2])


def test_dynamic_box_clip_trims_a_static_target_axis():
    # The target's axis 1 is static (size 2) while the source's is dynamic
    source = DynamicBoxSpace(NB, -5.0, 5.0, (1, 0), (3, 4), np.float32)
    target = DynamicBoxSpace(NB, np.array([[0.0, 1.0]]), np.array([[1.0, 2.0]]), (1, 2), (3, 2), np.float32)
    assert target.is_subspaceeq(source)
    y = target.clip(np.full((2, 4), 1.5, np.float32))
    np.testing.assert_array_equal(y, [[1.0, 1.5], [1.0, 1.5]])
    assert target.contains(y)


# ---------------------------------------------------------------------------
# DictSpace / TupleSpace / UnionSpace / BatchedSpace
# ---------------------------------------------------------------------------

def _box(low, high, shape=(2,)):
    return BoxSpace(NB, low, high, np.float32, shape=shape)


def test_dict_clip_drops_extra_keys_and_recurses():
    source = DictSpace(NB, {"a": _box(-5, 5), "b": DictSpace(NB, {"c": _box(-5, 5), "d": _box(-5, 5)}), "e": _box(-5, 5)})
    target = DictSpace(NB, {"a": _box(0, 1), "b": DictSpace(NB, {"c": _box(0, 1)})})
    assert target.is_subspaceeq(source)
    x = {"a": np.full(2, 3.0, np.float32), "b": {"c": np.full(2, -3.0, np.float32), "d": np.zeros(2, np.float32)},
         "e": np.zeros(2, np.float32)}
    y = target.clip(x)
    assert set(y) == {"a", "b"} and set(y["b"]) == {"c"}
    assert target.contains(y)
    assert set(x) == {"a", "b", "e"} and set(x["b"]) == {"c", "d"}  # input untouched
    with pytest.raises(ValueError, match="missing"):
        target.clip({"a": x["a"]})
    with pytest.raises(ValueError):
        target.clip([1, 2])


def test_empty_dict_clip():
    target = DictSpace(NB, {})
    source = DictSpace(NB, {"a": _box(0, 1)})
    assert target.is_subspaceeq(source)
    assert target.clip({"a": np.zeros(2, np.float32)}) == {}
    assert target.contains(target.clip({}))


def test_tuple_clip_is_elementwise():
    target = TupleSpace(NB, [_box(0, 1), TextSpace(NB, 2, charset="ab")])
    y = target.clip((np.array([-1.0, 2.0], np.float32), "abcab"))
    np.testing.assert_array_equal(y[0], [0.0, 1.0])
    assert y[1] == "ab" and target.contains(y)
    with pytest.raises(ValueError):
        target.clip((np.zeros(2, np.float32),))


def test_union_clip_uses_the_alternative_at_the_index():
    target = UnionSpace(NB, [_box(0, 1), TextSpace(NB, 3, charset="xy")])
    source = UnionSpace(NB, [_box(-5, 5), TextSpace(NB, 5)])
    assert target.is_subspaceeq(source)
    y = target.clip((0, np.full(2, 4.0, np.float32)))
    assert y[0] == 0 and target.contains(y)
    np.testing.assert_array_equal(y[1], [1.0, 1.0])
    assert target.clip((1, "xaybyx")) == (1, "xyy")
    with pytest.raises(ValueError):
        target.clip((2, "x"))
    with pytest.raises(ValueError):
        target.clip("x")


def test_union_clip_keeps_a_numpy_integer_index():
    space = UnionSpace(NB, [_box(0, 1), TextSpace(NB, 3)])
    assert space.clip((np.int32(1), "ab")) == (1, "ab")


def test_batched_space_clip_is_elementwise():
    target = BatchedSpace(TextSpace(NB, 2, min_length=1, charset="ab"), (2, 2))
    source = BatchedSpace(TextSpace(NB, 4), (2, 2))
    assert target.is_subspaceeq(source)
    x = np.empty((2, 2), dtype=object)
    for i, s in enumerate(["abab", "ba", "aXb", "b"]):
        x.flat[i] = s
    y = target.clip(x)
    assert y.dtype == object and y.shape == (2, 2) and list(y.flat) == ["ab", "ba", "ab", "b"]
    assert target.contains(y) and list(x.flat) == ["abab", "ba", "aXb", "b"]
    x.flat[3] = "XY"
    with pytest.raises(ValueError, match="min_length"):
        target.clip(x)
    with pytest.raises(ValueError):
        target.clip(np.empty((2,), dtype=object))


# ---------------------------------------------------------------------------
# TextSpace
# ---------------------------------------------------------------------------

def test_text_clip():
    target = TextSpace(NB, 3, min_length=2, charset="abc")
    assert target.clip("aXbYcZa") == "abc"
    assert target.clip("ab") == "ab"
    with pytest.raises(ValueError, match="min_length"):
        target.clip("aXYZ")
    with pytest.raises(ValueError):
        target.clip(5)
    free = TextSpace(NB, 2)
    assert free.clip("hello") == "he"
    assert free.clip("") == ""


# ---------------------------------------------------------------------------
# GraphSpace
# ---------------------------------------------------------------------------

def _graph_space(backend=NB, device=None, edge_features=True, batch_shape=(), wide=False, **counts):
    bound = 10.0 if wide else 1.0
    node = BoxSpace(backend, -bound if wide else 0.0, bound, backend.float32, shape=(2,), device=device)
    edge = BoxSpace(backend, -bound, bound, backend.float32, shape=(), device=device) if edge_features else None
    return GraphSpace(backend, node, edge, is_edge=True, batch_shape=batch_shape, device=device, **counts)


def _graph(backend, device, n_nodes, n_edges, nodes, edges, edge_features):
    return GraphInstance(
        n_nodes=_arr(backend, device, n_nodes, np.int64),
        n_edges=_arr(backend, device, n_edges, np.int64),
        nodes_features=_arr(backend, device, nodes, np.float32),
        edges=_arr(backend, device, edges, np.int64),
        edges_features=_arr(backend, device, edge_features, np.float32),
    )


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_graph_clip_drops_nodes_and_their_edges(backend, device):
    source = _graph_space(backend, device, wide=True, max_nodes=5, max_edges=6)
    target = _graph_space(backend, device, max_nodes=3, max_edges=2)
    assert target.is_subspaceeq(source)
    nodes = np.linspace(-1, 2, 10).reshape(5, 2)
    edges = [[0, 4], [1, 2], [3, 0], [2, 0], [0, 1]]
    features = [0.1, 0.2, 0.3, 0.4, 5.0]
    x = _graph(backend, device, 5, 5, nodes, edges, features)
    assert source.contains(x)
    snapshot = copy.deepcopy(x)
    y = target.clip(x)
    assert target.contains(y)
    assert int(y.n_nodes) == 3 and int(y.n_edges) == 2
    np.testing.assert_array_equal(to_numpy(y.nodes_features), np.clip(nodes[:3], 0.0, 1.0).astype(np.float32))
    # [0, 4] and [3, 0] touch dropped nodes; of the rest the first two (max_edges) are kept
    np.testing.assert_array_equal(to_numpy(y.edges), [[1, 2], [2, 0]])
    np.testing.assert_array_equal(to_numpy(y.edges_features), np.array([0.2, 0.4], np.float32))
    assert_tree_equal(x, snapshot)
    assert_tree_equal(target.clip(y), y)


def test_graph_clip_raises_when_too_small():
    target = _graph_space(min_nodes=2, max_nodes=3, min_edges=2)
    nodes = np.full((4, 2), 0.5)
    with pytest.raises(ValueError, match="min_nodes"):
        target.clip(_graph(NB, None, 1, 2, nodes[:1], [[0, 0], [0, 0]], [0.0, 0.0]))
    # Only one edge survives once node 3 is dropped
    with pytest.raises(ValueError, match="min_edges"):
        target.clip(_graph(NB, None, 4, 2, nodes, [[0, 3], [1, 2]], [0.0, 0.0]))


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_graph_clip_batched(backend, device):
    source = _graph_space(backend, device, batch_shape=(2,), wide=True, max_nodes=4)
    target = _graph_space(backend, device, batch_shape=(2,), max_nodes=2, max_edges=2)
    assert target.is_subspaceeq(source)
    nodes = np.full((2, 4, 2), 0.5)
    edges = [[[0, 1], [1, 0], [0, 1], [0, 0]],    # graph 0: 3 edges, 2 nodes -> capped at 2 edges
             [[3, 0], [1, 0], [2, 2], [0, 1]]]    # graph 1: 4 edges, 4 nodes -> [1, 0], [0, 1] survive
    features = [[0.1, 0.2, 0.3, -9.0], [0.4, 0.5, 0.6, 0.7]]
    x = _graph(backend, device, [2, 4], [3, 4], nodes, edges, features)
    assert source.contains(x)
    y = target.clip(x)
    assert target.contains(y)
    np.testing.assert_array_equal(to_numpy(y.n_nodes), [2, 2])
    np.testing.assert_array_equal(to_numpy(y.n_edges), [2, 2])
    assert tuple(y.nodes_features.shape) == (2, 2, 2)
    np.testing.assert_array_equal(to_numpy(y.edges), [[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
    np.testing.assert_array_equal(to_numpy(y.edges_features), np.array([[0.1, 0.2], [0.5, 0.7]], np.float32))


def test_graph_clip_without_edges_or_edge_features():
    node = BoxSpace(NB, 0.0, 1.0, np.float32, shape=())
    target = GraphSpace(NB, node, max_nodes=2)
    source = GraphSpace(NB, BoxSpace(NB, -1.0, 1.0, np.float32, shape=()), max_nodes=4)
    assert target.is_subspaceeq(source)
    x = GraphInstance(n_nodes=np.asarray(4), nodes_features=np.array([-1.0, 0.5, 1.0, 0.0], np.float32))
    y = target.clip(x)
    assert target.contains(y) and int(y.n_nodes) == 2 and y.edges is None
    np.testing.assert_array_equal(y.nodes_features, [0.0, 0.5])

    no_features = _graph_space(edge_features=False, max_nodes=3)
    g = _graph(NB, None, 2, 1, np.full((2, 2), 0.5), [[0, 1]], [0.5])
    assert no_features.clip(g).edges_features is None
