"""Unit tests for ``GraphSpace`` / ``GraphInstance``."""

import gymnasium as gym
import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, GraphInstance, GraphSpace
from unienv_interface.space.space_utils import gym_utils as sgu
from tests_common.backends import cpu_backend_params
from tests_common.trees import assert_tree_equal


def _node_space(backend=B, device=None):
    return BoxSpace(backend, 0.0, 1.0, backend.default_floating_dtype, shape=(2,), device=device)


def _edge_space(backend=B, device=None):
    return BoxSpace(backend, -1.0, 1.0, backend.default_floating_dtype, shape=(3,), device=device)


def _rng(seed=0, backend=B, device=None):
    return backend.random.random_number_generator(seed, device=device)


def test_default_max_nodes_and_edges_are_unbounded():
    space = GraphSpace(B, _node_space())
    assert space.max_nodes is None and space.max_edges is None
    edge_space = GraphSpace(B, _node_space(), _edge_space(), is_edge=True)
    assert edge_space.max_edges is None
    for seed in range(5):
        _, x = space.sample(_rng(seed))
        assert space.contains(x)
        _, y = edge_space.sample(_rng(seed))
        assert edge_space.contains(y)


def test_invalid_count_ranges_raise():
    with pytest.raises(AssertionError):
        GraphSpace(B, _node_space(), min_nodes=3, max_nodes=2)
    with pytest.raises(AssertionError):
        GraphSpace(B, _node_space(), is_edge=True, min_edges=3, max_edges=2, max_nodes=5)


def test_invalid_edge_feature_space_message():
    with pytest.raises(AssertionError, match="edge_space should be instances of BoxSpace"):
        GraphSpace(B, _node_space(), edge_feature_space=object(), is_edge=True)


@pytest.mark.parametrize("is_edge", [False, True])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 2)])
def test_sample_contains_and_has_node_features(is_edge, batch_shape):
    space = GraphSpace(
        B, _node_space(), _edge_space() if is_edge else None, is_edge=is_edge,
        min_nodes=2, max_nodes=6, min_edges=1, max_edges=5, batch_shape=batch_shape,
    )
    for seed in range(5):
        _, x = space.sample(_rng(seed))
        assert x.nodes_features is not None
        assert x.n_nodes.shape == batch_shape
        assert x.nodes_features.shape == batch_shape + (int(np.max(x.n_nodes)), 2)
        if is_edge:
            assert x.n_edges.shape == batch_shape
            assert x.edges.shape == batch_shape + (int(np.max(x.n_edges)), 2)
            # Every edge endpoint must be a valid node index of its own graph.
            n_nodes = np.broadcast_to(np.asarray(x.n_nodes)[..., None, None], x.edges.shape)
            assert np.all((x.edges >= 0) & (x.edges < n_nodes))
        else:
            assert x.n_edges is None and x.edges is None
        assert space.contains(x)


def test_contains_rejects_invalid_instances():
    space = GraphSpace(B, _node_space(), min_nodes=1, max_nodes=4)
    _, x = space.sample(_rng())
    assert not space.contains("graph")
    assert not space.contains(GraphInstance(n_nodes=np.array(9), nodes_features=x.nodes_features))
    assert not space.contains(GraphInstance(n_nodes=x.n_nodes, nodes_features=None))
    assert not space.contains(GraphInstance(n_nodes=x.n_nodes, nodes_features=x.nodes_features, n_edges=np.array(1)))


def test_equality_considers_batch_shape():
    a = GraphSpace(B, _node_space(), max_nodes=4)
    b = GraphSpace(B, _node_space(), max_nodes=4, batch_shape=(2,))
    assert a == GraphSpace(B, _node_space(), max_nodes=4)
    assert a != b
    assert not a.is_subspaceeq(b) and not b.is_subspaceeq(a)


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_to_and_data_to_roundtrip(backend, device):
    space = GraphSpace(B, _node_space(), _edge_space(), is_edge=True, max_nodes=5, max_edges=4)
    moved = space.to(backend, device)
    assert isinstance(moved, GraphSpace)
    assert moved.max_nodes == 5 and moved.max_edges == 4 and moved.is_edge
    assert moved.node_feature_space.backend is backend
    _, x = space.sample(_rng(3))
    y = space.data_to(x, backend, device)
    assert isinstance(y, GraphInstance)
    assert moved.contains(y)
    # jax without x64 narrows 64-bit dtypes to 32 bits
    narrowing = backend.simplified_name == "jax"
    assert_tree_equal(moved.data_to(y, B), x, check_dtype=not narrowing, rtol=1e-6 if narrowing else 0.0)
    # Same-backend device move keeps the node feature space too.
    assert space.to(device="cpu").node_feature_space == space.node_feature_space


def test_from_gym_space_graph_defaults():
    no_edges = sgu.from_gym_space(gym.spaces.Graph(gym.spaces.Box(0, 1, (2,)), None), B)
    assert isinstance(no_edges, GraphSpace) and not no_edges.is_edge and no_edges.max_nodes is None
    with_edges = sgu.from_gym_space(gym.spaces.Graph(gym.spaces.Box(0, 1, (2,)), gym.spaces.Box(0, 1, (3,))), B)
    assert with_edges.is_edge and with_edges.edge_feature_space is not None


@pytest.mark.parametrize("with_edge_features", [False, True])
def test_gym_data_roundtrip(with_edge_features):
    space = GraphSpace(
        B, BoxSpace(B, 0.0, 1.0, np.float32, shape=(2,)),
        BoxSpace(B, 0.0, 1.0, np.float32, shape=(3,)) if with_edge_features else None,
        is_edge=True, max_nodes=5, max_edges=6,
    )
    gym_space = sgu.to_gym_space(space)
    for seed in range(5):
        _, x = space.sample(_rng(seed))
        gx = sgu.to_gym_data(space, x)
        assert gym_space.contains(gx)
        if with_edge_features:
            back = sgu.from_gym_data(space, gx)
            assert space.contains(back)
            assert_tree_equal(back.nodes_features, x.nodes_features)
            assert_tree_equal(back.edges, x.edges, check_dtype=False)


def test_single_node_graph_without_edges():
    space = GraphSpace(B, _node_space(), max_nodes=1)
    _, x = space.sample(_rng())
    assert int(x.n_nodes) == 1 and space.contains(x)
    # With edges, a single node cannot hold the required edge (self-loops excluded)
    with pytest.raises(AssertionError):
        GraphSpace(B, _node_space(), is_edge=True, max_nodes=1)


@pytest.mark.parametrize("is_edge", [False, True])
def test_batching_graphs_of_different_sizes(is_edge):
    from unienv_interface.space.space_utils import batch_utils as sbu
    # Node features bounded away from 0, so zero padding would be detected
    space = GraphSpace(
        B, BoxSpace(B, 1.0, 2.0, np.float32, shape=(2,)),
        BoxSpace(B, 3.0, 4.0, np.float32, shape=(1,)) if is_edge else None,
        is_edge=is_edge, min_nodes=1, max_nodes=6, max_edges=8 if is_edge else None,
    )
    rng = _rng(4)
    items = []
    while len({int(i.n_nodes) for i in items}) < 2 or len(items) < 3:
        rng, x = space.sample(rng)
        items.append(x)
    batched = sbu.batch_space(space, len(items))
    batch = sbu.concatenate(space, items)
    assert batched.contains(batch)
    for i, x in enumerate(items):
        got = sbu.get_at(batched, batch, i)
        assert space.contains(got)
        assert_tree_equal(got, x)
    # set_at with a graph larger / smaller than the current padding
    rng, big = GraphSpace(B, space.node_feature_space, space.edge_feature_space, is_edge=is_edge,
                          min_nodes=6, max_nodes=6, min_edges=8 if is_edge else 1,
                          max_edges=8 if is_edge else None).sample(rng)
    updated = sbu.set_at(batched, batch, 0, big)
    assert batched.contains(updated)
    assert_tree_equal(sbu.get_at(batched, updated, 0), big)
    assert_tree_equal(sbu.get_at(batched, updated, 1), items[1])
    small = items[int(np.argmin([int(i.n_nodes) for i in items]))]
    updated = sbu.set_at(batched, updated, 0, small)
    assert batched.contains(updated)
    assert_tree_equal(sbu.get_at(batched, updated, 0), small)


def test_set_at_replacing_the_largest_graph_shrinks_padding():
    from unienv_interface.space.space_utils import batch_utils as sbu
    space = GraphSpace(B, _node_space(), max_nodes=6)
    small = GraphInstance(n_nodes=np.array(1), nodes_features=np.zeros((1, 2)))
    big = GraphInstance(n_nodes=np.array(6), nodes_features=np.zeros((6, 2)))
    batched = sbu.batch_space(space, 2)
    batch = sbu.concatenate(space, [small, big])
    assert batch.nodes_features.shape == (2, 6, 2)
    updated = sbu.set_at(batched, batch, 1, small)
    assert updated.nodes_features.shape == (2, 1, 2)
    assert batched.contains(updated)
