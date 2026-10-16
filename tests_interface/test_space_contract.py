"""Property-based contract tests that every space type must satisfy.

Spaces are drawn as backend-independent specs (``tests_common.space_strategies``);
data always comes from ``space.sample`` with a drawn seed. The full property set
runs on numpy; a smaller set runs on one CPU device of every backend.
"""

import json

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings, strategies as st

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import BatchedSpace, DynamicBoxSpace, GraphSpace
from unienv_interface.space.space_utils import (
    batch_utils as sbu, construct_utils as scu, flatten_utils as sfu, gym_utils as sgu,
    serialization_utils as ssu,
)
from tests_common.backends import cpu_backend_params
from tests_common.space_strategies import (
    ALL_KINDS, all_specs, build_space, kinds_in, seeds, space_specs, widened_specs,
)
from tests_common.trees import assert_tree_equal

FLATTENABLE_KINDS = ("box", "binary", "dynbox", "text", "dict", "tuple", "union")
# DynamicBox batches are padded with fill_value, so element-wise batch identities
# only hold for spaces without dynamic boxes (padding is covered separately).
BATCHABLE_KINDS = ("box", "binary", "text", "graph", "dict", "tuple", "union", "batched")
ARRAY_BATCH_KINDS = ("box", "binary", "graph", "dict", "tuple")
CONSTRUCTIBLE_KINDS = ("box", "binary", "dynbox", "text", "dict", "tuple")
GYM_KINDS = ("box", "binary", "text", "graph", "dict", "tuple", "union")


def _rng(space, seed):
    return space.backend.random.random_number_generator(seed, device=space.device)


def _samples(space, seed, n):
    rng = _rng(space, seed)
    out = []
    for _ in range(n):
        rng, x = space.sample(rng)
        out.append(x)
    return out


# ---------------------------------------------------------------------------
# 1. sampling
# ---------------------------------------------------------------------------

@given(space_specs(), seeds)
def test_sample_is_contained_and_deterministic(spec, seed):
    space = build_space(spec, NB)
    first = _samples(space, seed, 3)
    for x in first:
        assert space.contains(x)
        assert x in space
    assert_tree_equal(_samples(space, seed, 3), first)
    # is_bounded / repr never raise
    for manner in ("both", "below", "above"):
        assert isinstance(space.is_bounded(manner), bool)
    assert isinstance(repr(space), str) and isinstance(str(space), str)


@given(space_specs(kinds=[k for k in ALL_KINDS if k != "graph"]))
def test_create_empty_has_the_right_structure(spec):
    space = build_space(spec, NB)
    empty = space.create_empty()
    sample = space.sample(_rng(space, 0))[1]
    _assert_same_structure(empty, sample, space)


def _assert_same_structure(a, b, space):
    if isinstance(b, dict):
        assert isinstance(a, dict) and set(a) == set(b)
        for k in b:
            _assert_same_structure(a[k], b[k], space.spaces[k])
    elif isinstance(space, BatchedSpace):
        assert isinstance(a, np.ndarray) and a.dtype == object and a.shape == b.shape
    elif isinstance(b, tuple):
        assert isinstance(a, tuple) and len(a) == len(b)
        if type(space).__name__ == "UnionSpace":
            return
        for x, y, s in zip(a, b, space.spaces):
            _assert_same_structure(x, y, s)
    elif isinstance(space, DynamicBoxSpace):
        assert tuple(a.shape) == space.shape_high and a.dtype == b.dtype
    elif hasattr(b, "shape"):
        assert tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype
    else:
        assert type(a) is type(b)


# ---------------------------------------------------------------------------
# 3. JSON serialisation
# ---------------------------------------------------------------------------

@given(space_specs())
def test_json_roundtrip(spec):
    space = build_space(spec, NB)
    payload = ssu.space_to_json(space)
    assert ssu.json_to_space(payload, NB, None) == space
    assert ssu.json_to_space(json.loads(json.dumps(payload)), NB, None) == space


# ---------------------------------------------------------------------------
# 4. flattening
# ---------------------------------------------------------------------------

def _check_flatten(space, x, start_dim=0):
    assert sfu.is_flattenable(space, start_dim)
    flat_space = sfu.flatten_space(space, start_dim)
    flat = sfu.flatten_data(space, x, start_dim)
    assert flat.shape[-1] == sfu.flat_dim(space, start_dim)
    assert flat_space.contains(flat), (flat_space, flat)
    assert_tree_equal(sfu.unflatten_data(space, flat, start_dim), x)


@given(space_specs(kinds=FLATTENABLE_KINDS, text_kwargs={"allow_free_charset": False}), seeds)
def test_flatten_roundtrip(spec, seed):
    space = build_space(spec, NB)
    for x in _samples(space, seed, 2):
        _check_flatten(space, x)


@given(space_specs(kinds=("box", "binary", "dynbox", "dict", "tuple")), seeds, st.integers(1, 3))
def test_flatten_batched_from_dim_one(spec, seed, n):
    batched = sbu.batch_space(build_space(spec, NB), n)
    for x in _samples(batched, seed, 2):
        _check_flatten(batched, x, start_dim=1)


@given(space_specs(kinds=("graph", "batched", "text", "dict", "tuple")))
def test_non_flattenable_spaces_report_it(spec):
    space = build_space(spec, NB)
    unflattenable = any(
        s["kind"] in ("graph", "batched") or (s["kind"] == "text" and s["charset"] is None)
        for s in all_specs(spec)
    )
    assume(unflattenable)
    assert not sfu.is_flattenable(space)
    assert sfu.flat_dim(space) is None


# ---------------------------------------------------------------------------
# 5. batching
# ---------------------------------------------------------------------------

@given(space_specs(kinds=BATCHABLE_KINDS), seeds, st.integers(1, 3))
def test_batch_contract(spec, seed, n):
    space = build_space(spec, NB)
    items = _samples(space, seed, n + 1)
    items, extra = items[:n], items[n]
    batched_space = sbu.batch_space(space, n)
    batch = sbu.concatenate(space, items)
    assert batched_space.contains(batch)
    assert_tree_equal(sbu.concatenate(batched_space, items), batch)
    for i, x in enumerate(items):
        assert_tree_equal(sbu.get_at(batched_space, batch, i), x)
    iterated = list(sbu.iterate(batched_space, batch))
    assert len(iterated) == n
    for got, x in zip(iterated, items):
        assert_tree_equal(got, x)
    unbatched = list(sbu.unbatch_spaces(batched_space))
    assert len(unbatched) == n and all(u == space for u in unbatched)
    assert sbu.batch_size(batched_space) == n
    updated = sbu.set_at(batched_space, batch, n - 1, extra)
    assert batched_space.contains(updated)
    assert_tree_equal(sbu.get_at(batched_space, updated, n - 1), extra)
    if n > 1:
        assert_tree_equal(sbu.get_at(batched_space, updated, 0), items[0])
    if kinds_in(spec) <= {"box", "binary", "dict", "tuple"}:
        # Batching copies of the same space agrees with batch_space
        assert sbu.batch_differing_spaces([space] * n) == batched_space


@given(space_specs(kinds=ARRAY_BATCH_KINDS + ("batched", "text")), seeds, st.integers(1, 3), st.integers(1, 3))
def test_swap_and_reshape_batch_dims(spec, seed, n, m):
    space = build_space(spec, NB)
    batched = sbu.batch_space(sbu.batch_space(space, m), n)  # batch dims (n, m)
    (x,) = _samples(batched, seed, 1)
    assert batched.contains(x)

    swapped = sbu.swap_batch_dims(batched, 0, 1)
    swapped_x = sbu.swap_batch_dims_in_data(NB, x, 0, 1)
    assert swapped.contains(swapped_x)
    assert sbu.swap_batch_dims(swapped, 0, 1) == batched
    assert_tree_equal(sbu.swap_batch_dims_in_data(NB, swapped_x, 0, 1), x)

    flat = sbu.reshape_batch_size(batched, (n, m), (n * m,))
    flat_x = sbu.reshape_batch_size_in_data(NB, x, (n, m), (n * m,))
    assert flat.contains(flat_x)
    assert sbu.reshape_batch_size(flat, (n * m,), (n, m)) == batched
    assert_tree_equal(sbu.reshape_batch_size_in_data(NB, flat_x, (n * m,), (n, m)), x)


@given(space_specs(kinds=("dynbox",)), seeds, st.integers(2, 4))
def test_dynamic_box_batches_are_padded_with_fill_value(spec, seed, n):
    space = build_space(spec, NB)
    items = _samples(space, seed, n)
    batched_space = sbu.batch_space(space, n)
    batch = sbu.concatenate(space, items)
    assert batch.shape[0] == n and batched_space.shape_contains(batch.shape)
    for i, x in enumerate(items):
        assert_tree_equal(space.unpad_data(sbu.get_at(batched_space, batch, i)), x)


# ---------------------------------------------------------------------------
# 6. gymnasium
# ---------------------------------------------------------------------------

def _gym_convertible(spec):
    for s in all_specs(spec):
        if s["kind"] == "text" and s["charset"] is None:
            return False
        if s["kind"] == "graph" and (s["batch_shape"] != () or (s["is_edge"] and s["edge_shape"] is None)):
            return False
        if s["kind"] in ("dynbox", "batched"):
            return False
    return True


@given(space_specs(kinds=GYM_KINDS).filter(_gym_convertible), seeds)
def test_gym_roundtrip(spec, seed):
    space = build_space(spec, NB)
    gym_space = sgu.to_gym_space(space)
    for x in _samples(space, seed, 2):
        gx = sgu.to_gym_data(space, x)
        assert gym_space.contains(gx)
        back = sgu.from_gym_data(space, gx)
        assert space.contains(back)
        assert_tree_equal(back, x, check_dtype=False)
    if "graph" not in kinds_in(spec):
        # Graph count limits are not representable in gymnasium
        assert sgu.from_gym_space(gym_space, NB) == space


# ---------------------------------------------------------------------------
# 7. subspaces
# ---------------------------------------------------------------------------

@given(st.data())
def test_subspace_relation(data):
    spec = data.draw(space_specs())
    wide_spec = data.draw(widened_specs(spec))
    space = build_space(spec, NB)
    wide = build_space(wide_spec, NB)
    assert space.is_subspaceeq(space)
    assert not space.is_subspace(space)
    assert space.is_subspaceeq(wide)
    assert not wide.is_subspace(space)
    for x in _samples(space, data.draw(seeds), 3):
        assert wide.contains(x)
    # Extra dict keys keep the subspace relation (but not membership, which needs exact keys)
    wider = build_space(data.draw(widened_specs(wide_spec, extra_keys=True)), NB)
    assert space.is_subspaceeq(wider) and wide.is_subspaceeq(wider)


# ---------------------------------------------------------------------------
# 8. construction from data
# ---------------------------------------------------------------------------

@given(space_specs(kinds=CONSTRUCTIBLE_KINDS), seeds, st.booleans())
def test_construct_space_contains_every_sample(spec, seed, add_bounds):
    space = build_space(spec, NB)
    items = _samples(space, seed, 3)
    constructed = scu.construct_space_from_data_stream(items, NB, add_bounds=add_bounds)
    for x in items:
        assert constructed.contains(x), (constructed, x)


# ---------------------------------------------------------------------------
# Cross-backend subset (one CPU device per backend)
# ---------------------------------------------------------------------------

_cross_kinds = ("box", "binary", "dynbox", "text", "graph", "dict", "tuple", "union", "batched")


@pytest.mark.parametrize("backend,device", cpu_backend_params())
@settings(max_examples=10)
@given(spec=space_specs(kinds=_cross_kinds, max_leaves=3), seed=seeds)
def test_cross_backend_conversion(backend, device, spec, seed):
    space = build_space(spec, NB)
    assert space.to() == space
    moved_space = space.to(backend, device)
    assert moved_space.backend is backend
    native = build_space(spec, backend, device)
    assert moved_space == native
    # A space serialised on numpy restores natively on the other backend
    assert ssu.json_to_space(ssu.space_to_json(space), backend, device) == native
    for x in _samples(space, seed, 2):
        moved = space.data_to(x, backend, device)
        assert moved_space.contains(moved)
        # jax without x64 stores numpy's default 64-bit graph counts as int32
        assert_tree_equal(moved_space.data_to(moved, NB), x, check_dtype=backend.simplified_name != "jax")


@pytest.mark.parametrize("backend,device", cpu_backend_params())
@settings(max_examples=10)
@given(spec=space_specs(kinds=_cross_kinds, max_leaves=3), seed=seeds)
def test_backend_native_contract(backend, device, spec, seed):
    space = build_space(spec, backend, device)
    xs = _samples(space, seed, 2)
    for x in xs:
        assert space.contains(x)
    assert_tree_equal(_samples(space, seed, 2), xs)
    restored = ssu.json_to_space(json.loads(json.dumps(ssu.space_to_json(space))), backend, device)
    assert restored == space
    kinds = kinds_in(spec)
    if sfu.is_flattenable(space):
        for x in xs:
            _check_flatten(space, x)
    if kinds <= set(BATCHABLE_KINDS):
        batch = sbu.concatenate(space, xs)
        batched_space = sbu.batch_space(space, len(xs))
        assert batched_space.contains(batch)
        for i, x in enumerate(xs):
            assert_tree_equal(sbu.get_at(batched_space, batch, i), x)
