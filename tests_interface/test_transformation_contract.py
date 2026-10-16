"""Property-based contract tests for data transformations.

For each transformation ``T`` on a drawn source space ``S`` with sampled ``x``:
``T.get_target_space_from_source(S).contains(T.transform(S, x))``, the inverse
restores ``x`` where an exact (or numerically exact) inverse exists, and the
JSON round trip of ``T`` reproduces the same target space and output.
"""

import json

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.transformations import (
    BatchifyTransformation, ChainedTransformation, CropTransformation, DictExcludeKeyTransformation,
    DictIncludeKeyTransformation, DictTransformation, FlattenDictTransformation, IdentityTransformation,
    ImageResizeTransformation, IterativeTransformation, RescaleTransformation, UnBatchifyTransformation,
    UnflattenDictTransformation, json_to_transformation, transformation_to_json,
)
from tests_common.backends import cpu_backend_params
from tests_common.space_strategies import (
    ALL_KINDS, box_specs, build_space, seeds, space_specs,
)
from tests_common.trees import assert_tree_equal

APPROX = dict(rtol=1e-4, atol=1e-4, check_dtype=False)
BATCHABLE_KINDS = ("box", "binary", "text", "graph", "dict", "tuple", "union", "batched")

bounded_float_boxes = box_specs(dtypes=st.just("float32"), allow_inf=False)
nested_dict_specs = space_specs(kinds=("box", "binary", "text", "dict"), max_leaves=6).filter(
    lambda spec: spec["kind"] == "dict")


def _samples(space, seed, n=2):
    rng = space.backend.random.random_number_generator(seed, device=space.device)
    out = []
    for _ in range(n):
        rng, x = space.sample(rng)
        out.append(x)
    return out


def check_transformation(t, space, seed, *, inverse="none", approx=False, n=2):
    """Contract checks; ``inverse`` is "exact", "approx" or "none"."""
    target = t.get_target_space_from_source(space)
    restored_t = json_to_transformation(json.loads(json.dumps(transformation_to_json(t, space))), space)
    assert type(restored_t) is type(t)
    assert restored_t.get_target_space_from_source(space) == target
    inv = t.direction_inverse(space) if inverse != "none" else None
    if inverse != "none":
        assert t.has_inverse and inv is not None
        assert inv.get_target_space_from_source(target) == space
    tol = APPROX if approx else {}
    for x in _samples(space, seed, n):
        y = t.transform(space, x)
        assert target.contains(y), (target, y)
        assert_tree_equal(restored_t.transform(space, x), y, **tol)
        if inv is not None:
            back = inv.transform(target, y)
            assert_tree_equal(back, x, **(APPROX if inverse == "approx" else {}))
    return target


@given(space_specs(), seeds)
def test_identity(spec, seed):
    space = build_space(spec, NB)
    assert check_transformation(IdentityTransformation(), space, seed, inverse="exact") == space


@given(bounded_float_boxes, seeds, st.floats(-5, 5, width=32), st.floats(0.5, 5, width=32))
def test_rescale(spec, seed, new_low, span):
    space = build_space(spec, NB)
    t = RescaleTransformation(new_low=new_low, new_high=new_low + span)
    check_transformation(t, space, seed, inverse="approx", approx=True)


@given(bounded_float_boxes, seeds)
def test_rescale_with_array_bounds_and_dtype(spec, seed):
    space = build_space(spec, NB)
    shape = space.shape
    low = np.full(shape[-1:], -2.0, dtype=np.float32) if shape else -2.0
    t = RescaleTransformation(new_low=low, new_high=low + 4.0, new_dtype=np.float64)
    target = check_transformation(t, space, seed, approx=True)
    assert target.dtype == np.float64


@given(box_specs(), seeds, st.integers(-3, 0), st.integers(0, 3))
def test_crop(spec, seed, low, high):
    space = build_space(spec, NB)
    shape = space.shape
    use_array = len(shape) > 0 and seed % 2 == 0
    dtype = space.dtype
    crop_low = np.full(shape[-1:], low, dtype=dtype) if use_array else low
    crop_high = np.full(shape[-1:], high, dtype=dtype) if use_array else high
    t = CropTransformation(crop_low, crop_high)
    target = check_transformation(t, space, seed)
    # The inverse of a crop is the identity on the cropped data
    assert isinstance(t.direction_inverse(space), IdentityTransformation)


@given(space_specs(kinds=BATCHABLE_KINDS), seeds)
def test_batchify_and_unbatchify(spec, seed):
    space = build_space(spec, NB)
    target = check_transformation(BatchifyTransformation(), space, seed, inverse="exact")
    check_transformation(UnBatchifyTransformation(), target, seed, inverse="exact")


@given(box_specs(shapes=st.sampled_from([(2,), (2, 3)])), seeds)
def test_batchify_on_inner_axis(spec, seed):
    space = build_space(spec, NB)
    t = BatchifyTransformation(axis=1)
    target = t.get_target_space_from_source(space)
    assert target.shape == space.shape[:1] + (1,) + space.shape[1:]
    for x in _samples(space, seed):
        y = t.transform(space, x)
        assert target.contains(y)
        assert_tree_equal(UnBatchifyTransformation(axis=1).transform(target, y), x)


@given(nested_dict_specs, seeds)
def test_flatten_and_unflatten_dict(spec, seed):
    space = build_space(spec, NB)
    target = check_transformation(FlattenDictTransformation(), space, seed, inverse="exact")
    assert all(not isinstance(s, DictSpace) for s in target.spaces.values())
    check_transformation(UnflattenDictTransformation(), target, seed, inverse="exact")


def _nested_keys(space, prefix=""):
    keys = []
    for k, sub in space.spaces.items():
        keys.append(prefix + k)
        if isinstance(sub, DictSpace):
            keys.extend(_nested_keys(sub, prefix + k + "/"))
    return keys


@given(nested_dict_specs, seeds, st.data())
def test_include_and_exclude_keys(spec, seed, data):
    space = build_space(spec, NB)
    keys = data.draw(st.lists(st.sampled_from(_nested_keys(space)), min_size=1, max_size=3, unique=True))
    include = DictIncludeKeyTransformation(keys)
    target = check_transformation(include, space, seed)
    exclude = DictExcludeKeyTransformation(keys)
    excluded = check_transformation(exclude, space, seed)
    assert set(keys[0].split("/")[:1]) <= set(target.keys())
    if "/" not in keys[0]:
        assert keys[0] not in excluded.keys()


@given(st.dictionaries(st.sampled_from(["a", "b", "c"]), bounded_float_boxes, min_size=1, max_size=3), seeds)
def test_dict_transformation(specs, seed):
    space = DictSpace(NB, {k: build_space(v, NB) for k, v in specs.items()})
    key = sorted(specs)[0]
    t = DictTransformation({key: RescaleTransformation(0.0, 1.0)})
    check_transformation(t, space, seed, inverse="approx", approx=True)


@given(st.lists(bounded_float_boxes, min_size=1, max_size=3), seeds, st.booleans())
def test_iterative_rescale(specs, seed, as_dict):
    from unienv_interface.space import TupleSpace
    leaves = [build_space(s, NB) for s in specs]
    space = DictSpace(NB, {f"k{i}": s for i, s in enumerate(leaves)}) if as_dict else TupleSpace(NB, leaves)
    t = IterativeTransformation(RescaleTransformation(-1.0, 1.0))
    check_transformation(t, space, seed, inverse="approx", approx=True)


@given(bounded_float_boxes, seeds)
def test_chained(spec, seed):
    space = build_space(spec, NB)
    t = ChainedTransformation([RescaleTransformation(0.0, 2.0), BatchifyTransformation(), IdentityTransformation()])
    check_transformation(t, space, seed, inverse="approx", approx=True)


@settings(max_examples=15)
@given(st.integers(2, 6), st.integers(2, 6), st.integers(1, 3), st.integers(1, 5), st.integers(1, 5), seeds)
def test_image_resize(h, w, c, new_h, new_w, seed):
    pytest.importorskip("cv2")
    space = BoxSpace(NB, 0.0, 1.0, np.float32, shape=(h, w, c))
    t = ImageResizeTransformation(new_h, new_w)
    target = check_transformation(t, space, seed)
    assert target.shape == (new_h, new_w, c)
    inv = t.direction_inverse(space)
    assert inv.get_target_space_from_source(target).shape == space.shape


@pytest.mark.parametrize("backend,device", cpu_backend_params())
@settings(max_examples=8)
@given(spec=bounded_float_boxes, seed=seeds)
def test_rescale_and_batchify_on_every_backend(backend, device, spec, seed):
    space = build_space(spec, backend, device)
    check_transformation(RescaleTransformation(0.0, 1.0), space, seed, inverse="approx", approx=True)
    check_transformation(BatchifyTransformation(), space, seed, inverse="exact")
    check_transformation(CropTransformation(-1.0, 1.0), space, seed)
