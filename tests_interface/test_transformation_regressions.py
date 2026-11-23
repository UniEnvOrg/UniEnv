"""Regression tests for transformation bugs (resize backends, serialization, inverses, nesting)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.transformations import (
    BatchifyTransformation, ChainedTransformation, DictTransformation, FlattenDictTransformation,
    IdentityTransformation, ImageResizeTransformation, IterativeTransformation, RescaleTransformation,
    UnBatchifyTransformation, UnflattenDictTransformation, json_to_transformation, transformation_to_json,
)
from tests_common.backends import backend_params


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _int_dtype(backend):
    return backend.__array_namespace_info__().dtypes()["uint8"]


def _float_dtype(backend):
    return backend.__array_namespace_info__().dtypes()["float32"]


def _image_space(backend, device, shape, integer):
    if integer:
        return BoxSpace(backend, 0, 255, _int_dtype(backend), shape=shape, device=device)
    return BoxSpace(backend, 0.0, 1.0, _float_dtype(backend), shape=shape, device=device)


def _image(backend, device, shape, integer, seed=0):
    raw = np.random.default_rng(seed).random(shape)
    raw = (raw * 255).round().astype(np.uint8) if integer else raw.astype(np.float32)
    return backend.from_numpy(raw, device=device), raw


# ---------------------------------------------------------------------------
# ImageResizeTransformation: every backend, leading batch dims, integer images
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend,device", backend_params())
@pytest.mark.parametrize("integer", [False, True], ids=["float32", "uint8"])
@pytest.mark.parametrize("shape", [(8, 12, 3), (2, 8, 12, 3), (2, 3, 8, 12, 3), (8, 12, 1)])
def test_image_resize_output_in_target_space(backend, device, integer, shape):
    space = _image_space(backend, device, shape, integer)
    data, _ = _image(backend, device, shape, integer)
    t = ImageResizeTransformation(4, 6)
    target = t.get_target_space_from_source(space)
    y = t.transform(space, data)
    assert tuple(y.shape) == shape[:-3] + (4, 6, shape[-1])
    assert y.dtype == space.dtype
    assert target.contains(y)


@pytest.mark.parametrize("backend,device", backend_params())
@pytest.mark.parametrize("integer", [False, True], ids=["float32", "uint8"])
@pytest.mark.parametrize("in_hw,out_hw", [
    ((8, 8), (2, 2)), ((8, 12), (4, 6)), ((64, 48), (17, 33)), ((5, 7), (11, 13)), ((10, 10), (10, 10)), ((3, 9), (7, 2)),
])
def test_image_resize_matches_reference_on_every_backend(backend, device, integer, in_hw, out_hw):
    """All backends implement antialiased bilinear resizing (torch ``antialias=True`` semantics)."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    shape = (2, *in_hw, 3)
    space = _image_space(backend, device, shape, integer)
    data, raw = _image(backend, device, shape, integer, seed=1)
    reference = F.interpolate(
        torch.as_tensor(raw, dtype=torch.float64).permute(0, 3, 1, 2), size=out_hw,
        mode="bilinear", align_corners=False, antialias=True,
    ).permute(0, 2, 3, 1).numpy()
    y = _to_numpy(ImageResizeTransformation(*out_hw).transform(space, data))
    if integer:
        assert np.abs(y.astype(np.int64) - np.clip(np.round(reference), 0, 255)).max() <= 1
    else:
        np.testing.assert_allclose(y, reference, atol=1e-5)


def test_image_resize_downscale_is_antialiased_on_numpy():
    # A 2x2-periodic pattern downscaled 4x: plain bilinear sampling (the previous
    # numpy path) returned 0.25 everywhere; the antialiased result matches torch/jax.
    img = np.zeros((8, 8, 1), np.float32)
    img[::2, ::2] = 1.0
    space = BoxSpace(NB, 0.0, 1.0, np.float32, shape=(8, 8, 1))
    y = ImageResizeTransformation(2, 2).transform(space, img)[..., 0]
    np.testing.assert_allclose(y, [[0.287, 0.249], [0.249, 0.216]], atol=1e-3)


# ---------------------------------------------------------------------------
# RescaleTransformation keeps new_dtype through JSON without a source space
# ---------------------------------------------------------------------------

def test_rescale_new_dtype_survives_json_without_source_space():
    t = RescaleTransformation(-1.0, 1.0, new_dtype=np.float64)
    restored = json_to_transformation(json.loads(json.dumps(transformation_to_json(t))))
    assert restored.new_dtype == np.float64
    space = BoxSpace(NB, 0.0, 10.0, np.float32, shape=(3,))
    y = restored.transform(space, np.array([0.0, 5.0, 10.0], np.float32))
    assert y.dtype == np.float64
    np.testing.assert_allclose(y, [-1.0, 0.0, 1.0])


def test_rescale_new_dtype_survives_json_inside_iterative_without_source_space():
    t = IterativeTransformation(RescaleTransformation(-1.0, 1.0, new_dtype=np.float64))
    restored = json_to_transformation(json.loads(json.dumps(transformation_to_json(t))))
    assert restored.transformation.new_dtype == np.float64


def test_rescale_torch_new_dtype_survives_json_without_source_space():
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    t = RescaleTransformation(-1.0, 1.0, new_dtype=torch.float64)
    restored = json_to_transformation(json.loads(json.dumps(transformation_to_json(t))))
    assert restored.new_dtype == torch.float64
    space = BoxSpace(TB, 0.0, 10.0, torch.float32, shape=(2,))
    assert restored.transform(space, torch.tensor([0.0, 10.0])).dtype == torch.float64


def test_rescale_serialized_dtype_follows_source_space_backend_on_load():
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    t = RescaleTransformation(-1.0, 1.0, new_dtype=np.float64)
    space = BoxSpace(TB, 0.0, 10.0, torch.float32, shape=(2,))
    restored = json_to_transformation(json.loads(json.dumps(transformation_to_json(t))), space)
    assert restored.new_dtype == torch.float64


# ---------------------------------------------------------------------------
# DictTransformation with nested (overlapping) keys
# ---------------------------------------------------------------------------

def _box(low, high):
    return BoxSpace(NB, low, high, np.float32, shape=(1,))


def test_dict_transformation_nested_keys_transform_matches_target_space():
    source = DictSpace(NB, {"a": DictSpace(NB, {"x": _box(0.0, 100.0), "y": _box(0.0, 1.0)})})
    t = DictTransformation({
        "a": IterativeTransformation(RescaleTransformation(-1.0, 1.0)),
        "a/x": RescaleTransformation(0.0, 10.0),
    })
    target = t.get_target_space_from_source(source)
    x = {"a": {"x": np.array([0.0], np.float32), "y": np.array([1.0], np.float32)}}
    y = t.transform(source, x)
    assert target.contains(y)
    np.testing.assert_allclose(y["a"]["x"], [0.0], atol=1e-6)
    np.testing.assert_allclose(y["a"]["y"], [1.0], atol=1e-6)

    inverse = t.direction_inverse(source)
    assert inverse.get_target_space_from_source(target) == source
    back = inverse.transform(target, y)
    np.testing.assert_allclose(back["a"]["x"], x["a"]["x"], atol=1e-4)
    np.testing.assert_allclose(back["a"]["y"], x["a"]["y"], atol=1e-6)


def test_dict_transformation_disjoint_keys_unchanged():
    source = DictSpace(NB, {"x": _box(0.0, 100.0), "y": _box(0.0, 1.0)})
    t = DictTransformation({"x": RescaleTransformation(0.0, 1.0), "y": RescaleTransformation(-1.0, 1.0)})
    y = t.transform(source, {"x": np.array([50.0], np.float32), "y": np.array([0.5], np.float32)})
    np.testing.assert_allclose(y["x"], [0.5])
    np.testing.assert_allclose(y["y"], [0.0])
    back = t.direction_inverse(source).transform(t.get_target_space_from_source(source), y)
    np.testing.assert_allclose(back["x"], [50.0], atol=1e-4)


# ---------------------------------------------------------------------------
# ChainedTransformation.direction_inverse() without a source space
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("members,expected", [
    ([FlattenDictTransformation()], [UnflattenDictTransformation]),
    ([BatchifyTransformation()], [UnBatchifyTransformation]),
    ([IdentityTransformation(), FlattenDictTransformation()], [UnflattenDictTransformation, IdentityTransformation]),
])
def test_chained_inverse_without_source_space(members, expected):
    inverse = ChainedTransformation(members).direction_inverse()
    assert [type(t) for t in inverse.transformations] == expected


def test_chained_inverse_without_source_space_round_trips():
    source = DictSpace(NB, {"a": DictSpace(NB, {"x": _box(0.0, 1.0)}), "b": _box(0.0, 1.0)})
    t = ChainedTransformation([FlattenDictTransformation(), BatchifyTransformation()])
    x = {"a": {"x": np.array([0.5], np.float32)}, "b": np.array([0.25], np.float32)}
    y = t.transform(source, x)
    back = t.direction_inverse().transform(t.get_target_space_from_source(source), y)
    np.testing.assert_allclose(back["a"]["x"], [0.5])
    np.testing.assert_allclose(back["b"], [0.25])


# ---------------------------------------------------------------------------
# FlattenDictTransformation keeps empty nested dicts
# ---------------------------------------------------------------------------

def test_flatten_dict_round_trips_empty_nested_dict():
    source = DictSpace(NB, {"a": DictSpace(NB, {}), "b": _box(0.0, 1.0), "c": DictSpace(NB, {"d": DictSpace(NB, {})})})
    flatten = FlattenDictTransformation()
    target = flatten.get_target_space_from_source(source)
    inverse = flatten.direction_inverse(source)
    assert inverse.get_target_space_from_source(target) == source
    x = {"a": {}, "b": np.zeros(1, np.float32), "c": {"d": {}}}
    flat = flatten.transform(source, x)
    assert target.contains(flat)
    back = inverse.transform(target, flat)
    assert source.contains(back)
    assert back["a"] == {} and back["c"] == {"d": {}}


def test_rescale_inverse_with_differently_abbreviated_bounds():
    """Stored bounds of different shapes (per-row low, scalar high) still invert (found with xbarray 1.0)."""
    import numpy as np
    from unienv_interface.backends.numpy import NumpyComputeBackend as NB
    from unienv_interface.space import BoxSpace
    from unienv_interface.transformations.rescale import RescaleTransformation

    space = BoxSpace(NB, np.array([[0.0], [1.0], [2.0]]), 5.0, np.float64, shape=(3, 1))
    space._low, space._high = np.array([[0.0], [1.0], [2.0]]), np.array(5.0)  # as abbreviation may store them
    inverse = RescaleTransformation(0.0, 1.0).direction_inverse(space)
    target = RescaleTransformation(0.0, 1.0).get_target_space_from_source(space)
    data = np.array([[2.5], [3.0], [3.5]])
    forward = RescaleTransformation(0.0, 1.0).transform(space, data)
    np.testing.assert_allclose(inverse.transform(target, forward), data)
