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
