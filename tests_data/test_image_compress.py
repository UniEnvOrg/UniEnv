"""ImageCompressTransformation / ImageDecompressTransformation (JPEG bytes in fixed-size arrays)."""

import pickle

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import BoxSpace, DictSpace
from unienv_data.transformations.image_compress import (
    ImageCompressTransformation,
    ImageDecompressTransformation,
    get_jpeg_compression_ratio,
)

# Smooth images compress well, so JPEG round trips stay within a small tolerance.
TOLERANCE = 8


def smooth_image(shape, channel_axis=True, seed=0):
    """uint8 image(s) of ``shape`` = (..., H, W, C) (or (..., H, W)) made of gentle gradients."""
    rng = np.random.default_rng(seed)
    h, w = (shape[-3], shape[-2]) if channel_axis else (shape[-2], shape[-1])
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    full = shape if channel_axis else tuple(shape) + (1,)
    flat = np.empty((int(np.prod(full[:-3], dtype=int)), h, w, full[-1]), np.uint8)
    for i in range(flat.shape[0]):
        for c in range(flat.shape[-1]):
            a, b = rng.uniform(40, 100, size=2)
            flat[i, :, :, c] = (60 + a * yy + b * xx).astype(np.uint8)
    return flat.reshape(shape)


def _space(shape, backend=NB, dtype=np.uint8, device=None):
    return BoxSpace(backend, 0, 255, dtype, shape=shape, device=device)


def _round_trip(tf, shape):
    space = _space(shape)
    target = tf.get_target_space_from_source(space)
    images = smooth_image(shape, channel_axis=tf.last_channel)
    encoded = tf.transform(space, images)
    assert tuple(encoded.shape) == target.shape and encoded.dtype == np.uint8
    inverse = tf.direction_inverse(space)
    decoded = inverse.transform(target, encoded)
    assert inverse.get_target_space_from_source(target).shape == space.shape
    return images, decoded


@pytest.mark.parametrize("shape", [(32, 48), (3, 16, 16)], ids=["single", "batched"])
def test_round_trip_channelless_images(shape):
    # Regression: last_channel=False built the decode buffer with a None channel count.
    tf = ImageCompressTransformation(last_channel=False, max_size_bytes=4096)
    space = _space(shape)
    assert tf.get_target_space_from_source(space).shape == shape[:-2] + (4096,)
    images, decoded = _round_trip(tf, shape)
    assert decoded.shape == images.shape
    assert np.abs(decoded.astype(int) - images.astype(int)).max() <= TOLERANCE


def test_grayscale_with_channel_axis_decodes_to_channel_axis():
    # Regression: decoding (H, W) grayscale JPEGs into an (H, W, 1) slot raised a broadcast error.
    tf = ImageCompressTransformation(max_size_bytes=4096)
    images, decoded = _round_trip(tf, (16, 16, 1))
    assert decoded.shape == (16, 16, 1)
    assert np.abs(decoded.astype(int) - images.astype(int)).max() <= TOLERANCE


@pytest.mark.parametrize("init_quality", [1, 10, 19, 20, 55, 100])
def test_any_init_quality_encodes(init_quality):
    # Regression: init_quality < 20 skipped the encode loop (UnboundLocalError).
    tf = ImageCompressTransformation(init_quality=init_quality, max_size_bytes=20_000)
    images, decoded = _round_trip(tf, (32, 32, 3))
    assert decoded.shape == images.shape
    if init_quality >= 55:
        assert np.abs(decoded.astype(int) - images.astype(int)).max() <= TOLERANCE


def test_quality_is_lowered_until_it_fits():
    tf = ImageCompressTransformation(init_quality=95)
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    full, q_full = tf.encode_to_size(img, 10**6)
    assert q_full == 95
    budget = len(full) - 1
    small, q_small = tf.encode_to_size(img, budget)
    assert len(small) <= budget and 20 <= q_small < 95


def test_overflow_raises_clear_error():
    # Regression: output that still did not fit crashed with a numpy broadcast error.
    tf = ImageCompressTransformation(max_size_bytes=300)
    space = _space((32, 32, 3))
    noise = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="max_size_bytes"):
        tf.transform(space, noise)

