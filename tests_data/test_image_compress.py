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

# Smooth images compress well, so JPEG round trips stay close (chroma subsampling
# costs a few levels on tiny images).
TOLERANCE = 24
MEAN_TOLERANCE = 3


def assert_close_images(decoded, images):
    err = np.abs(np.asarray(decoded).astype(int) - images.astype(int))
    assert err.max() <= TOLERANCE and err.mean() <= MEAN_TOLERANCE, (err.max(), err.mean())


def smooth_image(shape, channel_axis=True, seed=0):
    """uint8 image(s) of ``shape`` = (..., H, W, C) (or (..., H, W)) made of gentle gradients."""
    rng = np.random.default_rng(seed)
    h, w = (shape[-3], shape[-2]) if channel_axis else (shape[-2], shape[-1])
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    full = shape if channel_axis else tuple(shape) + (1,)
    flat = np.empty((int(np.prod(full[:-3], dtype=int)), h, w, full[-1]), np.uint8)
    for i in range(flat.shape[0]):
        for c in range(flat.shape[-1]):
            a, b = rng.uniform(30, 90, size=2)  # stays below 255
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


@pytest.mark.parametrize("shape", [(64, 64, 3), (48, 40, 3), (64, 64, 1), (2, 3, 32, 32, 3), (8, 8, 3), (1, 1, 3), (4, 4, 1)],
                         ids=["rgb64", "rgb-rect", "gray", "batched", "rgb8", "rgb1", "gray4"])
def test_round_trip_default_budget(shape):
    # Regression: the default budget ignored the JPEG header (~630 bytes for RGB), so
    # small images (anything up to ~64x64 RGB) always overflowed with a broadcast error.
    images, decoded = _round_trip(ImageCompressTransformation(), shape)
    assert decoded.shape == images.shape and decoded.dtype == np.uint8
    assert_close_images(decoded, images)


@pytest.mark.parametrize("shape", [(32, 48), (3, 16, 16)], ids=["single", "batched"])
def test_round_trip_channelless_images(shape):
    # Regression: last_channel=False built the decode buffer with a None channel count.
    tf = ImageCompressTransformation(last_channel=False, max_size_bytes=4096)
    space = _space(shape)
    assert tf.get_target_space_from_source(space).shape == shape[:-2] + (4096,)
    images, decoded = _round_trip(tf, shape)
    assert decoded.shape == images.shape
    assert_close_images(decoded, images)


def test_grayscale_with_channel_axis_decodes_to_channel_axis():
    # Regression: decoding (H, W) grayscale JPEGs into an (H, W, 1) slot raised a broadcast error.
    tf = ImageCompressTransformation(max_size_bytes=4096)
    images, decoded = _round_trip(tf, (16, 16, 1))
    assert decoded.shape == (16, 16, 1)
    assert_close_images(decoded, images)


@pytest.mark.parametrize("init_quality", [1, 10, 19, 20, 55, 100])
def test_any_init_quality_encodes(init_quality):
    # Regression: init_quality < 20 skipped the encode loop (UnboundLocalError).
    tf = ImageCompressTransformation(init_quality=init_quality, max_size_bytes=20_000)
    images, decoded = _round_trip(tf, (32, 32, 3))
    assert decoded.shape == images.shape
    if init_quality >= 55:
        assert_close_images(decoded, images)


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


def test_explicit_size_and_ratio_budgets():
    space = _space((32, 32, 3))
    assert ImageCompressTransformation(max_size_bytes=5000).get_target_space_from_source(space).shape == (5000,)
    tf = ImageCompressTransformation(compression_ratio=4)
    assert tf.get_target_space_from_source(space).shape == (32 * 32 * 3 // 4 + 1 + tf.header_bytes,)
    with pytest.raises(AssertionError):
        ImageCompressTransformation(max_size_bytes=10, compression_ratio=2)
    with pytest.raises(AssertionError):
        ImageCompressTransformation(format="PNG")


def test_legacy_configuration_keeps_its_target_shape():
    # Datasets written before the header allowance must load with the same shape.
    space = _space((128, 128, 3))
    legacy_json = {"init_quality": 70, "max_size_bytes": None, "compression_ratio": 18,
                   "mode": None, "format": "JPEG", "last_channel": True}
    legacy = ImageCompressTransformation.deserialize_from(legacy_json)
    assert legacy.get_target_space_from_source(space).shape == (128 * 128 * 3 // 18 + 1,)
    state = dict(ImageCompressTransformation().__dict__)
    del state["header_bytes"], state["last_channel"]
    restored = ImageCompressTransformation.__new__(ImageCompressTransformation)
    restored.__setstate__(state)
    assert restored.header_bytes == 0 and restored.last_channel


def test_serialization_round_trip():
    tf = ImageCompressTransformation(init_quality=80, last_channel=False)
    again = ImageCompressTransformation.deserialize_from(tf.serialize())
    space = _space((16, 16))
    assert again.get_target_space_from_source(space) == tf.get_target_space_from_source(space)
    assert pickle.loads(pickle.dumps(tf)).serialize() == tf.serialize()

    dec = ImageDecompressTransformation(16, 16, None, format="JPEG")
    dec_again = ImageDecompressTransformation.deserialize_from(dec.serialize())
    assert dec_again.serialize() == dec.serialize()
    inv = dec.direction_inverse(BoxSpace(NB, 0, 255, np.uint8, shape=(700,)))
    assert isinstance(inv, ImageCompressTransformation)
    assert inv.max_size_bytes == 700 and not inv.last_channel


def test_source_space_validation():
    tf = ImageCompressTransformation()
    with pytest.raises(AssertionError):
        tf.get_target_space_from_source(_space((8, 8, 2)))
    with pytest.raises(AssertionError):
        tf.get_target_space_from_source(_space((8,)))
    with pytest.raises(AssertionError):
        tf.get_target_space_from_source(DictSpace(NB, {"a": _space((8, 8, 3))}))
    with pytest.raises(AssertionError):
        ImageCompressTransformation(last_channel=False).get_target_space_from_source(_space((8,)))
    with pytest.raises(AssertionError):
        tf.direction_inverse(None)


def test_jpeg_ratio_table():
    ratios = [get_jpeg_compression_ratio(q) for q in range(1, 101)]
    assert all(a >= b for a, b in zip(ratios, ratios[1:]))
    assert get_jpeg_compression_ratio(10) == get_jpeg_compression_ratio(55)
    assert get_jpeg_compression_ratio(100) >= 1


def test_torch_backend_round_trip():
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    tf = ImageCompressTransformation()
    space = BoxSpace(TB, 0, 255, torch.uint8, shape=(24, 24, 3), device="cpu")
    images = smooth_image((24, 24, 3))
    encoded = tf.transform(space, torch.from_numpy(images))
    assert isinstance(encoded, torch.Tensor) and encoded.dtype == torch.uint8
    target = tf.get_target_space_from_source(space)
    decoded = tf.direction_inverse(space).transform(target, encoded)
    assert isinstance(decoded, torch.Tensor)
    assert_close_images(decoded.numpy(), images)
