"""ImageDecompressTransformation decodes JPEG batches with torchvision when it can.

The batch path must give exactly the pixels of the per-image PIL path it replaces, and hand
everything it does not handle back to PIL.
"""

import io

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import BoxSpace
from unienv_data.transformations import image_compress
from unienv_data.transformations.image_compress import ImageCompressTransformation, ImageDecompressTransformation


def _images(shape, last_channel=True, seed=0):
    """Gradients plus noise: realistic work for JPEG (chroma subsampling, rounding)."""
    rng = np.random.default_rng(seed)
    h, w = shape[-3:-1] if last_channel else shape[-2:]
    yy, xx = np.mgrid[0:h, 0:w]
    base = (xx * 3 + yy * 2) % 256
    if last_channel:
        base = base[:, :, None]
    return np.clip(base + rng.normal(0, 12, shape), 0, 255).astype(np.uint8)


def _encode(shape, quality, last_channel=True):
    images = _images(shape, last_channel)
    tf = ImageCompressTransformation(init_quality=quality, max_size_bytes=int(np.prod(shape[-3:])) + 2048,
                                     last_channel=last_channel)
    space = BoxSpace(NB, 0, 255, np.uint8, shape=shape)
    return tf.transform(space, images), tf.direction_inverse(space), tf.get_target_space_from_source(space)


def _decode_with_pil(monkeypatch, decoder, space, encoded):
    with monkeypatch.context() as m:
        m.setattr(image_compress, "_TORCHVISION_DECODE_JPEG", False)
        return decoder.transform(space, encoded)


@pytest.fixture
def batch_decode_calls(monkeypatch):
    pytest.importorskip("torchvision")
    decode_jpeg, read_mode = image_compress._torchvision_decode_jpeg()
    calls = []

    def counting(tensors, *args, **kwargs):
        calls.append(len(tensors))
        return decode_jpeg(tensors, *args, **kwargs)
    monkeypatch.setattr(image_compress, "_TORCHVISION_DECODE_JPEG", (counting, read_mode))
    return calls


@pytest.mark.parametrize("shape,last_channel", [
    ((16, 96, 96, 3), True),
    ((5, 37, 53, 3), True),
    ((2, 3, 24, 24, 3), True),
    ((6, 40, 40, 1), True),
    ((4, 32, 48), False),
    ((24, 24, 3), True),
], ids=["rgb", "odd-size", "batched", "gray-channel", "gray-channelless", "single"])
@pytest.mark.parametrize("quality", [40, 75, 95])
def test_batch_decode_is_bit_identical_to_pil(monkeypatch, batch_decode_calls, shape, last_channel, quality):
    encoded, decoder, space = _encode(shape, quality, last_channel)
    decoded = decoder.transform(space, encoded)
    assert batch_decode_calls == [int(np.prod(shape[:-3 if last_channel else -2], dtype=int))]
    expected = _decode_with_pil(monkeypatch, decoder, space, encoded)
    assert decoded.shape == expected.shape == shape and decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, expected)


def test_grayscale_jpeg_into_rgb_slots_matches_pil(monkeypatch, batch_decode_calls):
    # A grayscale JPEG decoded into a 3-channel image fills every channel (PIL path broadcast).
    encoded, _, space = _encode((3, 20, 20, 1), 90)
    decoder = ImageDecompressTransformation(20, 20, 3)
    decoded = decoder.transform(space, encoded)
    assert batch_decode_calls == [3]
    np.testing.assert_array_equal(decoded, _decode_with_pil(monkeypatch, decoder, space, encoded))


def test_torch_backend_batch_decode(monkeypatch, batch_decode_calls):
    import torch
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    encoded, _, _ = _encode((4, 32, 32, 3), 80)
    space = BoxSpace(TB, 0, 255, torch.uint8, shape=encoded.shape, device="cpu")
    decoder = ImageDecompressTransformation(32, 32, 3)
    decoded = decoder.transform(space, torch.from_numpy(encoded))
    assert isinstance(decoded, torch.Tensor) and batch_decode_calls == [4]
    np.testing.assert_array_equal(decoded.numpy(), _decode_with_pil(monkeypatch, decoder, space, torch.from_numpy(encoded)).numpy())


def test_read_only_input_is_decoded(batch_decode_calls):
    encoded, decoder, space = _encode((3, 16, 16, 3), 80)
    encoded.setflags(write=False)
    assert decoder.transform(space, encoded).shape == (3, 16, 16, 3)
    assert batch_decode_calls == [3]


def test_mode_conversion_and_other_formats_use_pil(monkeypatch, batch_decode_calls):
    encoded, _, space = _encode((2, 16, 16, 3), 80)
    gray = ImageDecompressTransformation(16, 16, None, mode="L")
    assert gray.transform(space, encoded).shape == (2, 16, 16)
    # PNG bytes in the fixed-size slots (format=None lets PIL detect the format).
    images = _images((2, 16, 16, 3))
    png = np.zeros((2, 2048), np.uint8)
    for i, image in enumerate(images):
        buf = io.BytesIO()
        image_compress.Image.fromarray(image).save(buf, format="PNG")
        png[i, :len(buf.getvalue())] = np.frombuffer(buf.getvalue(), np.uint8)
    decoded = ImageDecompressTransformation(16, 16, 3).transform(BoxSpace(NB, 0, 255, np.uint8, shape=png.shape), png)
    np.testing.assert_array_equal(decoded, images)  # lossless
    assert batch_decode_calls == []


@pytest.mark.parametrize("damage", ["garbled", "truncated"])
def test_corrupt_data_still_raises(damage):
    encoded, decoder, space = _encode((2, 16, 16, 3), 80)
    if damage == "garbled":
        encoded[1, 2:200] = 7  # keep the start-of-image marker, break the rest
    else:
        encoded[1, int(np.nonzero(encoded[1])[0].max()) // 2:] = 0
    with pytest.raises(Exception):
        decoder.transform(space, encoded)


def test_without_torchvision_pil_decodes(monkeypatch):
    monkeypatch.setattr(image_compress, "_TORCHVISION_DECODE_JPEG", False)
    assert image_compress._torchvision_decode_jpeg() is None
    images = _images((3, 24, 24, 3))
    tf = ImageCompressTransformation(init_quality=90, max_size_bytes=4096)
    space = BoxSpace(NB, 0, 255, np.uint8, shape=images.shape)
    decoded = tf.direction_inverse(space).transform(tf.get_target_space_from_source(space), tf.transform(space, images))
    assert decoded.shape == images.shape
    assert np.abs(decoded.astype(int) - images.astype(int)).mean() < 12
