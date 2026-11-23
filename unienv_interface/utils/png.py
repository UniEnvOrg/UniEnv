"""PNG encoding without a third-party image library."""

import struct
import zlib

import numpy as np

__all__ = ["encode_png"]


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_png(image: np.ndarray, compression: int = 6) -> bytes:
    """PNG bytes of a ``uint8`` image of shape ``(H, W)`` or ``(H, W, 1 | 3 | 4)`` (gray, RGB, RGBA)."""
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise TypeError(f"PNG images must be uint8, got {image.dtype}")
    if image.ndim == 2:
        image = image[..., None]
    height, width, channels = image.shape
    color_type = {1: 0, 3: 2, 4: 6}.get(channels)
    if color_type is None:
        raise ValueError(f"PNG images have 1, 3 or 4 channels, got {channels}")
    rows = np.concatenate([np.zeros((height, 1), np.uint8),
                           np.ascontiguousarray(image).reshape(height, width * channels)], axis=1)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(rows.tobytes(), compression))
            + _chunk(b"IEND", b""))
