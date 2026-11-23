"""Images in observations: finding them by space metadata, preparing them for display, encoding PNG."""

from __future__ import annotations

import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unienv_interface.space import BoxSpace, Space

__all__ = ["image_kind", "prepare_image", "encode_png"]

_CHANNELS = {"rgb": 3, "bgr": 3, "rgba": 4, "bgra": 4, "gray": 1}


def image_kind(space: Space, batched: bool) -> Optional[str]:
    """``"image"`` / ``"depth"`` for leaves shown as pictures, else ``None``.

    By metadata (``kind`` ``"image"`` or ``"depth"``), or for unannotated ``uint8`` boxes shaped like
    pictures (height and width of at least 16, then 1, 3 or 4 channels).
    """
    if not isinstance(space, BoxSpace):
        return None
    md = dict(space.metadata or {})
    if md.get("kind") in ("image", "depth"):
        return md["kind"]
    if md.get("kind") is not None:
        return None
    shape = tuple(space.shape)[1:] if batched else tuple(space.shape)
    dtype = np.dtype(str(np.asarray(space.backend.to_numpy(space.low)).dtype))
    if dtype == np.uint8 and len(shape) == 3 and shape[-1] in (1, 3, 4) and min(shape[:2]) >= 16:
        return "image"
    return None


def prepare_image(space: Space, image: np.ndarray, kind: str, *, max_side: int = 512) -> Tuple[np.ndarray, Dict[str, Any]]:
    """One image as ``uint8`` ``(H, W, 1|3|4)`` in RGB order, upright, at most ``max_side`` pixels wide or high.

    Returns the image and notes for the model (e.g. the depth range shown in gray levels).
    """
    md = dict(space.metadata or {})
    notes: Dict[str, Any] = {}
    array = np.asarray(image)
    layout = str(md.get("layout", "")).upper()
    if layout == "CHW" and array.ndim == 3:
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = array[..., None]
    if kind == "depth":
        values = array.astype(np.float64)[..., :1]
        finite = values[np.isfinite(values)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
        scaled = np.zeros_like(values) if hi <= lo else (values - lo) / (hi - lo)
        array = np.where(np.isfinite(values), np.round(scaled * 255), 0).astype(np.uint8)
        notes["depth_range"] = [lo, hi]
        if md.get("units") is not None:
            notes["depth_units"] = md["units"]
        notes["shading"] = "near is dark, far is light"
    elif array.dtype != np.uint8:
        floating = np.issubdtype(array.dtype, np.floating)
        array = np.clip(np.round(array.astype(np.float64) * (255.0 if floating else 1.0)), 0, 255).astype(np.uint8)
    fmt = str(md.get("format", "")).lower()
    if fmt in ("bgr", "bgra") and array.shape[-1] >= 3:
        array = array[..., [2, 1, 0] + ([3] if array.shape[-1] == 4 else [])]
    flipped = str(md.get("flipped", "")).lower()
    if flipped in ("vertical", "both"):
        array = array[::-1]
    if flipped in ("horizontal", "both"):
        array = array[:, ::-1]
    height, width = array.shape[:2]
    step = max(1, int(np.ceil(max(height, width) / max_side)))
    if step > 1:
        array = array[::step, ::step]
        notes["downscaled_by"] = step
    if array.shape[-1] == 2:  # gray + alpha is not a PNG color type we write: drop alpha
        array = array[..., :1]
    return np.ascontiguousarray(array), notes


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_png(image: np.ndarray) -> bytes:
    """PNG bytes of a ``uint8`` ``(H, W, 1|3|4)`` image (no third-party encoder needed)."""
    height, width, channels = image.shape
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    rows = np.concatenate([np.zeros((height, 1), np.uint8), image.reshape(height, width * channels)], axis=1)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
            + _chunk(b"IEND", b""))


def split_batch(array: np.ndarray, batch_ndim: int) -> List[Tuple[Tuple[int, ...], np.ndarray]]:
    """``(index, image)`` for each image of a batched image array."""
    if batch_ndim == 0:
        return [((), array)]
    return [(index, array[index]) for index in np.ndindex(*array.shape[:batch_ndim])]
