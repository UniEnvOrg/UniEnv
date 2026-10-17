"""Opt-in lossy compression of server-to-client image arrays.

A client asks for it in ``hello``, e.g. ``RemoteClient.connect(address,
compress={"image": "jpeg:90"})``. The server then encodes the uint8 leaves of
results whose space has ``metadata["kind"] == "image"`` (``layout`` ``"HWC"`` or
``"CHW"``, 1 or 3 channels; leading dimensions are batch dimensions) with that
codec. Depth maps, float arrays and everything the client sends stay raw.

Codecs are pluggable (:func:`unienv_interface.remote.wire.register_codec`); the
built-in ``jpeg`` codec uses OpenCV when installed, otherwise Pillow.
"""

from __future__ import annotations

import io
import math
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from unienv_interface.backends import get_backend_from_tensor

from .wire import ArrayCodec, CodecArray, WireError, _CODECS, register_codec

if TYPE_CHECKING:
    from .server import _Resource

__all__ = ["JpegCodec", "parse_spec", "compress_result"]

#: Space metadata kinds a client may ask to compress.
KINDS = ("image",)
_MAX_SIDE = 65500  # JPEG limit

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is optional, Pillow is a core dependency
    cv2 = None


def _jpeg_size(blob: Sequence[int]) -> Tuple[int, int, int]:
    """``(height, width, channels)`` from a JPEG's SOF header, without decoding it."""
    data = bytes(blob)
    if data[:2] != b"\xff\xd8":
        raise WireError("Not a JPEG image")
    offset = 2
    while offset + 9 < len(data):
        if data[offset] != 0xFF:
            raise WireError("Corrupt JPEG image")
        marker = data[offset + 1]
        length = struct.unpack(">H", data[offset + 2:offset + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height, width, channels = struct.unpack(">HHB", data[offset + 5:offset + 10])
            return height, width, channels
        offset += 2 + length
    raise WireError("JPEG image has no frame header")


def _swapped(images: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """RGB <-> BGR (three plane copies: much faster than a reversed-stride copy)."""
    out = np.empty(images.shape, images.dtype) if out is None else out
    out[..., 0], out[..., 1], out[..., 2] = images[..., 2], images[..., 1], images[..., 0]
    return out


class JpegCodec(ArrayCodec):
    """JPEG per image. Options: ``quality`` (1-100), ``layout`` (HWC/CHW), ``format`` (rgb/bgr/gray)."""

    def encode(self, array: np.ndarray, options: Dict[str, Any]) -> List[bytes]:
        quality = int(options.get("quality", 90))
        images = self._images(np.asarray(array), options["layout"])
        if images.shape[-1] == 1:
            images = images[..., 0]
        elif (options.get("format") == "bgr") != (cv2 is not None):  # cv2 wants BGR, Pillow RGB
            images = _swapped(images)
        blobs = []
        for image in images:
            image = np.ascontiguousarray(image)
            if cv2 is not None:
                blobs.append(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes())
            else:
                from PIL import Image
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, "JPEG", quality=quality)
                blobs.append(buffer.getvalue())
        return blobs

    def decode(self, blobs: Sequence[Any], dtype: np.dtype, shape: Tuple[int, ...],
               options: Dict[str, Any]) -> np.ndarray:
        layout = options.get("layout")
        if dtype != np.uint8 or layout not in ("HWC", "CHW") or len(shape) < 3:
            raise WireError("Invalid JPEG array")
        result = np.empty(shape, np.uint8)
        images = self._images(result, layout)
        if len(blobs) != len(images):
            raise WireError("JPEG array has the wrong number of images")
        height, width, channels = images.shape[1:]
        bgr = options.get("format") == "bgr"
        if cv2 is not None:
            flag = cv2.IMREAD_GRAYSCALE if channels == 1 else (
                cv2.IMREAD_COLOR if bgr else getattr(cv2, "IMREAD_COLOR_RGB", cv2.IMREAD_COLOR))
            swap = channels == 3 and not bgr and flag == cv2.IMREAD_COLOR
        else:
            swap = channels == 3 and bgr
        for image, blob in zip(images, blobs):
            if _jpeg_size(blob) != (height, width, channels):  # checked before decoding anything
                raise WireError("JPEG image does not match the array shape")
            if cv2 is not None:
                decoded = cv2.imdecode(np.frombuffer(blob, np.uint8), flag)
            else:
                from PIL import Image
                decoded = np.asarray(Image.open(io.BytesIO(bytes(blob))))
            if decoded is None:
                raise WireError("Corrupt JPEG image")
            decoded = decoded.reshape(height, width, channels)
            if swap:
                _swapped(decoded, out=image)
            else:
                image[...] = decoded
        return result

    @staticmethod
    def _images(array: np.ndarray, layout: str) -> np.ndarray:
        """A ``(N, H, W, C)`` view of the images (writing to it writes to ``array``)."""
        images = array.reshape(-1, *array.shape[-3:])
        return images.transpose(0, 2, 3, 1) if layout == "CHW" else images


register_codec("jpeg", JpegCodec())


def parse_spec(spec: Any) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """Validate ``{"image": "jpeg:90"}`` into ``{"image": ("jpeg", {"quality": 90})}``."""
    if not isinstance(spec, dict):
        raise ValueError(f"compress must be a dict like {{'image': 'jpeg:90'}}, got {spec!r}")
    result = {}
    for kind, text in spec.items():
        if kind not in KINDS:
            raise ValueError(f"Cannot compress {kind!r} data; supported: {', '.join(KINDS)}")
        codec, _, argument = str(text).partition(":")
        if codec not in _CODECS:
            raise ValueError(f"Unknown codec {codec!r}; registered: {', '.join(sorted(_CODECS))}")
        options: Dict[str, Any] = {}
        if codec == "jpeg":
            quality = int(argument or 90)
            if not 1 <= quality <= 100:
                raise ValueError(f"JPEG quality must be in 1..100, got {quality}")
            options["quality"] = quality
        result[kind] = (codec, options)
    return result


def _leaf(value: Any, space: Any, spec: Dict[str, Tuple[str, Dict[str, Any]]]) -> Any:
    metadata = space.metadata
    kind = metadata.get("kind")
    if kind not in spec:
        return value
    if not isinstance(value, np.ndarray):
        try:
            value = np.asarray(get_backend_from_tensor(value).to_numpy(value))
        except Exception:
            return value
    layout, image_format = metadata.get("layout", "HWC"), metadata.get("format")
    if value.dtype != np.uint8 or layout not in ("HWC", "CHW") or value.ndim < 3 or image_format == "rgba":
        return value
    height, width, channels = value.shape[-3:] if layout == "HWC" else (*value.shape[-2:], value.shape[-3])
    if channels not in (1, 3) or not 0 < height <= _MAX_SIDE or not 0 < width <= _MAX_SIDE or value.size == 0:
        return value
    codec, options = spec[kind]
    return CodecArray(value, codec, dict(options, layout=layout, format=image_format or ("gray" if channels == 1 else "rgb")))


def _walk(value: Any, space: Any, spec: Dict[str, Tuple[str, Dict[str, Any]]]) -> Any:
    if space is None:
        return value
    children = getattr(space, "spaces", None)
    if isinstance(children, dict) and isinstance(value, dict):
        return {key: _walk(item, children.get(key), spec) for key, item in value.items()}
    if isinstance(children, (tuple, list)) and isinstance(value, (tuple, list)) and len(value) == len(children):
        return type(value)(_walk(item, child, spec) for item, child in zip(value, children))
    return _leaf(value, space, spec)


def compress_result(value: Any, resource: _Resource, method: str,
                    spec: Dict[str, Tuple[str, Dict[str, Any]]]) -> Any:
    """Mark the image leaves of a call result for compression (others are returned as is)."""
    obj = resource.obj
    if resource.kind == "env" and method in ("reset", "reload") and isinstance(value, tuple) and len(value) == 3:
        return (_walk(value[0], obj.context_space, spec), _walk(value[1], obj.observation_space, spec), value[2])
    if resource.kind == "env" and method == "step" and isinstance(value, tuple) and len(value) == 5:
        return (_walk(value[0], obj.observation_space, spec), *value[1:])
    if resource.kind == "node" and method in ("get_observation", "get_context"):
        return _walk(value, obj.observation_space if method == "get_observation" else obj.context_space, spec)
    return value
