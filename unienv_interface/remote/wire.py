"""Safe wire format: one JSON document plus raw binary attachments. No pickle.

A frame is::

    u32 buffer count | u32 JSON length | u64 length per buffer | UTF-8 JSON | buffers...

(all little-endian). Plain JSON carries ``None``, bools, ints, strings, lists and
string-keyed dicts. Everything else is a JSON object tagged with the reserved key
``"__t"``:

==========  ===========================================================  ======================
tag         fields                                                       decodes to
==========  ===========================================================  ======================
``tuple``   ``v``: list of values                                        ``tuple``
``dict``    ``v``: list of ``[key, value]`` pairs                         ``dict`` (any hashable
                                                                         keys, or one using
                                                                         ``"__t"`` as a key)
``float``   ``v``: ``"nan"``, ``"inf"`` or ``"-inf"``                    non-finite ``float``
``bytes``   ``b``: buffer index                                          ``bytes``
``nd``      ``d``: dtype string, ``s``: shape, ``b``: buffer index;      array of the receiver's
            or ``c``: codec, ``o``: codec options, ``b``: index list     backend and device
``space``   ``v``: ``space_to_json`` document                            ``Space``
``graph``   ``v``: dict of the ``GraphInstance`` fields                  ``GraphInstance``
==========  ===========================================================  ======================

NumPy scalars travel as 0-d arrays; torch / JAX arrays are sent as NumPy arrays.
Anything else raises ``TypeError``: no arbitrary objects are ever reconstructed.
Decoding validates every frame (sizes, buffer references, dtype allowlist,
shape x itemsize == length, nesting depth) and raises ``WireError`` otherwise.
"""

from __future__ import annotations

import json
import math
import re
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from unienv_interface.backends import get_backend_from_tensor
from unienv_interface.space import Space
from unienv_interface.space.spaces.graph import GraphInstance
from unienv_interface.space.space_utils.serialization_utils import json_to_space, space_to_json

if TYPE_CHECKING:
    from unienv_interface.backends import ComputeBackend

__all__ = [
    "WireError", "SpaceJSON", "CodecArray", "ArrayCodec", "register_codec", "get_codec",
    "encode", "split", "parse", "restore", "dumps", "loads", "DEFAULT_MAX_FRAME_SIZE",
]

DEFAULT_MAX_FRAME_SIZE: int = 256 * 1024 * 1024
MAX_DEPTH: int = 64
MAX_BUFFERS: int = 1 << 16
MAX_NDIM: int = 32
TAG: str = "__t"

_HEAD = struct.Struct("<II")
_GRAPH_FIELDS: Tuple[str, ...] = ("n_nodes", "n_edges", "nodes_features", "edges_features", "edges")
_DTYPE_RE = re.compile(r"[<>|=]?(b1|[iu][1248]|f[248]|c(8|16))")
_DTYPES: Dict[str, np.dtype] = {}
_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
_SMALL_ARRAY_BYTES = 2048  # below this, ``tobytes`` is cheaper than a zero-copy view
_dumps = json.JSONEncoder(allow_nan=False, check_circular=False, separators=(",", ":")).encode


class WireError(ValueError):
    """A frame is malformed or violates the wire format's limits."""


class SpaceJSON:
    """An already serialized space (``space_to_json`` output); encodes as a ``space`` tag."""

    __slots__ = ("document",)

    def __init__(self, document: Dict[str, Any]) -> None:
        self.document = document


class CodecArray:
    """An array leaf the encoder compresses with a registered :class:`ArrayCodec`."""

    __slots__ = ("array", "codec", "options")

    def __init__(self, array: np.ndarray, codec: str, options: Dict[str, Any]) -> None:
        self.array, self.codec, self.options = array, codec, options


class ArrayCodec:
    """Interface of a pluggable array codec (see :func:`register_codec`)."""

    def encode(self, array: np.ndarray, options: Dict[str, Any]) -> List[bytes]:
        raise NotImplementedError

    def decode(self, blobs: Sequence[memoryview], dtype: np.dtype, shape: Tuple[int, ...],
               options: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


_CODECS: Dict[str, ArrayCodec] = {}


def register_codec(name: str, codec: ArrayCodec) -> None:
    """Register an array codec under ``name`` (needed on both peers)."""
    _CODECS[name] = codec


def get_codec(name: Any) -> ArrayCodec:
    from . import compression  # noqa: F401  registers the built-in codecs
    codec = _CODECS.get(name) if isinstance(name, str) else None
    if codec is None:
        raise WireError(f"Unknown array codec {name!r}")
    return codec


def _wire_dtype(dtype: np.dtype) -> bool:
    return dtype.fields is None and _DTYPE_RE.fullmatch(dtype.str) is not None


def _dtype(text: Any) -> np.dtype:
    dtype = _DTYPES.get(text) if isinstance(text, str) else None
    if dtype is None:
        if not isinstance(text, str) or _DTYPE_RE.fullmatch(text) is None:
            raise WireError(f"Unsupported array dtype {text!r}")
        dtype = _DTYPES[text] = np.dtype(text)
    return dtype


def _shape(shape: Any) -> Tuple[int, ...]:
    if type(shape) is not list or len(shape) > MAX_NDIM or any(type(n) is not int or n < 0 for n in shape):
        raise WireError(f"Invalid array shape {shape!r}")
    return tuple(shape)


# ---------------------------------------------------------------------------- encoding

class _Encoder:
    __slots__ = ("buffers", "sizes", "copy")

    def __init__(self, copy: bool) -> None:
        self.buffers: List[Any] = []
        self.sizes: List[int] = []
        self.copy = copy

    def attach(self, data: Any, size: int) -> int:
        self.buffers.append(data)
        self.sizes.append(size)
        return len(self.buffers) - 1

    def array(self, array: np.ndarray) -> Dict[str, Any]:
        dtype = array.dtype
        if not _wire_dtype(dtype):
            raise TypeError(f"Cannot send arrays of dtype {dtype}")
        if self.copy or array.nbytes < _SMALL_ARRAY_BYTES:
            data = array.tobytes()
        else:
            # Zero-copy: the caller sends the frame before the array can change.
            data = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
        return {TAG: "nd", "d": dtype.str, "s": list(array.shape), "b": self.attach(data, array.nbytes)}

    def value(self, value: Any, depth: int) -> Any:
        kind = type(value)
        if kind is str or kind is int or kind is bool or value is None:
            return value
        if kind is float:
            if math.isfinite(value):
                return value
            return {TAG: "float", "v": "nan" if value != value else ("inf" if value > 0 else "-inf")}
        if depth >= MAX_DEPTH:
            raise ValueError(f"Values nested deeper than {MAX_DEPTH} levels cannot be sent")
        depth += 1
        if kind is np.ndarray:
            return self.array(value)
        if isinstance(value, dict):
            if all(type(key) is str for key in value) and TAG not in value:
                return {key: self.value(item, depth) for key, item in value.items()}
            return {TAG: "dict", "v": [[self.value(key, depth), self.value(item, depth)] for key, item in value.items()]}
        if isinstance(value, list):
            return [self.value(item, depth) for item in value]
        if isinstance(value, tuple):
            return {TAG: "tuple", "v": [self.value(item, depth) for item in value]}
        if isinstance(value, np.generic):
            return self.array(np.asarray(value))
        if isinstance(value, (bool, int, str)):  # subclasses, e.g. IntEnum
            return self.value((bool if isinstance(value, bool) else int if isinstance(value, int) else str)(value), depth)
        if isinstance(value, float):
            return self.value(float(value), depth)
        if isinstance(value, (bytes, bytearray, memoryview)):
            data = bytes(value) if self.copy or not isinstance(value, bytes) else value
            return {TAG: "bytes", "b": self.attach(data, len(data))}
        if isinstance(value, Space):
            try:
                document = space_to_json(value)
            except NotImplementedError as exc:
                raise TypeError(str(exc)) from exc
            return {TAG: "space", "v": self.value(document, depth)}
        if isinstance(value, SpaceJSON):
            return {TAG: "space", "v": self.value(value.document, depth)}
        if isinstance(value, GraphInstance):
            return {TAG: "graph", "v": {name: self.value(getattr(value, name), depth) for name in _GRAPH_FIELDS}}
        if isinstance(value, CodecArray):
            array = value.array
            blobs = get_codec(value.codec).encode(array, value.options)
            return {TAG: "nd", "d": array.dtype.str, "s": list(array.shape), "c": value.codec,
                    "o": value.options, "b": [self.attach(blob, len(blob)) for blob in blobs]}
        try:
            array = get_backend_from_tensor(value).to_numpy(value)
        except Exception:
            raise TypeError(f"Cannot send a value of type {type(value).__module__}.{type(value).__qualname__}") from None
        return self.array(np.asarray(array))


def encode(value: Any, *, copy: bool = False, max_size: Optional[int] = None) -> Tuple[List[Any], int]:
    """Encode ``value`` into frame parts ``[header, json, *buffers]`` and the frame size.

    Large arrays are attached zero-copy unless ``copy`` is set, so the parts must be
    sent before those arrays change. Raises ``TypeError`` for unsupported values.
    """
    encoder = _Encoder(copy)
    document = _dumps(encoder.value(value, 0)).encode()
    sizes = encoder.sizes
    header = _HEAD.pack(len(sizes), len(document))
    if sizes:
        header += struct.pack(f"<{len(sizes)}Q", *sizes)
    total = len(header) + len(document) + sum(sizes)
    if max_size is not None and total > max_size:
        raise ValueError(f"Frame of {total} bytes exceeds the {max_size}-byte limit")
    return [header, document, *encoder.buffers], total


def dumps(value: Any) -> bytes:
    """Encode ``value`` as one contiguous frame."""
    return b"".join(encode(value, copy=True)[0])


# ---------------------------------------------------------------------------- decoding

def read_header(head: bytes, max_size: int) -> Tuple[int, int]:
    """Validate the fixed 8-byte header; return ``(buffer count, JSON length)``."""
    count, length = _HEAD.unpack(head)
    if count > MAX_BUFFERS or length > max_size:
        raise WireError("Frame header exceeds the size limits")
    return count, length


def split(data: Any, max_size: int = DEFAULT_MAX_FRAME_SIZE) -> Tuple[memoryview, List[memoryview]]:
    """Split one contiguous (preferably writable) frame into its JSON and buffers."""
    view = memoryview(data).cast("B")
    if len(view) > max_size:
        raise WireError(f"Frame of {len(view)} bytes exceeds the {max_size}-byte limit")
    if len(view) < 8:
        raise WireError("Truncated frame header")
    count, length = read_header(bytes(view[:8]), max_size)
    offset = 8 + 8 * count
    if offset > len(view):
        raise WireError("Truncated frame header")
    sizes = struct.unpack(f"<{count}Q", view[8:offset]) if count else ()
    if offset + length + sum(sizes) != len(view):
        raise WireError("Frame length does not match its header")
    document = view[offset:offset + length]
    offset += length
    buffers = []
    for size in sizes:
        buffers.append(view[offset:offset + size])
        offset += size
    return document, buffers


def _reject_constant(name: str) -> Any:
    raise WireError(f"Non-standard JSON constant {name}")


def parse(document: Any) -> Any:
    """Parse the JSON part of a frame (strict JSON; tags are not resolved yet)."""
    try:
        return json.loads(bytes(document), parse_constant=_reject_constant)
    except WireError:
        raise
    except (ValueError, RecursionError) as exc:
        raise WireError(f"Invalid frame JSON: {exc}") from None


def _to_backend(array: np.ndarray, backend: ComputeBackend, device: Optional[object]) -> Any:
    if not array.dtype.isnative:
        array = array.astype(array.dtype.newbyteorder("="))
    dtype = backend.__array_namespace_info__().dtypes().get(array.dtype.name)
    if dtype is None:
        raise TypeError(f"{backend.simplified_name} cannot represent dtype {array.dtype.name}")
    return backend.from_numpy(array, dtype=dtype, device=device)


class _Decoder:
    __slots__ = ("buffers", "used", "backend", "device", "convert")

    def __init__(self, buffers: Sequence[Any], backend: Optional[ComputeBackend], device: Optional[object]) -> None:
        self.buffers = buffers
        self.used = [False] * len(buffers)
        self.backend, self.device = backend, device
        self.convert = backend is not None and backend.simplified_name != "numpy"

    def take(self, index: Any) -> Any:
        if type(index) is not int or not 0 <= index < len(self.buffers) or self.used[index]:
            raise WireError(f"Invalid buffer reference {index!r}")
        self.used[index] = True
        return self.buffers[index]

    def array(self, node: Dict[str, Any]) -> Any:
        dtype, shape = _dtype(node.get("d")), _shape(node.get("s"))
        codec = node.get("c")
        if codec is None:
            data = self.take(node.get("b"))
            if math.prod(shape) * dtype.itemsize != len(data):
                raise WireError("Array shape and dtype do not match the buffer length")
            array = np.frombuffer(data, dtype=dtype).reshape(shape)
        else:
            indices, options = node.get("b"), node.get("o", {})
            if type(indices) is not list or type(options) is not dict:
                raise WireError("Invalid codec array")
            blobs = [self.take(index) for index in indices]
            array = get_codec(codec).decode(blobs, dtype, shape, options)
            if array.dtype != dtype or array.shape != shape:
                raise WireError(f"Codec {codec!r} returned a mismatched array")
        return _to_backend(array, self.backend, self.device) if self.convert else array

    def value(self, node: Any, depth: int) -> Any:
        kind = type(node)
        if kind is not list and kind is not dict:
            return node
        if depth >= MAX_DEPTH:
            raise WireError(f"Frame nesting exceeds {MAX_DEPTH} levels")
        depth += 1
        if kind is list:
            return [self.value(item, depth) for item in node]
        tag = node.get(TAG)
        if tag is None:
            return {key: self.value(item, depth) for key, item in node.items()}
        if tag == "nd":
            return self.array(node)
        if tag == "tuple":
            return tuple(self.value(item, depth) for item in self._list(node))
        if tag == "dict":
            result = {}
            for pair in self._list(node):
                if type(pair) is not list or len(pair) != 2:
                    raise WireError("Invalid dict entry")
                try:
                    result[self.value(pair[0], depth)] = self.value(pair[1], depth)
                except TypeError as exc:
                    raise WireError(f"Invalid dict key: {exc}") from None
            return result
        if tag == "float":
            if node.get("v") not in _NONFINITE:
                raise WireError("Invalid non-finite float")
            return _NONFINITE[node["v"]]
        if tag == "bytes":
            return bytes(self.take(node.get("b")))
        if tag == "space":
            document = self.value(node.get("v"), depth)
            from unienv_interface.backends.numpy import NumpyComputeBackend
            try:
                return json_to_space(document, self.backend or NumpyComputeBackend, self.device)
            except Exception as exc:
                raise WireError(f"Invalid space: {exc}") from None
        if tag == "graph":
            fields = node.get("v")
            if type(fields) is not dict or set(fields) != set(_GRAPH_FIELDS):
                raise WireError("Invalid graph fields")
            return GraphInstance(**{name: self.value(fields[name], depth) for name in _GRAPH_FIELDS})
        raise WireError(f"Unknown tag {tag!r}")

    @staticmethod
    def _list(node: Dict[str, Any]) -> List[Any]:
        items = node.get("v")
        if type(items) is not list:
            raise WireError("Invalid tagged container")
        return items


def restore(node: Any, buffers: Sequence[Any], backend: Optional[ComputeBackend] = None,
            device: Optional[object] = None) -> Any:
    """Resolve the tags of a parsed frame; arrays become ``backend`` arrays on ``device``.

    Every buffer must be referenced exactly once. Arrays share memory with the
    (writable) buffers when no backend conversion is needed.
    """
    decoder = _Decoder(buffers, backend, device)
    value = decoder.value(node, 0)
    if not all(decoder.used):
        raise WireError("Frame has unreferenced buffers")
    return value


def loads(data: Any, backend: Optional[ComputeBackend] = None, device: Optional[object] = None,
          max_size: int = DEFAULT_MAX_FRAME_SIZE) -> Any:
    """Decode one contiguous frame (copied once so that arrays are writable)."""
    document, buffers = split(bytearray(data), max_size)
    return restore(parse(document), buffers, backend, device)
