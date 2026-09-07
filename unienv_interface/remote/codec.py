"""Portable v1 binary envelope. No Python object deserialization is performed."""

from __future__ import annotations

import json
import math
import struct
from typing import Any, Dict, List, NoReturn, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from unienv_interface.backends import get_backend_from_tensor
from unienv_interface.space.spaces.graph import GraphInstance

if TYPE_CHECKING:
    from unienv_interface.backends import ComputeBackend

PROTOCOL_VERSION: int = 1
DEFAULT_MAX_MESSAGE_SIZE: int = 64 * 1024 * 1024
_DTYPE_SIZES: Dict[str, Set[int]] = {"b": {1}, "i": {1, 2, 4, 8}, "u": {1, 2, 4, 8}, "f": {2, 4, 8}, "c": {8, 16}}
_GRAPH_FIELDS: Tuple[str, ...] = ("n_nodes", "n_edges", "nodes_features", "edges_features", "edges")


def _valid_dtype(dtype: np.dtype) -> bool:
    return dtype.fields is None and dtype.itemsize in _DTYPE_SIZES.get(dtype.kind, ())


class CodecError(ValueError):
    """A value or envelope cannot be represented by the v1 codec."""


class Codec:
    """Encode a tagged value tree and contiguous binary attachments.

    Limits apply to the whole envelope, on both send and receive. Arrays decode
    to independently owned NumPy memory; clients may subsequently map backends.
    """

    def __init__(self, max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE) -> None:
        if max_message_size < 8:
            raise ValueError("max_message_size must be at least 8")
        self.max_message_size = max_message_size

    def encode(self, value: Any) -> bytes:
        attachments: List[bytes] = []
        offset = 0

        def encode_value(value: Any, depth: int = 0) -> Any:
            nonlocal offset
            if depth > 64:
                raise CodecError("Maximum value nesting exceeded")
            if value is None or isinstance(value, (str, bool, int)):
                return value
            if isinstance(value, float):
                if math.isfinite(value):
                    return value
                return {"t": "float", "v": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
            if isinstance(value, np.generic):
                return encode_value(np.asarray(value), depth + 1)
            if isinstance(value, GraphInstance):
                return {"t": "graph", "v": encode_value({k: getattr(value, k) for k in _GRAPH_FIELDS}, depth + 1)}
            if isinstance(value, np.ndarray) and value.dtype == object:
                if value.nbytes > self.max_message_size:
                    raise CodecError("Object container size limit exceeded")
                return {"t": "object_array", "shape": list(value.shape),
                        "v": [encode_value(v, depth + 1) for v in value.flat]}
            if isinstance(value, dict):
                if not all(isinstance(k, str) for k in value):
                    raise CodecError("Dictionary keys must be strings")
                return {"t": "dict", "v": [[k, encode_value(v, depth + 1)] for k, v in value.items()]}
            if isinstance(value, (list, tuple)):
                return {"t": "tuple" if isinstance(value, tuple) else "list",
                        "v": [encode_value(v, depth + 1) for v in value]}
            if isinstance(value, bytes):
                node: Dict[str, Any] = {"t": "bytes", "offset": offset, "length": len(value)}
                data = value
            else:
                try:
                    array = value if isinstance(value, np.ndarray) else get_backend_from_tensor(value).to_numpy(value)
                except Exception as exc:
                    raise CodecError(f"Unsupported value type: {type(value).__name__}") from exc
                if not _valid_dtype(array.dtype):
                    raise CodecError(f"Unsupported array dtype: {array.dtype}")
                if array.nbytes > self.max_message_size - offset:
                    raise CodecError("Message size limit exceeded")
                node = {"t": "array", "dtype": array.dtype.str, "shape": list(array.shape),
                        "offset": offset, "length": array.nbytes}
                data = array.tobytes(order="C")
            offset += len(data)
            if offset > self.max_message_size:
                raise CodecError("Message size limit exceeded")
            attachments.append(data)
            return node

        try:
            header = json.dumps(encode_value(value), allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise CodecError(str(exc)) from exc
        if 4 + len(header) + offset > self.max_message_size:
            raise CodecError("Message size limit exceeded")
        return struct.pack("!I", len(header)) + header + b"".join(attachments)

    def decode(self, payload: bytes) -> Any:
        if not isinstance(payload, bytes) or not 4 <= len(payload) <= self.max_message_size:
            raise CodecError("Invalid envelope size or type")
        header_size = struct.unpack("!I", payload[:4])[0]
        if header_size > len(payload) - 4:
            raise CodecError("Truncated header")
        binary = memoryview(payload)[4 + header_size:]
        offset = 0

        def decode_value(node: Any, depth: int = 0) -> Any:
            nonlocal offset
            if depth > 64:
                raise CodecError("Maximum value nesting exceeded")
            if node is None or isinstance(node, (str, bool, int)):
                return node
            if isinstance(node, float) and math.isfinite(node):
                return node
            if not isinstance(node, dict):
                raise CodecError("Invalid value node")
            kind = node.get("t")
            if kind == "graph":
                fields = decode_value(node["v"], depth + 1)
                if not isinstance(fields, dict) or set(fields) != set(_GRAPH_FIELDS):
                    raise CodecError("Invalid graph fields")
                return GraphInstance(**fields)
            if kind == "object_array":
                shape, values = node["shape"], node["v"]
                self._validate_shape(shape)
                size = math.prod(shape)
                if not isinstance(values, list) or len(values) != size or size * np.dtype(object).itemsize > self.max_message_size:
                    raise CodecError("Invalid object container size")
                result: Any = np.empty(shape, dtype=object)
                for index, value in enumerate(values):
                    result.flat[index] = decode_value(value, depth + 1)
                return result
            if kind == "float":
                return {"nan": float("nan"), "inf": float("inf"), "-inf": -float("inf")}[node["v"]]
            if kind in ("list", "tuple"):
                if not isinstance(node["v"], list):
                    raise CodecError("Invalid sequence")
                values = [decode_value(v, depth + 1) for v in node["v"]]
                return tuple(values) if kind == "tuple" else values
            if kind == "dict":
                result = {}
                for key, value in node["v"]:
                    if not isinstance(key, str) or key in result:
                        raise CodecError("Invalid or duplicate dictionary key")
                    result[key] = decode_value(value, depth + 1)
                return result
            if kind not in ("array", "bytes"):
                raise CodecError("Unknown value tag")
            start, length = node["offset"], node["length"]
            if type(start) is not int or type(length) is not int or start != offset or length < 0 or length > len(binary) - start:
                raise CodecError("Invalid attachment bounds")
            offset += length
            data = binary[start:offset]
            if kind == "bytes":
                return bytes(data)
            dtype = np.dtype(node["dtype"])
            shape = node["shape"]
            if not _valid_dtype(dtype):
                raise CodecError("Unsupported array dtype")
            self._validate_shape(shape)
            if math.prod(shape) * dtype.itemsize != length:
                raise CodecError("Array shape does not match attachment length")
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

        try:
            def reject_constant(value: str) -> NoReturn:
                raise CodecError(f"Nonstandard JSON constant: {value}")

            header = json.loads(payload[4:4 + header_size], parse_constant=reject_constant)
            result = decode_value(header)
            if offset != len(binary):
                raise CodecError("Unused attachment data")
            return result
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            raise CodecError(str(exc)) from exc

    @staticmethod
    def _validate_shape(shape: object) -> None:
        if not isinstance(shape, list) or len(shape) > 32 or any(type(n) is not int or n < 0 or n > np.iinfo(np.intp).max for n in shape):
            raise CodecError("Invalid array shape")


def map_arrays(value: Any, backend: ComputeBackend, device: Optional[object] = None) -> Any:
    """Copy received arrays into the chosen local compute backend."""
    if isinstance(value, GraphInstance):
        return GraphInstance(**{k: map_arrays(getattr(value, k), backend, device) for k in _GRAPH_FIELDS})
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            result: Any = np.empty(value.shape, dtype=object)
            for index, item in enumerate(value.flat):
                result.flat[index] = map_arrays(item, backend, device)
            return result
        if backend.simplified_name == "numpy":
            return backend.from_numpy(value, device=device)
        # Non-native byte order is a wire concern; tensor backends use native order.
        if not value.dtype.isnative:
            value = value.astype(value.dtype.newbyteorder("="))
        target_dtype = backend.__array_namespace_info__().dtypes().get(value.dtype.name)
        if target_dtype is None:
            raise CodecError(f"Backend cannot represent dtype {value.dtype.name}")
        result = backend.from_numpy(value, dtype=target_dtype, device=device)
        if result.dtype != target_dtype or tuple(result.shape) != value.shape:
            raise CodecError("Backend conversion changed dtype or shape")
        return result
    if isinstance(value, dict):
        return {k: map_arrays(v, backend, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(map_arrays(v, backend, device) for v in value)
    return value
