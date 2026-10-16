"""Wire format: round trips, backend conversion, strict JSON and malformed-frame rejection."""

from __future__ import annotations

import json
import math
import socket
import struct

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.remote import wire
from unienv_interface.remote.transport import MemoryTransport, StreamTransport, parse_address
from unienv_interface.remote.wire import WireError
from unienv_interface.space import BoxSpace, DictSpace, TupleSpace
from unienv_interface.space.spaces.graph import GraphInstance

from tests_common.backends import backend_params


def roundtrip(value, **kwargs):
    return wire.loads(wire.dumps(value), **kwargs)


def frame(document, buffers=()):
    """A frame with an arbitrary (possibly invalid) JSON document and buffers."""
    body = document if isinstance(document, bytes) else json.dumps(document).encode()
    header = struct.pack("<II", len(buffers), len(body)) + b"".join(struct.pack("<Q", len(b)) for b in buffers)
    return header + body + b"".join(buffers)


def same(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return type(b) is dict and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


VALUES = [
    None, True, False, 0, -2 ** 70, 1.5, -0.0, math.nan, math.inf, -math.inf, "", "ünïcødé", b"", b"\x00\xff",
    [], [1, [2, [3]]], (), (1, "a", None), {}, {"a": 1, "b": [1.0, (2,)]}, {1: "a", (1, 2): "b", None: 3},
    {"__t": "nd", "x": 1}, {"nested": {"__t": 5}},
    np.arange(6, dtype=np.float32).reshape(2, 3), np.array([True, False]), np.array(3.5), np.zeros((0, 4), np.int16),
    np.array([1 + 2j], np.complex64), np.array([1, 2], ">i4"), np.array([np.nan, np.inf], np.float64),
    np.arange(10000, dtype=np.uint8), np.zeros((2, 3), np.float16), np.array([2 ** 63 - 1], np.int64),
]


@pytest.mark.parametrize("value", VALUES, ids=[f"v{i}" for i in range(len(VALUES))])
def test_roundtrip(value):
    assert same(value, roundtrip(value))


def test_conversions_and_writable_arrays():
    data = {"scalar": np.float32(2.5), "int": np.int64(3), "non_contiguous": np.arange(12).reshape(3, 4)[:, ::2],
            "fortran": np.asfortranarray(np.arange(6.0).reshape(2, 3)), "bytearray": bytearray(b"ab"),
            "memoryview": memoryview(b"cd"), "subclass": type("S", (str,), {})("s"), "flag": np.bool_(True)}
    result = roundtrip(data)
    assert result["scalar"].shape == () and result["scalar"].dtype == np.float32 and result["scalar"] == 2.5
    assert result["int"].dtype == np.int64
    np.testing.assert_array_equal(result["non_contiguous"], data["non_contiguous"])
    np.testing.assert_array_equal(result["fortran"], data["fortran"])
    assert result["bytearray"] == b"ab" and result["memoryview"] == b"cd" and type(result["subclass"]) is str
    assert result["flag"].dtype == np.bool_
    result["non_contiguous"][0, 0] = 99  # decoded arrays own writable memory


def test_spaces_and_graphs_roundtrip():
    space = DictSpace(B, {"box": BoxSpace(B, -np.inf, 1.0, np.float32, shape=(2,)),
                          "tuple": TupleSpace(B, [BoxSpace(B, 0, 255, np.uint8, shape=(4, 4, 3))])})
    decoded = roundtrip(space)
    assert type(decoded) is DictSpace and decoded["box"].shape == (2,)
    assert np.isneginf(decoded["box"].low).all() and decoded["tuple"][0].dtype == np.uint8
    graph = GraphInstance(n_nodes=np.array([2]), n_edges=None, nodes_features=np.ones((1, 2, 3)),
                          edges_features=None, edges=np.zeros((1, 0, 2), np.int64))
    decoded = roundtrip(graph)
    assert isinstance(decoded, GraphInstance) and decoded.n_edges is None
    np.testing.assert_array_equal(decoded.nodes_features, graph.nodes_features)


def test_strict_json_and_zero_copy_parts():
    image = np.arange(4096, dtype=np.uint8)
    parts, size = wire.encode({"x": [math.nan, math.inf], "image": image})
    assert size == sum(memoryview(p).nbytes for p in parts) and len(parts) == 3
    count, length = struct.unpack("<II", parts[0][:8])
    assert count == 1 and length == len(parts[1])
    json.loads(parts[1], parse_constant=lambda name: pytest.fail(f"non-standard constant {name}"))
    assert np.shares_memory(parts[2], image)  # large arrays are attached without a copy
    copied = wire.encode({"image": image}, copy=True)[0]
    assert isinstance(copied[2], bytes)


@pytest.mark.parametrize("backend,device", backend_params())
def test_backend_conversion(backend, device):
    array = backend.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3), device=device)
    payload = wire.dumps({"a": array, "flag": backend.from_numpy(np.array([True, False]), device=device),
                          "big_endian": np.array([1, 2], ">i4")})
    decoded = wire.loads(payload, backend=backend, device=device)
    assert backend.is_backendarray(decoded["a"])
    np.testing.assert_array_equal(backend.to_numpy(decoded["a"]), np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(backend.to_numpy(decoded["big_endian"]), [1, 2])
    assert backend.to_numpy(decoded["flag"]).dtype == np.bool_
    if device is not None:
        assert str(device) in str(backend.device(decoded["a"]))
    space = wire.loads(wire.dumps(BoxSpace(B, 0.0, 1.0, np.float32, shape=(2,))), backend=backend, device=device)
    assert space.backend is backend


@pytest.mark.parametrize("value", [object(), {1, 2}, np.array([object()]), np.zeros(2, [("a", "i4")]),
                                   np.array(["text"]), np.array(["2020-01-01"], "datetime64[D]"), print],
                         ids=["object", "set", "object-array", "structured", "unicode", "datetime", "function"])
def test_unsupported_values_raise_type_error(value):
    with pytest.raises(TypeError, match="Cannot send"):
        wire.encode({"value": value})


def test_depth_limits_and_size_limit():
    deep = []
    for _ in range(wire.MAX_DEPTH):
        deep = [deep]
    with pytest.raises(ValueError, match="nested"):
        wire.encode(deep)
    with pytest.raises(WireError, match="nesting"):
        wire.loads(frame(json.dumps(deep).encode()))
    with pytest.raises(WireError, match="JSON"):
        wire.loads(frame(b"[" * 100000 + b"]" * 100000))
    with pytest.raises(ValueError, match="limit"):
        wire.encode(np.zeros(100, np.uint8), max_size=50)
    with pytest.raises(WireError, match="limit"):
        wire.loads(wire.dumps(np.zeros(100, np.uint8)), max_size=50)


ARRAY = {"__t": "nd", "d": "<f4", "s": [2], "b": 0}


@pytest.mark.parametrize("payload,match", [
    (b"\x01\x00", "Truncated"),
    (struct.pack("<II", 5, 0), "Truncated"),
    (struct.pack("<II", 0, 10) + b"[]", "length"),
    (frame([1]) + b"x", "length"),
    (struct.pack("<II", 1 << 20, 0), "limits"),
    (frame(b"[NaN]"), "constant"),
    (frame(b"[Infinity]"), "constant"),
    (frame(b"{not json"), "JSON"),
    (frame(b"\xff\xfe"), "JSON"),
    (frame(ARRAY), "buffer reference"),
    (frame(dict(ARRAY, b=1), [b"\x00" * 8]), "buffer reference"),
    (frame(dict(ARRAY, b="0"), [b"\x00" * 8]), "buffer reference"),
    (frame([ARRAY, ARRAY], [b"\x00" * 8]), "buffer reference"),
    (frame(1, [b"\x00"]), "unreferenced"),
    (frame(dict(ARRAY, d="|O"), [b"\x00" * 8]), "dtype"),
    (frame(dict(ARRAY, d="<U1"), [b"\x00" * 8]), "dtype"),
    (frame(dict(ARRAY, d="V8"), [b"\x00" * 8]), "dtype"),
    (frame(dict(ARRAY, d=4), [b"\x00" * 8]), "dtype"),
    (frame(dict(ARRAY, d="float32"), [b"\x00" * 8]), "dtype"),
    (frame(dict(ARRAY, s=[3]), [b"\x00" * 8]), "buffer length"),
    (frame(dict(ARRAY, s=[-2]), [b"\x00" * 8]), "shape"),
    (frame(dict(ARRAY, s=[2.0]), [b"\x00" * 8]), "shape"),
    (frame(dict(ARRAY, s=[1] * 33), [b"\x00" * 4]), "shape"),
    (frame(dict(ARRAY, s=2), [b"\x00" * 8]), "shape"),
    (frame({"__t": "nd", "d": "|u1", "s": [1], "c": "nope", "b": [0]}, [b"\x00"]), "codec"),
    (frame({"__t": "nd", "d": "|u1", "s": [1], "c": "jpeg", "b": 0}, [b"\x00"]), "codec array"),
    (frame({"__t": "what"}), "Unknown tag"),
    (frame({"__t": "float", "v": "huge"}), "non-finite"),
    (frame({"__t": "tuple", "v": 3}), "container"),
    (frame({"__t": "dict", "v": [[1]]}), "dict entry"),
    (frame({"__t": "dict", "v": [[[1], 2]]}), "dict key"),
    (frame({"__t": "graph", "v": {"n_nodes": 1}}), "graph"),
    (frame({"__t": "space", "v": {"type": "NoSuchSpace"}}), "space"),
    (frame({"__t": "bytes", "b": 0}), "buffer reference"),
], ids=lambda value: value if isinstance(value, str) else "")
def test_malformed_frames(payload, match):
    with pytest.raises(WireError, match=match):
        wire.loads(payload)


def test_codec_registry_and_parse_errors():
    with pytest.raises(WireError, match="codec"):
        wire.get_codec(3)
    with pytest.raises(NotImplementedError):
        wire.ArrayCodec().encode(np.zeros(1), {})
    with pytest.raises(NotImplementedError):
        wire.ArrayCodec().decode([], np.dtype("u1"), (1,), {})


def test_stream_transport_partial_sends_many_parts_and_limits():
    left, right = socket.socketpair()
    sender, receiver = StreamTransport(left), StreamTransport(right, max_frame_size=1 << 20)

    class Trickle:  # a socket whose sendmsg writes only a few bytes
        def __init__(self, sock):
            self.sock = sock

        def sendmsg(self, parts):
            return self.sock.send(bytes(memoryview(parts[0]).cast("B")[:3]))

        def __getattr__(self, name):
            return getattr(self.sock, name)

    sender.sock = Trickle(left)
    value = {"image": np.arange(5000, dtype=np.uint8), "chunks": [b"x%d" % i for i in range(600)]}
    sender.send(*wire.encode(value))
    document, buffers = receiver.recv()
    decoded = wire.restore(wire.parse(document), buffers)
    np.testing.assert_array_equal(decoded["image"], value["image"])
    assert decoded["chunks"] == value["chunks"]
    sender.sock = left
    left.sendall(struct.pack("<IIQ", 1, 0, 1 << 30))
    with pytest.raises(WireError, match="limit"):
        receiver.recv()
    left.sendall(struct.pack("<II", 0, 10) + b"[")
    sender.close()
    with pytest.raises(ConnectionError):
        receiver.recv()
    with pytest.raises(ConnectionError):
        sender.send(*wire.encode(1))
    receiver.close()
    receiver.close()


def test_memory_transport_close():
    left, right = MemoryTransport.pair()
    left.send(*wire.encode([1]))
    assert wire.parse(right.recv()[0]) == [1]
    left.close()
    with pytest.raises(ConnectionError):
        right.recv()
    with pytest.raises(ConnectionError):
        right.send(*wire.encode([1]))


@pytest.mark.parametrize("address,expected", [
    ("tcp://127.0.0.1:80", ("tcp", "127.0.0.1", 80)),
    ("tcp://[::1]:9", ("tcp", "::1", 9)),
    ("ws://localhost:0", ("ws", "localhost", 0)),
    ("unix:///tmp/x.sock", ("unix", "/tmp/x.sock", None)),
    ("unix://relative.sock", ("unix", "relative.sock", None)),
])
def test_parse_address(address, expected):
    assert parse_address(address) == expected


@pytest.mark.parametrize("address", ["http://x:1", "tcp://host", "unix://", "localhost:80"])
def test_parse_address_rejects(address):
    with pytest.raises(ValueError):
        parse_address(address)
