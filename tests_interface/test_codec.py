"""Tests for the portable binary envelope in ``unienv_interface.remote.codec``."""

import json
import math
import struct

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp

from unienv_interface.remote.codec import Codec, CodecError, map_arrays
from unienv_interface.space import GraphInstance
from tests_common.backends import cpu_backend_params
from tests_common.space_strategies import build_space, seeds, space_specs
from tests_common.trees import assert_tree_equal


def _envelope(header, binary=b""):
    raw = json.dumps(header).encode("utf-8")
    return struct.pack("!I", len(raw)) + raw + binary


def _assert_same(actual, expected):
    """Like assert_tree_equal but also checks list/tuple and float NaN identity."""
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_same(a, e)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and list(actual) == list(expected)
        for k in expected:
            _assert_same(actual[k], expected[k])
    elif isinstance(expected, float) and math.isnan(expected):
        assert isinstance(actual, float) and math.isnan(actual)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
        if expected.dtype == object:
            assert actual.shape == expected.shape
            for a, e in zip(actual.flat, expected.flat):
                _assert_same(a, e)
        else:
            np.testing.assert_array_equal(actual, expected)
    else:
        assert type(actual) is type(expected) and actual == expected


# ---------------------------------------------------------------------------
# Hypothesis strategies for codec values
# ---------------------------------------------------------------------------

_dtypes = st.sampled_from([
    np.dtype(t) for t in ("?", "i1", "u1", "i2", "<u2", ">i4", "<i4", "u4", "<i8", ">u8",
                          "<f2", "<f4", ">f4", "<f8", ">f8", "<c8", ">c16")
])
_arrays = _dtypes.flatmap(lambda dt: hnp.arrays(dt, hnp.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=4)))
_scalars = st.one_of(
    st.none(), st.booleans(), st.integers(min_value=-(2 ** 70), max_value=2 ** 70),
    st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=8), st.binary(max_size=8),
)
_keys = st.text(max_size=5)


def _containers(children):
    return st.one_of(
        st.lists(children, max_size=3),
        st.lists(children, max_size=3).map(tuple),
        st.dictionaries(_keys, children, max_size=3),
    )


codec_values = st.recursive(st.one_of(_scalars, _arrays), _containers, max_leaves=8)


@given(codec_values)
def test_roundtrip_preserves_values_and_container_types(value):
    codec = Codec()
    _assert_same(codec.decode(codec.encode(value)), value)


@given(codec_values, st.data())
def test_mutated_or_truncated_payloads_only_raise_codec_error(value, data):
    codec = Codec()
    payload = codec.encode(value)
    mutation = data.draw(st.sampled_from(["truncate", "flip", "append"]))
    if mutation == "truncate":
        cut = data.draw(st.integers(0, len(payload) - 1))
        with pytest.raises(CodecError):
            codec.decode(payload[:cut])
        return
    if mutation == "append":
        with pytest.raises(CodecError):
            codec.decode(payload + data.draw(st.binary(min_size=1, max_size=4)))
        return
    position = data.draw(st.integers(0, len(payload) - 1))
    flipped = bytearray(payload)
    flipped[position] ^= data.draw(st.integers(1, 255))
    try:
        codec.decode(bytes(flipped))
    except CodecError:
        pass  # Any other exception type fails the test


@given(space_specs(), seeds)
def test_roundtrip_of_sampled_space_data(spec, seed):
    from unienv_interface.backends.numpy import NumpyComputeBackend as B
    space = build_space(spec, B)
    codec = Codec()
    rng = B.random.random_number_generator(seed)
    for _ in range(2):
        rng, x = space.sample(rng)
        decoded = codec.decode(codec.encode(x))
        assert space.contains(decoded)
        assert_tree_equal(decoded, x)


# ---------------------------------------------------------------------------
# Specific encodings
# ---------------------------------------------------------------------------

def test_numpy_scalars_decode_as_zero_dim_arrays():
    codec = Codec()
    for value in (np.float32(1.5), np.int64(-3), np.uint8(7), np.bool_(True), np.complex64(1 + 2j)):
        decoded = codec.decode(codec.encode(value))
        assert isinstance(decoded, np.ndarray) and decoded.shape == () and decoded.dtype == value.dtype
        assert decoded == value
    # Subclasses of Python scalars stay Python scalars
    assert type(codec.decode(codec.encode(np.float64(2.5)))) is float
    assert type(codec.decode(codec.encode(np.str_("ab")))) is str


def test_non_native_byte_order_and_non_contiguous_arrays():
    codec = Codec()
    big = np.arange(6, dtype=">f8").reshape(2, 3)
    decoded = codec.decode(codec.encode(big))
    assert decoded.dtype == np.dtype(">f8")
    np.testing.assert_array_equal(decoded, big)
    strided = np.arange(20, dtype=np.int32).reshape(4, 5)[::2, 1::2]
    np.testing.assert_array_equal(codec.decode(codec.encode(strided)), strided)
    decoded.flags.writeable = True  # decoded arrays own their memory


def test_object_arrays_and_graph_instances():
    codec = Codec()
    obj = np.empty((2, 2), dtype=object)
    obj[0, 0], obj[0, 1], obj[1, 0], obj[1, 1] = "a", (1, np.zeros(2)), None, {"k": [1]}
    _assert_same(codec.decode(codec.encode(obj)), obj)
    graph = GraphInstance(n_nodes=np.array(3), nodes_features=np.ones((3, 2), dtype=np.float32),
                          n_edges=np.array(1), edges=np.array([[0, 2]]))
    decoded = codec.decode(codec.encode({"g": graph}))["g"]
    assert isinstance(decoded, GraphInstance)
    assert_tree_equal(decoded, graph)


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_backend_arrays_encode_and_map_back(backend, device):
    codec = Codec()
    x = backend.asarray(np.arange(6, dtype=np.float32).reshape(2, 3), device=device)
    value = {"x": x, "t": (x, [x])}
    decoded = codec.decode(codec.encode(value))
    assert isinstance(decoded["x"], np.ndarray) and isinstance(decoded["t"], tuple)
    mapped = map_arrays(decoded, backend, device)
    assert backend.is_backendarray(mapped["x"]) and isinstance(mapped["t"][1], list)
    assert_tree_equal(mapped, value)


def test_map_arrays_handles_object_arrays_graphs_and_byte_order():
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as T
    obj = np.empty((1,), dtype=object)
    obj[0] = np.arange(3, dtype=">i4")
    graph = GraphInstance(n_nodes=np.array(2), nodes_features=np.ones((2, 1), dtype=">f4"))
    mapped = map_arrays({"o": obj, "g": graph, "s": "keep"}, T)
    assert T.is_backendarray(mapped["o"][0]) and mapped["o"][0].tolist() == [0, 1, 2]
    assert isinstance(mapped["g"], GraphInstance) and mapped["g"].n_edges is None
    assert mapped["s"] == "keep"


def test_depth_limit_is_symmetric():
    codec = Codec()

    def nest(value, depth):
        for _ in range(depth):
            value = [value]
        return value

    for leaf in (1, np.zeros(1), np.float32(1.0), "s"):
        codec.decode(codec.encode(nest(leaf, 64)))
        with pytest.raises(CodecError):
            codec.encode(nest(leaf, 65))
    # A header nested deeper than the limit is rejected on decode, too.
    header = {"t": "list", "v": []}
    for _ in range(70):
        header = {"t": "list", "v": [header]}
    with pytest.raises(CodecError):
        codec.decode(_envelope(header))


def test_size_limits():
    with pytest.raises(ValueError):
        Codec(4)
    small = Codec(64)
    with pytest.raises(CodecError):
        small.encode(b"x" * 100)
    with pytest.raises(CodecError):
        small.encode("y" * 100)
    with pytest.raises(CodecError):
        small.decode(bytes(65))
    with pytest.raises(CodecError):
        small.encode(np.empty(20, dtype=object))
    with pytest.raises(CodecError):
        Codec().decode(bytearray(Codec().encode(1)))  # only bytes are accepted


@pytest.mark.parametrize("header,binary", [
    ({"t": "dict", "v": [["a", 1], ["a", 2]]}, b""),             # duplicate key
    ({"t": "dict", "v": [[1, 2]]}, b""),                          # non-string key
    ({"t": "dict", "v": {"ab": 1}}, b""),                         # mapping instead of pairs
    ({"t": "dict", "v": ["ab"]}, b""),                            # string instead of a pair
    ({"t": "dict", "v": [["a", 1, 2]]}, b""),                     # pair of wrong length
    ({"t": "tuple", "v": "ab"}, b""),
    ({"t": "graph", "v": {"t": "dict", "v": [["n_nodes", 1]]}}, b""),
    ({"t": "object_array", "shape": [2], "v": [1]}, b""),
    ({"t": "float", "v": "huge"}, b""),
    ({"t": "array", "dtype": None, "shape": [1], "offset": 0, "length": 8}, b"\0" * 8),
    ({"t": "array", "dtype": ["<f8"], "shape": [1], "offset": 0, "length": 8}, b"\0" * 8),
    ({"t": "array", "dtype": "<f4", "shape": [3], "offset": 0, "length": 8}, b"\0" * 8),
    ({"t": "array", "dtype": "<f4", "shape": [-1], "offset": 0, "length": 4}, b"\0" * 4),
    ({"t": "array", "dtype": "<M8", "shape": [1], "offset": 0, "length": 8}, b"\0" * 8),
    ({"t": "bytes", "offset": 1, "length": 1}, b"\0\0"),           # skipped attachment byte
    ({"t": "bytes", "offset": 0, "length": 1}, b"\0\0"),           # unused attachment data
    ({"t": "bytes", "offset": True, "length": 1}, b"\0"),
    ([1.5, {"t": "unknown"}], b""),
    ({"no_tag": 1}, b""),
])
def test_malformed_payloads_raise_codec_error(header, binary):
    with pytest.raises(CodecError):
        Codec().decode(_envelope(header, binary))


def test_nonstandard_json_constants_and_bad_utf8_rejected():
    codec = Codec()
    raw = b'{"t":"list","v":[NaN]}'
    with pytest.raises(CodecError):
        codec.decode(struct.pack("!I", len(raw)) + raw)
    raw = b"\xff\xfe\xfd"
    with pytest.raises(CodecError):
        codec.decode(struct.pack("!I", len(raw)) + raw)
    with pytest.raises(CodecError):
        codec.decode(b"\x00\x00")


# ---------------------------------------------------------------------------
# Zero-copy attachments must not change the wire format
# ---------------------------------------------------------------------------

def _tobytes_codec_encode(value, monkeypatch):
    """Encode with every array attached via ``tobytes`` (the original implementation)."""
    import unienv_interface.remote.codec as codec_module
    with monkeypatch.context() as patch:
        patch.setattr(codec_module, "_ZERO_COPY_MIN_BYTES", float("inf"))
        return Codec().encode(value)


_large_arrays = _dtypes.flatmap(lambda dt: hnp.arrays(
    dt, hnp.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=40)))
_large_values = st.recursive(st.one_of(_scalars, _large_arrays), _containers, max_leaves=6)


@given(_large_values)
def test_zero_copy_attachments_are_byte_identical(value):
    monkeypatch = pytest.MonkeyPatch()
    assert Codec().encode(value) == _tobytes_codec_encode(value, monkeypatch)


def test_zero_copy_attachments_byte_identical_on_varied_layouts(monkeypatch):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (84, 84, 3), dtype=np.uint8)
    graph = GraphInstance(n_nodes=np.array(600), nodes_features=rng.random((600, 4)).astype(np.float32),
                          n_edges=np.array(2), edges=np.array([[0, 2], [1, 3]]))
    value = {
        "image": image,
        "fortran": np.asfortranarray(rng.random((64, 32))),
        "strided": np.arange(40000, dtype=np.int32).reshape(200, 200)[::2, 1::3],
        "big_endian": np.arange(3000, dtype=">f4"),
        "complex": (rng.random(600) + 1j * rng.random(600)).astype(">c16"),
        "bools": rng.random(5000) > 0.5,
        "half": rng.random(4096).astype(np.float16),
        "scalar": np.float32(1.5),
        "small": np.arange(3, dtype=np.int64),
        "boundary": np.zeros(4096, np.uint8),
        "below_boundary": np.zeros(4095, np.uint8),
        "graph": graph,
        "nested": [(image[::2], b"x" * 5000), {"blob": b"\x00\x01"}],
    }
    payload = Codec().encode(value)
    assert payload == _tobytes_codec_encode(value, monkeypatch)
    _assert_same(Codec().decode(payload)["nested"][0][1], b"x" * 5000)
    np.testing.assert_array_equal(Codec().decode(payload)["strided"], value["strided"])


def test_zero_copy_envelope_matches_hand_built_layout():
    image = np.arange(84 * 84 * 3, dtype=np.uint16).reshape(84, 84, 3)
    header = {"t": "dict", "v": [["image", {"t": "array", "dtype": "<u2", "shape": [84, 84, 3],
                                            "offset": 0, "length": image.nbytes}]]}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    expected = struct.pack("!I", len(raw)) + raw + image.tobytes()
    assert Codec().encode({"image": image}) == expected


def test_zero_copy_encode_snapshots_array_memory():
    array = np.zeros(10000, np.float32)
    payload = Codec().encode(array)
    array[:] = 1.0  # later writes must not leak into an already encoded payload
    np.testing.assert_array_equal(Codec().decode(payload), 0.0)


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_zero_copy_backend_arrays_byte_identical(backend, device, monkeypatch):
    x = backend.asarray(np.arange(4096, dtype=np.float32).reshape(64, 64), device=device)
    value = {"x": x, "t": [x[::2]]}
    assert Codec().encode(value) == _tobytes_codec_encode(value, monkeypatch)
