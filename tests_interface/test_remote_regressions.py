"""Regression tests for bugs found in the remote layer (one section per bug)."""

from __future__ import annotations

import json
import struct
import threading
import time

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.remote import RemoteClient, RemoteServer, wire
from unienv_interface.remote.compression import JpegCodec
from unienv_interface.remote.transport import MemoryTransport
from unienv_interface.remote.wire import WireError
from unienv_interface.space import BoxSpace

from tests_common.remote_envs import ProbeEnv, ScaleTransformation

ACTION = np.zeros(1, np.float32)


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.005)


def raw_frame(document, buffers=()):
    """Frame parts of an arbitrary (possibly invalid) request document."""
    body = json.dumps(document).encode()
    header = struct.pack("<II", len(buffers), len(body)) + b"".join(struct.pack("<Q", len(b)) for b in buffers)
    parts = [header, body, *buffers]
    return parts, sum(len(p) for p in parts)


def jpeg_blob(height, width):
    return JpegCodec().encode(np.zeros((1, height, width, 3), np.uint8), {"quality": 50, "layout": "HWC"})[0]


# ---------------------------------------------------------------------------- 1: compressed arrays in requests

def test_server_rejects_compressed_arrays_in_requests():
    """A client could send a tiny JPEG array that decodes to far more than the frame size limit."""
    hosted = ScaleTransformation()
    with RemoteServer() as server:
        server.register("t", hosted)
        client_side, server_side = MemoryTransport.pair()
        server._accept(server_side)
        blob = jpeg_blob(16, 16)
        array = {"__t": "nd", "d": "|u1", "s": [1, 16, 16, 3], "c": "jpeg", "o": {"layout": "HWC"}, "b": [0]}
        client_side.send(*raw_frame([1, "t", "transform", [None, array], {}], [blob]))
        with pytest.raises(ConnectionError):  # malformed request: the connection is dropped
            client_side.recv()
        assert hosted.received == []
        with server.connect() as client:  # other connections are unaffected
            assert client.transformation("t") is not None


def test_jpeg_decode_validates_the_shape_before_allocating():
    blob = jpeg_blob(16, 16)
    codec = JpegCodec()
    options = {"layout": "HWC"}
    with pytest.raises(WireError, match="number of images"):  # would be a ~200 TB allocation
        codec.decode([memoryview(blob)], np.dtype(np.uint8), (10 ** 6, 8192, 8192, 3), options)
    with pytest.raises(WireError, match="shape"):
        codec.decode([memoryview(blob)], np.dtype(np.uint8), (1, 70000, 16, 3), options)
    with pytest.raises(WireError, match="shape"):
        codec.decode([], np.dtype(np.uint8), (0, 0, 3), options)
    decoded = codec.decode([memoryview(blob)], np.dtype(np.uint8), (1, 16, 16, 3), options)
    assert decoded.shape == (1, 16, 16, 3)


# ---------------------------------------------------------------------------- 2: accept loop survives errors

class _FlakyListeningSocket:
    """A listening socket whose first ``accept`` fails like it does when the process is out of fds."""

    def __init__(self, sock):
        self.sock, self.failures = sock, 1

    def accept(self):
        if self.failures:
            self.failures -= 1
            raise OSError(24, "Too many open files")
        return self.sock.accept()

    def shutdown(self, how):
        self.sock.shutdown(how)

    def close(self):
        self.sock.close()


def test_accept_loop_survives_accept_and_setup_errors():
    import socket
    from unienv_interface.remote.transport import _SocketListener

    sock = socket.create_server(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    accepted, calls = [], []

    def on_connect(transport):
        calls.append(transport)
        if len(calls) == 1:
            raise RuntimeError("setting up the first connection fails")
        accepted.append(transport)

    listener = _SocketListener(_FlakyListeningSocket(sock), f"tcp://127.0.0.1:{port}", on_connect, 1 << 20)
    try:
        clients = [socket.create_connection(("127.0.0.1", port)) for _ in range(2)]
        wait_for(lambda: len(accepted) == 1)  # the accept error and the failed setup did not stop the loop
        assert listener._thread.is_alive()
        for client in clients:
            client.close()
        accepted[0].close()
    finally:
        listener.close()
    assert not listener._thread.is_alive()


# ---------------------------------------------------------------------------- 3: subscriptions vs release()

@pytest.mark.parametrize("release_after", [0, 60.0])
def test_release_ends_subscriptions_of_a_factory_env(release_after):
    with RemoteServer() as server:
        server.register_factory("sim", ProbeEnv, release_after=release_after)
        with server.connect() as client, server.connect() as other:
            events = client.subscribe("sim")
            client.env("sim").step(ACTION)
            assert events.next(timeout=5).kind == "step"
            client.release("sim")
            with pytest.raises(StopIteration):  # the subscription ended with the release
                events.next(timeout=5)
            events.close()  # used to raise RemoteError(PermissionError)
            served = server._connections[0]  # the server side of ``client``
            assert served.subscriptions == {} and served.factories == set()
            other.env("sim").step(ACTION)
            assert all(served not in r.subscribers for r in server._resources.values())

            again = client.subscribe("sim")  # reopens it; used to never receive anything
            client.env("sim").step(ACTION)
            assert again.next(timeout=5).kind == "step"
            again.close()


def test_unsubscribe_after_release_is_accepted_by_the_server():
    """An older client keeps its subscription after release() and unsubscribes later."""
    with RemoteServer() as server:
        server.register_factory("sim", ProbeEnv, release_after=60.0)
        with server.connect() as client:
            client.env("sim")
            client._request("", "subscribe", ["sim"], {})
            client._request("", "release", ["sim"], {})
            assert client._request("", "unsubscribe", ["sim"], {}) is None
            assert all(not c.subscriptions for c in server._connections)


# ---------------------------------------------------------------------------- 4 / 9: close() vs open / subscribe

class _CloseWhileAdding(set):
    """``record.users`` that closes the connection (on another thread) the moment ``_open`` adds it,
    i.e. right after ``_open`` checked that the connection was still open."""

    closer = None

    def add(self, connection):
        self.closer = threading.Thread(target=connection.close)
        self.closer.start()
        self.closer.join(0.3)  # without the fix close() completes here, before the connection is added
        super().add(connection)


def test_connection_closing_while_its_open_runs_releases_the_object():
    built = []

    def make():
        built.append(ProbeEnv())
        return built[-1]

    with RemoteServer() as server:
        server.register_factory("sim", make, release_after=0)
        record = server._factories["sim"]
        record.users = users = _CloseWhileAdding()
        client_side, server_side = MemoryTransport.pair()
        server._accept(server_side)
        client_side.send(*wire.encode([1, "", "open", ["sim"], {}]))
        wait_for(lambda: users.closer is not None)
        users.closer.join(5)
        wait_for(lambda: not record.users and built and built[0].closed_on is not None, timeout=2)
        assert record.resource is None


def test_subscribe_processed_after_the_connection_closed_is_ignored():
    """The reader may still process a buffered 'subscribe' after another thread closed the connection."""
    with RemoteServer() as server:
        server.register("sim", ProbeEnv())
        resource = server._resources["sim"]
        client = server.connect()
        served = server._connections[0]
        served.close()  # e.g. the worker failed to send it a reply
        with pytest.raises(ConnectionError):
            server._protocol_call(served, "subscribe", ["sim"], {})
        assert resource.subscribers == () and served.subscriptions == {}
        client.close()


# ---------------------------------------------------------------------------- 5: a peer that stops reading

def test_a_peer_that_stops_reading_does_not_stall_the_other_clients():
    from unienv_interface.remote.transport import connect as transport_connect
    from tests_common.remote_envs import LiveImageEnv

    with RemoteServer(send_timeout=0.5) as server:
        server.register("cam", LiveImageEnv(image_shape=(512, 512, 3)))
        address = server.listen("tcp://127.0.0.1:0")
        with RemoteClient.connect(address, timeout=10) as client:
            env = client.env("cam")
            env.reset(seed=0)
            stalled = transport_connect(address)  # sends ~30 MB of replies' worth of requests, reads nothing
            for request_id in range(1, 40):
                stalled.send(*wire.encode([request_id, "cam", "step", [np.zeros(3, np.float32)], {}]))
            wait_for(lambda: len(server._connections) == 2)
            started = time.monotonic()
            env.step(np.zeros(3, np.float32))  # used to wait until the stalled peer disconnected
            assert time.monotonic() - started < 8
            wait_for(lambda: len(server._connections) == 1)  # the stalled connection was closed
            stalled.close()


def test_send_timeout_validation():
    with pytest.raises(ValueError, match="send_timeout"):
        RemoteServer(send_timeout=0)
    with RemoteServer(send_timeout=None) as server:
        assert server.send_timeout is None


# ---------------------------------------------------------------------------- 6: dtypes the client backend lacks

class _Float64Env(ProbeEnv):
    """A numpy env with float64 observations and int64 info, as numpy simulators often have."""

    def __init__(self):
        super().__init__()
        self.observation_space = BoxSpace(B, -1.0, 1.0, np.float64, shape=(2,))

    def step(self, action):
        return np.full(2, 0.5), 1.0, False, False, {"count": np.int64(7), "half": np.float16(0.5)}


class _HugeIntEnv(ProbeEnv):
    """int64 bounds no 32-bit dtype can hold."""

    def __init__(self):
        super().__init__()
        self.observation_space = BoxSpace(B, 0, np.iinfo(np.int64).max, np.int64, shape=(1,))


def test_jax_client_receives_64_bit_data_and_skips_unrepresentable_resources():
    jax = pytest.importorskip("jax")
    if jax.config.jax_enable_x64:
        pytest.skip("needs JAX without x64")
    from unienv_interface.backends.jax import JaxComputeBackend as J

    with RemoteServer() as server:
        server.register("f32", ProbeEnv())
        server.register("f64", _Float64Env())
        server.register("huge", _HugeIntEnv())
        server.register_factory("huge_factory", _HugeIntEnv)
        with server.connect(backend=J) as client:  # used to fail: WireError: Unknown dtype string: float64
            env = client.env("f64")
            assert env.observation_space.dtype == J.__array_namespace_info__().dtypes()["float32"]
            observation, _, _, _, info = env.step(ACTION)
            assert str(observation.dtype) == "float32" and np.allclose(np.asarray(observation), 0.5)
            assert str(info["count"].dtype) == "int32" and int(info["count"]) == 7
            assert str(info["half"].dtype) == "float32"
            assert observation in env.observation_space
            with pytest.raises(TypeError, match="cannot be used with the jax backend"):
                client.env("huge")
            with pytest.raises(TypeError, match="cannot be used with the jax backend"):
                client.env("huge_factory")
            assert not server._factories["huge_factory"].users  # released again
            assert client.env("f32").step(ACTION)[1] == 1.0  # other resources are unaffected


def test_torch_client_receives_float16_and_unsigned_data():
    pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as T

    value = {"half": np.arange(3, dtype=np.float16), "u16": np.arange(3, dtype=np.uint16),
             "u32": np.array([2 ** 32 - 1], np.uint32)}
    restored = wire.loads(wire.dumps(value), T)
    assert str(restored["half"].dtype) == "torch.float32" and restored["half"].tolist() == [0.0, 1.0, 2.0]
    assert str(restored["u16"].dtype) == "torch.int32"
    assert str(restored["u32"].dtype) == "torch.int64" and restored["u32"].tolist() == [2 ** 32 - 1]


# ---------------------------------------------------------------------------- 7 / 16: compression of nested values

_LABELS = (np.arange(48 * 64 * 3).reshape(48, 64, 3) % 7).astype(np.uint8)  # a label map: must stay exact


def _image_space(**metadata):
    return BoxSpace(B, 0, 255, np.uint8, shape=(48, 64, 3), metadata=metadata)


class _UnionEnv(ProbeEnv):
    def __init__(self):
        super().__init__()
        from unienv_interface.space.spaces.union import UnionSpace
        self.observation_space = UnionSpace(B, [_image_space(kind="segmentation"), _image_space(kind="image")])
        self.index = 0

    def step(self, action):
        return (self.index, _LABELS.copy()), 0.0, False, False, {}


def test_compression_follows_the_union_alternative():
    hosted = _UnionEnv()
    with RemoteServer() as server:
        server.register("u", hosted)
        with server.connect(compress={"image": "jpeg:90"}) as client:
            index, labels = client.env("u").step(ACTION)[0]
            assert index == 0 and np.array_equal(labels, _LABELS)  # used to be JPEG-compressed (lossy)
            hosted.index = 1  # the image alternative is compressed
            index, image = client.env("u").step(ACTION)[0]
            assert index == 1 and image.shape == _LABELS.shape and not np.array_equal(image, _LABELS)


def test_compression_of_namedtuple_observations():
    from collections import namedtuple
    from unienv_interface.space import TupleSpace

    Observation = namedtuple("Observation", "image state")

    class NamedTupleEnv(ProbeEnv):
        def __init__(self):
            super().__init__()
            self.observation_space = TupleSpace(B, [_image_space(kind="image"),
                                                    BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,))])

        def step(self, action):
            return Observation(np.full((48, 64, 3), 100, np.uint8), np.ones(2, np.float32)), 0.0, False, False, {}

    with RemoteServer() as server:
        server.register("nt", NamedTupleEnv())
        with server.connect(compress={"image": "jpeg:90"}) as client:
            image, state = client.env("nt").step(ACTION)[0]  # used to fail: Observation.__new__() missing 'state'
            assert np.abs(image.astype(int) - 100).max() <= 2 and np.array_equal(state, np.ones(2, np.float32))


# ---------------------------------------------------------------------------- 8: BaseException in hosted code

class _ExitingEnv(ProbeEnv):
    def step(self, action):
        if isinstance(action, str) and action == "exit":
            raise SystemExit("the simulator gave up")
        return super().step(action)


def test_system_exit_in_a_hosted_call_is_reported_and_the_worker_survives():
    hosted = _ExitingEnv()
    with RemoteServer() as server:
        server.register("sim", hosted)
        with server.connect(timeout=5) as client:
            with pytest.raises(Exception, match="SystemExit: the simulator gave up"):  # used to hang
                client.call("sim", "step", "exit")
            assert client.call("sim", "step", ACTION)[1] == 1.0
    assert hosted.closed_on == "unienv-remote:sim"  # closed on its worker at shutdown


def test_system_exit_in_a_factory_is_reported():
    def make():
        raise SystemExit("cannot start")

    with RemoteServer() as server:
        server.register_factory("sim", make)
        with server.connect(timeout=5) as client:
            for _ in range(2):  # the worker is still there for the second attempt
                with pytest.raises(Exception, match="SystemExit: cannot start"):
                    client.env("sim")


# ---------------------------------------------------------------------------- 10 / 11: frame size limits

class _Camera(ProbeEnv):
    """Observations of one ``side`` x ``side`` RGB image (marked as an image, so it can be compressed)."""

    def __init__(self, side):
        super().__init__()
        self.observation_space = BoxSpace(B, 0, 255, np.uint8, shape=(side, side, 3), metadata={"kind": "image"})

    def step(self, action):
        super().step(action)
        return np.full(self.observation_space.shape, 100, np.uint8), 0.0, False, False, {}


def test_an_event_too_large_for_one_subscriber_still_reaches_the_others():
    with RemoteServer(max_frame_size=200_000) as server:
        server.register("cam", _Camera(260))  # a raw observation (~203 KB) exceeds the limit, a JPEG one not
        with server.connect() as raw, server.connect(compress={"image": "jpeg:90"}) as jpeg:
            raw_events = raw.subscribe("cam")  # subscribed first
            jpeg_events = jpeg.subscribe("cam")
            jpeg.env("cam").step(ACTION)
            assert jpeg_events.next(timeout=5).kind == "step"  # used to get nothing
            with pytest.raises(TimeoutError):
                raw_events.next(timeout=0.2)


def test_replies_respect_the_client_frame_size_limit():
    with RemoteServer() as server:
        server.register("cam", _Camera(1024))  # ~3 MB observations
        server.register("sim", ProbeEnv())
        address = server.listen("tcp://127.0.0.1:0")
        with RemoteClient.connect(address, max_frame_size=1 << 20, timeout=5) as small, \
                RemoteClient.connect(address, timeout=5) as large:
            small_events, large_events = small.subscribe("cam"), large.subscribe("cam")
            with pytest.raises(Exception, match="exceeds the 1048576-byte limit"):  # an error for this call only
                small.env("cam").step(ACTION)
            assert large_events.next(timeout=5).kind == "step"
            with pytest.raises(TimeoutError):  # the event is too large for the small client: skipped
                small_events.next(timeout=0.2)
            assert not small.closed  # used to drop the whole connection ("malformed frame")
            assert small.env("sim").step(ACTION)[1] == 1.0


def test_clients_that_send_no_frame_limit_keep_the_server_limit():
    with RemoteServer(max_frame_size=1 << 22) as server:
        client_side, server_side = MemoryTransport.pair(1 << 22)
        server._accept(server_side)
        from unienv_interface.remote import PROTOCOL_VERSION
        client_side.send(*wire.encode([1, "", "hello", [], {"version": PROTOCOL_VERSION}]))
        client_side.recv()
        assert server._connections[0].max_frame_size == 1 << 22
        client_side.close()


# ---------------------------------------------------------------------------- 12: malformed frames

def _loads_document(document, buffers=()):
    parts, _ = raw_frame(document, buffers)
    return wire.loads(b"".join(parts))


@pytest.mark.parametrize("document,buffers", [
    ({"__t": "float", "v": [1]}, ()),
    ({"__t": "float", "v": {}}, ()),
    ({"__t": "nd", "d": "<f4", "s": [0, 2 ** 70], "b": 0}, (b"",)),
    ({"__t": "nd", "d": "<f4", "s": [0, 2 ** 62, 2 ** 62], "b": 0}, (b"",)),
    ({"__t": "nd", "d": "|u1", "s": [2 ** 40, 2 ** 40, 2 ** 40, 3], "c": "jpeg", "o": {"layout": "HWC"}, "b": [0]},
     (b"",)),
    ({"__t": "nd", "d": "|u1", "s": [0, 0, 3], "c": "jpeg", "o": {"layout": "HWC"}, "b": [0]}, (b"",)),
])
def test_malformed_values_raise_wire_errors(document, buffers):
    with pytest.raises(WireError):  # used to raise TypeError / ValueError
        _loads_document(document, buffers)


def test_malformed_requests_close_only_their_connection_cleanly(monkeypatch):
    from unienv_interface.remote.transport import connect as transport_connect

    crashes = []
    monkeypatch.setattr(threading, "excepthook", lambda args: crashes.append(args.exc_value))
    nd = {"__t": "nd", "d": "<f4", "s": [0, 2 ** 70], "b": 0}
    requests = [
        ([1, "", "open", [[["unhashable"]]], {}], ()),
        ([1, "missing", "step", [{"__t": "float", "v": [1]}], {}], ()),
        ([1, "", "open", [{"__t": "float", "v": [1]}], {}], ()),
        ([1, "", "release", [nd], {}], (b"",)),
    ]
    with RemoteServer() as server:
        server.register("sim", ProbeEnv())
        server.register_factory("factory", ProbeEnv)
        address = server.listen("tcp://127.0.0.1:0")
        for document, buffers in requests:
            transport = transport_connect(address)
            transport.send(*raw_frame(document, buffers))
            try:
                transport.recv()  # either an error reply or a closed connection
            except ConnectionError:
                pass
            transport.close()
        wait_for(lambda: len(server._connections) == 0)
        assert crashes == []  # reader threads used to die with unhandled TypeError / ValueError
        with RemoteClient.connect(address) as client:
            assert client.env("sim").step(ACTION)[1] == 1.0


# ---------------------------------------------------------------------------- 13: release vs a call in flight

def test_space_update_of_a_resource_released_meanwhile_is_ignored():
    entered, gate = threading.Event(), threading.Event()

    class SlowReset(ProbeEnv):
        def reset(self, **kwargs):
            entered.set()
            gate.wait(5)
            return super().reset(**kwargs)

    with RemoteServer() as server:
        server.register_factory("sim", SlowReset, release_after=0)
        with server.connect(timeout=10) as client:
            env = client.env("sim")
            outcome = {}

            def reset():
                try:
                    outcome["value"] = env.reset(size=5)  # changes the observation space: carries an update
                except Exception as exc:
                    outcome["value"] = exc

            thread = threading.Thread(target=reset)
            thread.start()
            assert entered.wait(5)
            client.release("sim")  # the descriptor goes back to the factory stub
            gate.set()
            thread.join(10)
            assert not isinstance(outcome["value"], Exception), outcome["value"]  # used to be KeyError('version')
            assert outcome["value"][1].shape == (5,)


# ---------------------------------------------------------------------------- 14: numpy strings

def test_numpy_strings_and_bytes_travel_as_str_and_bytes():
    labels = np.array(["left", "right"])
    value = {"label": labels[0], "raw": np.bytes_(b"\x00x"), "name": np.str_("cam")}
    restored = wire.loads(wire.dumps(value))  # used to raise TypeError: Cannot send arrays of dtype <U4
    assert restored == {"label": "left", "raw": b"\x00x", "name": "cam"}
    assert type(restored["label"]) is str and type(restored["raw"]) is bytes
    parts, _ = wire.encode(np.bytes_(b"ab"))  # zero-copy path
    assert wire.loads(b"".join(bytes(p) for p in parts)) == b"ab"
