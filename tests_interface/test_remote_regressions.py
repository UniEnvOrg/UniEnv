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
