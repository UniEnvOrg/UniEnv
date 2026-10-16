"""The remote server's encode / freeze shortcuts must not change a single wire byte.

Each scenario runs twice over MemoryTransport with deterministic ids and
timestamps: once with the current server, once with a reference server that
re-freezes every result and snapshot and re-encodes every response (the
behaviour before ``_FrozenResult`` / ``_Session.encoded``). Every payload the
server enqueues, per session and in order, must be byte-identical.
"""

from __future__ import annotations

import itertools
import threading
import time
import types
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
import pytest

import unienv_interface.remote.client as client_module
import unienv_interface.remote.server as server_module
from unienv_interface.remote import Codec, RemoteServer, RemoteWorldEnv
from unienv_interface.remote.codec import PROTOCOL_VERSION
from unienv_interface.world import WorldEnv
from tests_common.env_perf_remote import LiveImageEnv
from tests_common.env_worlds import CounterNode, SimWorld


class _ReferenceServer(RemoteServer):
    """Freezes and encodes like the server did before the shortcuts."""

    def _capture_snapshot(self, resource, fields: Iterable[str], captured) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field in sorted(fields):
            key = (resource.id, field)
            if key not in captured:
                try:
                    captured[key] = (self._freeze(self._snapshot(resource, [field])), None)
                except Exception as exc:
                    captured[key] = ({}, exc)
            data, error = captured[key]
            if error is not None:
                raise error
            result.update(data)
        return result

    def _freeze_result(self, session, request, result):
        def encode() -> Any:
            frozen = self._freeze(result)
            session.codec.encode({"version": PROTOCOL_VERSION, "type": "response", "session_id": session.id,
                                  "request_id": request["request_id"], "resource_id": request.get("resource_id"),
                                  "operation_id": request.get("operation_id"), "result": frozen})
            return frozen

        try:
            return encode()
        except server_module.CodecError:
            descriptors = result if isinstance(result, list) else result["descriptors"]
            if not any("snapshot" in descriptor for descriptor in descriptors):
                raise
            for descriptor in descriptors:
                descriptor.pop("snapshot", None)
            return encode()


def _record(monkeypatch, server_cls, scenario: Callable[[RemoteServer], None]) -> Dict[str, List[bytes]]:
    counter = itertools.count()
    lock = threading.Lock()

    def uuid4():
        with lock:
            return types.SimpleNamespace(hex=f"{next(counter):032x}")

    fake_time = types.SimpleNamespace(**{name: getattr(time, name) for name in ("monotonic", "time", "sleep")})
    fake_time.time_ns = lambda: 1234567890
    sent: Dict[str, List[bytes]] = {}
    original_enqueue = server_module._Session.enqueue

    def enqueue(session, payload, subscription=None):
        sent.setdefault(session.id, []).append(bytes(payload))
        return original_enqueue(session, payload, subscription)

    with monkeypatch.context() as patch:
        patch.setattr(server_module, "uuid4", uuid4)
        patch.setattr(client_module, "uuid4", uuid4)
        patch.setattr(server_module, "time", fake_time)
        patch.setattr(server_module._Session, "enqueue", enqueue)
        if server_cls is _ReferenceServer:
            patch.setattr(server_module._Session, "remember_encoded", lambda *args: None)
        with server_cls() as server:
            scenario(server)
    return sent


def _responses_with_results(sent: Dict[str, List[bytes]]) -> List[Dict[str, Any]]:
    decoded = [Codec().decode(p) for payloads in sent.values() for p in payloads]
    return [m for m in decoded if m.get("type") == "response" and isinstance(m.get("result"), dict)
            and "descriptors" in m["result"]]


def _assert_identical_wire(monkeypatch, scenario, min_payloads=10):
    expected = _record(monkeypatch, _ReferenceServer, scenario)
    actual = _record(monkeypatch, RemoteServer, scenario)
    assert actual.keys() == expected.keys()
    for session_id in expected:
        assert len(actual[session_id]) == len(expected[session_id])
        for index, (a, e) in enumerate(zip(actual[session_id], expected[session_id])):
            assert a == e, f"session {session_id} payload {index} differs"
    assert sum(len(v) for v in expected.values()) >= min_payloads
    return expected


def _drain(subscription, count):
    for _ in range(count):
        subscription.next(timeout=2)


@pytest.mark.parametrize("batch_size", [None, 3])
def test_env_mode_wire_bytes_unchanged(monkeypatch, batch_size):
    def scenario(server):
        hosted = LiveImageEnv(batch_size)
        server.register("env", hosted)
        with server.connect() as controller, server.connect() as observer:
            remote = controller.env("env")
            subscription = observer.subscribe("env", ["observation", "info", "reward", "render"])
            remote.reset(seed=1)
            _drain(subscription, 1)
            action = np.full(remote.action_space.shape, 0.5, np.float32)
            for _ in range(4):
                remote.step(action)
                _drain(subscription, 1)
            if batch_size is not None:
                remote.reset(mask=np.array([True, False, True]), seed=3)
                _drain(subscription, 1)
            remote.render()
            remote.step(action)
            _drain(subscription, 1)
            subscription.close()

    sent = _assert_identical_wire(monkeypatch, scenario)
    # call results embed boundary snapshots of the same arrays
    assert any("snapshot" in d for m in _responses_with_results(sent) for d in m["result"]["descriptors"])


@pytest.mark.parametrize("component_mode", [False, True])
def test_world_wire_bytes_unchanged(monkeypatch, component_mode):
    def scenario(server):
        world = SimWorld(0.125)
        hosted = WorldEnv(world, [CounterNode(world, "a", with_signals=True), CounterNode(world, "b")])
        server.register("env", hosted)
        with server.connect() as controller, server.connect() as observer:
            descriptor = controller.descriptors["env"]
            if component_mode:
                remote = RemoteWorldEnv(controller.world(descriptor["world_id"]), controller.node(descriptor["node_id"]))
            else:
                remote = controller.env("env")
            subscription = observer.subscribe(descriptor["node_id"])
            remote.reset(seed=2)
            _drain(subscription, 1)
            action = {"a": np.ones(1, np.float32), "b": np.full(1, 0.5, np.float32)}
            for _ in range(3):
                remote.step(action)
                _drain(subscription, 1)
            subscription.close()

    _assert_identical_wire(monkeypatch, scenario)


def test_snapshot_dropped_for_size_limit_wire_bytes_unchanged(monkeypatch):
    # The controller's negotiated limit fits the step result but not the result
    # plus its embedded boundary snapshot, exercising the fallback re-encode.
    probe = LiveImageEnv(None, (64, 64, 3))
    probe.reset(seed=0)
    result_size = len(Codec().encode({"value": probe.step(np.zeros(3, np.float32))}))

    def scenario(server):
        server.register("env", LiveImageEnv(None, (64, 64, 3)))
        with server.connect(max_message_size=result_size + 4096) as controller:
            remote = controller.env("env")
            remote.reset(seed=0)
            for _ in range(3):
                remote.step(np.zeros(3, np.float32))

    sent = _assert_identical_wire(monkeypatch, scenario, min_payloads=5)
    steps = _responses_with_results(sent)
    assert steps and all("snapshot" not in d for m in steps for d in m["result"]["descriptors"])


def test_failure_after_result_was_encoded_sends_the_error(monkeypatch):
    # _freeze_result has already encoded the successful envelope when a later
    # step fails; that cached payload must not be sent in place of the error.
    from unienv_interface.remote import RemoteError

    with RemoteServer() as server:
        server.register("env", LiveImageEnv())
        with server.connect() as client:
            remote = client.env("env")
            remote.reset(seed=0)

            def publish(*args, **kwargs):
                raise RuntimeError("publish failed")

            monkeypatch.setattr(server, "_publish", publish)
            with pytest.raises(RemoteError) as error:
                remote.step(np.zeros(3, np.float32))
            assert error.value.code == "execution_error"
            assert "publish failed" in str(error.value)
