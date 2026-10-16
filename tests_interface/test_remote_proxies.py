"""Client proxies reuse their spaces while the wire descriptors are unchanged."""

from __future__ import annotations

import collections

import numpy as np
import pytest

import unienv_interface.remote.proxies as proxies_module
from unienv_interface.backends import NumpyComputeBackend as B
from unienv_interface.remote import RemoteServer
from unienv_interface.space import BoxSpace, DictSpace
from tests_common.env_perf_remote import LiveImageEnv


@pytest.fixture
def counted(monkeypatch):
    calls = collections.Counter()
    for name in ("space_to_json", "json_to_space"):
        original = getattr(proxies_module, name)

        def wrapper(*args, _original=original, _name=name, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(proxies_module, name, wrapper)
    return calls


def _set_on_server(server, resource_id, obj, name, value):
    server._resources[resource_id].domain.executor.submit(setattr, obj, name, value).result()


def test_unchanged_descriptors_skip_space_serialization(counted):
    with RemoteServer() as server:
        server.register("env", LiveImageEnv())
        with server.connect() as client:
            remote = client.env("env")
            spaces = (remote.observation_space, remote.action_space, remote.context_space)
            remote.reset(seed=0)
            counted.clear()
            for _ in range(5):
                remote.step(np.zeros(3, np.float32))
            client.describe("env")
            assert counted == {}
            assert (remote.observation_space, remote.action_space, remote.context_space) == spaces
            assert all(a is b for a, b in zip(spaces, (remote.observation_space, remote.action_space,
                                                        remote.context_space)))


def test_changed_descriptor_rebuilds_space(counted):
    with RemoteServer() as server:
        hosted = LiveImageEnv()
        server.register("env", hosted)
        with server.connect() as client:
            remote = client.env("env")
            remote.reset(seed=0)
            before = remote.action_space
            _set_on_server(server, "env", hosted, "action_space", BoxSpace(B, -2.0, 2.0, np.float32, shape=(3,)))
            remote.step(np.zeros(3, np.float32))
            assert remote.action_space is not before
            np.testing.assert_array_equal(remote.action_space.high, 2.0)
            rebuilt = remote.action_space
            counted.clear()
            remote.step(np.zeros(3, np.float32))
            assert remote.action_space is rebuilt and counted == {}


def test_context_space_set_to_none_and_back():
    with RemoteServer() as server:
        hosted = LiveImageEnv()
        server.register("env", hosted)
        with server.connect() as client:
            remote = client.env("env")
            remote.reset(seed=0)
            context_space = hosted.context_space
            _set_on_server(server, "env", hosted, "context_space", None)
            client.describe("env")
            assert remote.context_space is None
            _set_on_server(server, "env", hosted, "context_space", context_space)
            client.describe("env")
            assert isinstance(remote.context_space, BoxSpace)
            assert remote.context_space.shape == (3,)


def test_reassigned_space_attribute_is_checked_against_the_wire(counted):
    with RemoteServer() as server:
        server.register("env", LiveImageEnv())
        with server.connect() as client:
            remote = client.env("env")
            remote.reset(seed=0)
            # A different object (not the one last checked) takes the full comparison
            # path and, as before, is replaced by the server's space when it differs.
            remote.observation_space = DictSpace(B, {"other": BoxSpace(B, 0.0, 1.0, np.float32, shape=(1,))})
            counted.clear()
            remote.step(np.zeros(3, np.float32))
            assert counted["space_to_json"] == 1 and counted["json_to_space"] == 1
            assert set(remote.observation_space.keys()) == {"state", "image"}
            # an equal replacement object is kept
            equal = proxies_module.json_to_space(client.descriptors["env"]["observation_space"], B, None)
            remote.observation_space = equal
            remote.step(np.zeros(3, np.float32))
            assert remote.observation_space is equal
