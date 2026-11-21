"""RemoteServer.register_factory: objects built when a client opens them, closed once none uses them."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.remote import PROTOCOL_VERSION, RemoteError, RemoteServer

from tests_common.remote_envs import TRANSPORTS, ProbeEnv, ScaleTransformation, connect

ACTION = np.zeros(1, np.float32)


@pytest.fixture(params=TRANSPORTS)
def transport(request):
    if request.param == "websocket":
        pytest.importorskip("websockets")
    return request.param


@pytest.fixture
def server():
    with RemoteServer() as server:
        yield server


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.005)


class Factory:
    """Builds ProbeEnvs (the n-th with an observation of n + 1 dimensions) and keeps them."""

    def __init__(self, grow=False):
        self.built = []
        self.grow = grow
        self.threads = []

    def __call__(self):
        self.threads.append(threading.current_thread().name)
        env = ProbeEnv(size=2 + len(self.built) if self.grow else 2)
        self.built.append(env)
        return env


def test_built_on_open_and_closed_when_the_last_client_leaves(server, transport):
    make = Factory()
    server.register_factory("sim", make, release_after=0)
    assert server.resources == ["sim"]
    first, second = connect(server, transport), connect(server, transport)
    assert first.resources["sim"] == {"kind": None, "factory": True} and not make.built  # listing builds nothing

    env = first.env("sim")
    assert len(make.built) == 1 and make.threads == ["unienv-remote:sim"]
    assert env.observation_space.shape == (2,)
    env.reset(seed=0)
    assert env.step(ACTION)[1] == 1.0
    assert second.env("sim").step(ACTION)[1] == 2.0  # the same object, shared
    assert len(make.built) == 1

    first.close()
    assert second.env("sim").step(ACTION)[1] == 3.0  # still used by the second client
    assert make.built[0].closed_on is None
    second.close()
    wait_for(lambda: make.built[0].closed_on is not None)
    assert make.built[0].closed_on == "unienv-remote:sim"  # closed on its worker

    third = connect(server, transport)
    assert third.env("sim").step(ACTION)[1] == 1.0  # a newly built object
    assert len(make.built) == 2
    third.close()


def test_proxy_close_releases_and_reopening_rebuilds_with_new_spaces(server):
    make = Factory(grow=True)
    server.register_factory("sim", make, release_after=0)
    client = server.connect()
    env = client.env("sim")
    assert env.observation_space.shape == (2,)
    env.close()
    wait_for(lambda: make.built[0].closed_on is not None)
    with pytest.raises(RemoteError, match="PermissionError"):
        env.step(ACTION)  # a released proxy stops working
    again = client.env("sim")
    assert again is not env and again.observation_space.shape == (3,)
    assert again.step(ACTION)[0].shape == (3,)
    with again:
        pass  # leaving the with-block closes the env, i.e. releases it
    wait_for(lambda: make.built[1].closed_on is not None)
    client.close()


def test_grace_period_keeps_the_object_for_a_quick_reconnect(server):
    make = Factory()
    server.register_factory("sim", make, release_after=0.3)
    client = server.connect()
    client.env("sim").step(ACTION)
    client.close()
    time.sleep(0.05)
    client = server.connect()
    assert client.env("sim").step(ACTION)[1] == 2.0 and len(make.built) == 1  # kept: reopened within the grace
    client.close()
    time.sleep(0.15)
    assert make.built[0].closed_on is None
    wait_for(lambda: make.built[0].closed_on is not None, timeout=2.0)


def test_never_released_objects_close_with_the_server():
    make = Factory()
    server = RemoteServer()
    server.register_factory("sim", make, release_after=None)
    server.register_factory("timed", Factory(), release_after=60)
    client = server.connect()
    client.env("sim").step(ACTION)
    client.env("timed").step(ACTION)
    client.close()
    time.sleep(0.05)
    assert make.built[0].closed_on is None
    server.close()  # also cancels the pending 60 s release
    assert make.built[0].closed_on == "unienv-remote:sim"


def test_calls_and_subscriptions_open_on_demand(server):
    make = Factory()
    server.register_factory("sim", make, release_after=0)
    client = server.connect()
    assert client.call("sim", "step", ACTION)[1] == 1.0  # call() opens it
    client.release("sim")
    wait_for(lambda: make.built[0].closed_on is not None)
    with client.subscribe("sim") as events:  # so does subscribe()
        other = server.connect()
        other.env("sim").step(ACTION)
        assert events.next(timeout=5).kind == "step"
        other.close()
    assert len(make.built) == 2
    client.close()


def test_unopened_and_unknown_factories_are_refused(server):
    server.register_factory("sim", Factory(), release_after=0)
    client = server.connect()
    with pytest.raises(RemoteError, match="PermissionError"):
        client._request("sim", "step", [ACTION], {})  # skipping the open
    with pytest.raises(RemoteError, match="PermissionError"):
        client._request("", "subscribe", ["sim"], {})
    with pytest.raises(RemoteError, match="KeyError"):
        client._request("", "open", ["nothing"], {})
    # A client without factory support (older versions) is not shown them
    hello = client._request("", "hello", [], {"version": PROTOCOL_VERSION})
    assert "sim" not in hello["resources"]
    client.close()


def test_factory_failures_reach_the_client_and_leave_nothing_open(server):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("simulator failed to start")
        return ProbeEnv()

    server.register_factory("sim", flaky, release_after=0)
    server.register_factory("junk", lambda: 42)
    client = server.connect()
    with pytest.raises(RemoteError, match="simulator failed to start"):
        client.env("sim")
    assert client.env("sim").step(ACTION)[1] == 1.0  # the next open tries again
    with pytest.raises(RemoteError, match="TypeError"):
        client.resource("junk")
    client.close()


def test_transformation_factories_take_a_backend(server):
    server.register_factory("scale", lambda: ScaleTransformation(2.0), backend=B)
    server.register_factory("env_with_backend", lambda: ProbeEnv(), backend=B)
    client = server.connect()
    scale = client.transformation("scale")
    np.testing.assert_allclose(scale.transform(_box(), np.ones(2, np.float32)), 2.0)
    with pytest.raises(RemoteError, match="only given for a DataTransformation"):
        client.env("env_with_backend")
    client.close()


def _box():
    from unienv_interface.space import BoxSpace
    return BoxSpace(B, -np.inf, np.inf, np.float32, shape=(2,))


def test_registration_validation(server):
    server.register("plain", ProbeEnv())
    server.register_factory("sim", Factory())
    with pytest.raises(ValueError, match="already registered"):
        server.register_factory("plain", Factory())
    with pytest.raises(ValueError, match="already registered"):
        server.register("sim", ProbeEnv())
    with pytest.raises(ValueError, match="release_after"):
        server.register_factory("bad", Factory(), release_after=-1)
    with pytest.raises(TypeError, match="callable"):
        server.register_factory("bad", ProbeEnv())
    assert server.resources == ["plain", "sim"]
