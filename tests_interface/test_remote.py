"""Remote execution contract, run over every transport (memory, TCP, Unix socket, WebSocket)."""

from __future__ import annotations

import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from unienv_interface.remote import (PROTOCOL_VERSION, RemoteClient, RemoteEnv, RemoteError, RemoteServer,
                                     RemoteWorld, RemoteWorldNode, wire)
from unienv_interface.remote.transport import connect as transport_connect
from unienv_interface.space import BoxSpace, Space
from unienv_interface.world import WorldEnv

from tests_common.envs import StubEnv
from tests_common.remote_envs import TRANSPORTS, LiveImageEnv, ProbeEnv, connect, make_world_env


@pytest.fixture(params=TRANSPORTS)
def transport(request):
    if request.param == "websocket":
        pytest.importorskip("websockets")
    return request.param


@pytest.fixture
def server():
    with RemoteServer() as server:
        yield server


def same_tree(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            same_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            same_tree(x, y)
    elif isinstance(a, (np.ndarray, np.generic)):
        np.testing.assert_array_equal(np.asarray(b), a)
        assert np.asarray(b).dtype == np.asarray(a).dtype
    else:
        assert type(a) is type(b) and a == b


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.005)


# ---------------------------------------------------------------------------- environments

@pytest.mark.parametrize("batch_size", [None, 3])
def test_env_matches_local_env(server, transport, batch_size):
    local, hosted = StubEnv(batch_size), StubEnv(batch_size)
    server.register("stub", hosted)
    with connect(server, transport) as client:
        env = client.env("stub")
        assert isinstance(env, RemoteEnv) and env.batch_size == batch_size
        assert (env.render_mode, env.render_fps, env.metadata) == ("rgb_array", 10, {"render_modes": ["rgb_array"]})
        for attr in ("observation_space", "action_space", "context_space"):
            assert getattr(env, attr) == getattr(local, attr)
        same_tree(local.reset(seed=3), env.reset(seed=3))
        action = np.full(local.action_space.shape, 0.5, np.float32)
        for _ in range(3):
            same_tree(local.step(action), env.step(action))
        same_tree(local.render(), env.render())
        if batch_size:
            mask = np.array([True, False, True])
            same_tree(local.reset(mask=mask, seed=5), env.reset(mask=mask, seed=5))
            np.testing.assert_array_equal(hosted.reset_calls[-1]["mask"], mask)
        env.close()  # proxies never close hosted objects
        assert hosted.closed == 0
        assert env.sample_action().shape == local.action_space.shape


def test_live_buffers_are_copied_before_they_change(server, transport):
    server.register("live", LiveImageEnv(image_shape=(160, 160, 3)))  # > 64 KiB: fragmented on WebSocket
    with connect(server, transport) as client, client.subscribe("live") as stream:
        env = client.env("live")
        _, first, _ = env.reset(seed=1)
        assert first["image"].flags.writeable
        second = env.step(np.ones(3, np.float32))[0]
        env.step(np.ones(3, np.float32))
        assert (first["image"] == 1).all() and (second["image"] == 2).all()
        np.testing.assert_array_equal(second["state"], [2, 2, 2])
        event = stream.next(timeout=5)
        while event.sequence < 3:
            event = stream.next(timeout=5)
        assert event.kind == "step" and (event.value[0]["image"] == 3).all()


def test_remote_errors_carry_type_and_traceback(server, transport):
    server.register("probe", ProbeEnv())
    with connect(server, transport) as client:
        env = client.env("probe")
        with pytest.raises(RemoteError) as info:
            env.step("fail")
        assert info.value.type == "ValueError" and info.value.message == "step failed on purpose"
        assert "in step" in info.value.remote_traceback and "Remote traceback" in str(info.value)
        with pytest.raises(RemoteError, match="TypeError: Cannot send"):
            env.step("object")  # the result cannot be encoded
        with pytest.raises(RemoteError, match="AttributeError"):
            client.call("probe", "close")  # not in the allowlist
        with pytest.raises(RemoteError, match="AttributeError"):
            client.call("probe", "__class__")
        with pytest.raises(RemoteError, match="KeyError"):
            client.call("missing", "step", 1)
        with pytest.raises(TypeError, match="Cannot send"):
            env.step(object())  # rejected before anything is sent
        assert env.step(np.zeros(1, np.float32))[1] == 1.0  # the connection is still usable


def test_descriptor_attributes_methods_and_lookup(server, transport):
    server.register("probe", ProbeEnv())
    server.register_recursive("world_env", make_world_env())
    with connect(server, transport) as client:
        assert set(client.resources) == {"probe", "world_env", "world_env/world", "world_env/node"}
        probe = client.resources["probe"]
        assert probe["kind"] == "env" and "reload" not in probe["methods"]
        assert probe["attributes"]["metadata"] == {"render_modes": []}  # unportable entries are dropped
        assert "reload" in client.resources["world_env"]["methods"]
        assert client.env("world_env").control_timestep == 0.02
        assert client.resource("probe") is client.env("probe")
        with pytest.raises(TypeError, match="is a world"):
            client.env("world_env/world")
        with pytest.raises(KeyError, match="no resource"):
            client.resource("nope")
        with pytest.raises(RemoteError, match="does not expose 'reload'"):
            client.env("probe").reload()
        assert repr(client.env("probe")) == "<RemoteEnv 'probe'>"


def test_backend_and_device_conversion(server, transport):
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as T
    hosted = StubEnv()
    server.register("stub", hosted)
    with connect(server, transport, backend=T, device="cpu") as client:
        env = client.env("stub")
        assert env.backend is T and env.observation_space.backend is T
        _, obs, _ = env.reset(seed=1)
        assert isinstance(obs["position"], torch.Tensor)
        obs, reward, *_ = env.step(torch.ones(2))
        assert isinstance(hosted.last_action, np.ndarray)  # converted to the hosted backend
        np.testing.assert_array_equal(hosted.last_action, [1, 1])
        assert isinstance(obs["velocity"], torch.Tensor) and isinstance(reward, float)
    received = []
    torch_env = ProbeEnv()
    torch_env.backend, torch_env.step = T, lambda action: received.append(action)
    server.register("torch_probe", torch_env)
    with connect(server, transport) as client:
        client.env("torch_probe").step(np.ones(1, np.float32))
        assert isinstance(received[0], torch.Tensor)  # arguments use the hosted backend
        with pytest.raises(RemoteError, match="cannot represent dtype uint16"):
            client.env("torch_probe").step(np.ones(1, np.uint16))
        client.env("torch_probe").step(np.ones(1, np.float32))  # still connected


# ---------------------------------------------------------------------------- components

def test_components_mode_matches_server_side_world_env(server, transport):
    local, hosted = make_world_env(), make_world_env()
    server.register_recursive("sim", hosted)
    with connect(server, transport) as client:
        world, node = client.world("sim/world"), client.node("sim/node")
        assert isinstance(world, RemoteWorld) and isinstance(node, RemoteWorldNode)
        assert node.world is world and world.world_timestep == 0.01 and node.control_timestep == 0.02
        assert node.reset_priorities == {0} and node.has_reward
        env = WorldEnv(world, node)
        same_tree(local.reset(seed=1), env.reset(seed=1))
        assert hosted.world.log == local.world.log == ["reload", "reset", "after_reload"]
        for _ in range(2):
            same_tree(local.step(np.ones(1, np.float32)), env.step(np.ones(1, np.float32)))
        same_tree(local.reset(), env.reset())
        assert hosted.world.log == local.world.log
        assert hosted.node.log == local.node.log  # one RPC per hook, in the same order
        env.close()
        assert "close" not in hosted.world.log
        same_tree(local.step(np.ones(1, np.float32)), client.env("sim").step(np.ones(1, np.float32)))


def test_node_trees_and_local_combination(server, transport):
    hosted = make_world_env(("arm", "camera"))
    server.register_recursive("sim", hosted)
    with connect(server, transport) as client:
        assert {"sim/node/arm", "sim/node/camera"} <= set(client.resources)
        root = client.node("sim/node")
        assert [n.remote_name for n in root.nodes] == ["sim/node/arm", "sim/node/camera"]
        assert root.get_node("arm") is client.node("sim/node/arm") and root.get_node(["arm", "x"]) is None
        assert client.env("sim").get_node("camera").name == "camera"
        assert len(root.get_nodes_by_type(RemoteWorldNode)) == 3
        assert client.node("sim/node/arm").nodes == []
        # A local combination of remote children (each hook is one remote call).
        local = make_world_env(("arm", "camera"))
        env = WorldEnv(client.world("sim/world"), root.nodes)
        same_tree(local.reset(), env.reset())
        action = {"arm": np.ones(1, np.float32), "camera": np.zeros(1, np.float32)}
        same_tree(local.step(action), env.step(action))


def test_register_links_only_hosted_objects(server, transport):
    hosted = make_world_env(("arm", "camera"))
    server.register("sim", hosted)
    server.register("root", hosted.node)
    server.register("arm", hosted.node.get_node("arm"))
    with connect(server, transport) as client:
        assert set(client.resources) == {"sim", "root", "arm"}
        root = client.node("root")
        assert root.world is None  # the world is not hosted
        assert [n.remote_name for n in root.nodes] == ["arm"]  # only hosted children are linked
        assert root.get_node("arm") is client.node("arm") and root.get_node("camera") is None
        assert client.env("sim").get_node("arm") is client.node("arm")  # via the hosted root node
    server.register("world", hosted.world)  # registered after that client connected
    with connect(server, transport) as client:
        assert client.node("root").world is client.world("world")


def test_registration_names_groups_and_validation():
    world_env = make_world_env()
    other = make_world_env()
    with RemoteServer() as server:
        assert server.register("a", world_env) == "a"
        assert server.resources == ["a"]  # only the object itself, not its world / nodes
        assert server.register("again", world_env) == "a"  # already hosted: keeps its name
        assert server.register("w", world_env.world) == "w"
        assert server._resources["w"].group is server._resources["a"].group  # shares its world's worker
        assert server.register_recursive("n", other.node) == "n" and "n/world" in server.resources
        assert server.register_recursive("w2", other.world) == "n/world"  # already hosted: keeps its name
        assert server.register("e", other) == "e"  # joins the worker of its world
        assert server._resources["e"].group is server._resources["n"].group
        assert server._resources["a"].group is not server._resources["e"].group
        with pytest.raises(ValueError, match="already registered"):
            server.register("a", StubEnv())
        with pytest.raises(ValueError, match="non-empty"):
            server.register("", StubEnv())
        with pytest.raises(TypeError, match="Expected"):
            server.register("x", object())
        with pytest.raises(TypeError, match="space"):
            env = StubEnv()
            custom = type("Custom", (Space,), {})
            custom.__abstractmethods__ = frozenset()
            env.observation_space = object.__new__(custom)
            server.register("custom", env)
        assert "x" not in server.resources and "custom" not in server.resources
    with pytest.raises(RuntimeError, match="closed"):
        server.register("late", StubEnv())
    with pytest.raises(RuntimeError, match="closed"):
        server.listen()


# ---------------------------------------------------------------------------- spaces

def test_spaces_refresh_after_reset_for_every_client(server, transport):
    hosted = ProbeEnv(size=2)
    server.register("probe", hosted)
    with connect(server, transport) as first, connect(server, transport) as second:
        a, b = first.env("probe"), second.env("probe")
        space = a.observation_space
        a.reset()
        assert a.observation_space is space  # unchanged spaces are not re-sent
        a.reset(size=4)
        assert a.observation_space.shape == (4,) and a.reset()[1].shape == (4,)
        assert b.observation_space.shape == (2,)
        b.step(np.zeros(1, np.float32))  # any call delivers pending space updates
        assert b.observation_space.shape == (4,)
        assert second.resources["probe"]["spaces"]["observation_space"].shape == (4,)
        hosted.observation_space = ProbeEnv._space(4)  # new object, same content
        a.reset()
        assert a.observation_space.shape == (4,) and server._resources["probe"].version == 1


def test_components_mode_sees_spaces_built_on_reload(server, transport):
    hosted = make_world_env()
    hosted.node.observation_space = None  # unknown until the scene is built
    original = hosted.node.after_reload

    def after_reload(*, priority=0, mask=None):
        hosted.node.observation_space = BoxSpace(hosted.backend, -np.inf, np.inf, np.float32, shape=(1,))
        original(priority=priority, mask=mask)

    hosted.node.after_reload = after_reload
    server.register_recursive("sim", hosted)
    with connect(server, transport) as client:
        env = WorldEnv(client.world("sim/world"), client.node("sim/node"))
        assert env.observation_space is None
        _, obs, _ = env.reset()
        assert env.observation_space.shape == (1,) and obs.shape == (1,)


# ---------------------------------------------------------------------------- concurrency

def test_calls_run_on_one_worker_per_root(server, transport):
    first, second = ProbeEnv(), ProbeEnv()
    server.register("first", first)
    server.register("second", second)
    with connect(server, transport) as a, connect(server, transport) as b:
        for client in (a, b, a):
            client.env("first").step(np.zeros(1, np.float32))
            client.env("second").step(np.zeros(1, np.float32))
    assert len(set(first.threads)) == 1 and len(set(second.threads)) == 1
    assert set(first.threads) != set(second.threads)


def test_concurrent_clients_and_callers(server, transport):
    hosted = ProbeEnv()
    server.register("probe", hosted)
    clients = [connect(server, transport) for _ in range(3)]
    try:
        def work(index):
            env = clients[index % 3].env("probe")
            return [int(env.step(np.zeros(1, np.float32))[1]) for _ in range(25)]

        with ThreadPoolExecutor(6) as pool:
            counts = [count for result in pool.map(work, range(6)) for count in result]
        assert sorted(counts) == list(range(1, 151)) and hosted.count == 150
    finally:
        for client in clients:
            client.close()


def test_slow_call_queues_others_and_timeouts(server, transport):
    hosted = ProbeEnv()
    server.register("probe", hosted)
    with connect(server, transport) as client, connect(server, transport, timeout=0.05) as impatient:
        with ThreadPoolExecutor(1) as pool:
            blocked = pool.submit(client.env("probe").step, "block")
            assert hosted.entered.wait(5)
            with pytest.raises(TimeoutError, match="may still run"):
                impatient.env("probe").step(np.zeros(1, np.float32))  # queued behind the blocked call
            hosted.release.set()
            assert blocked.result(5)[1] == 1.0
        wait_for(lambda: hosted.count == 2)  # the timed-out call still ran
        assert impatient.env("probe").step(np.zeros(1, np.float32))[1] == 3.0  # late reply was ignored


# ---------------------------------------------------------------------------- connections

def test_disconnect_fails_calls_in_flight(server, transport):
    hosted = ProbeEnv()
    server.register("probe", hosted)
    client = connect(server, transport)
    with ThreadPoolExecutor(1) as pool:
        blocked = pool.submit(client.env("probe").step, "block")
        assert hosted.entered.wait(5)
        for connection in list(server._connections):
            connection.close()  # the server drops the connection
        with pytest.raises(ConnectionError, match="may or may not"):
            blocked.result(5)
    hosted.release.set()
    with pytest.raises(ConnectionError):
        client.env("probe").step(np.zeros(1, np.float32))
    client.close()
    with connect(server, transport) as fresh:  # the server keeps serving
        fresh.env("probe").step(np.zeros(1, np.float32))
    with pytest.raises(ConnectionError, match="Client is closed"):
        fresh.call("probe", "step", np.zeros(1, np.float32))


def test_server_close_disconnects_clients_and_closes_owned_objects(transport):
    owned, borrowed = ProbeEnv(), ProbeEnv()
    server = RemoteServer()
    server.register("owned", owned)
    server.register("borrowed", borrowed, close=False)
    client = connect(server, transport)
    client.env("owned").step(np.zeros(1, np.float32))
    server.close()
    server.close()
    assert owned.closed_on == owned.threads[0]  # closed on its worker thread
    assert borrowed.closed_on is None
    wait_for(lambda: client.closed)
    with pytest.raises(ConnectionError):
        client.env("owned").step(np.zeros(1, np.float32))
    client.close()


def test_serve_blocks_until_close(tmp_path):
    server = RemoteServer()
    server.register("stub", StubEnv())
    thread = threading.Thread(target=server.serve, args=("tcp://127.0.0.1:0",))
    thread.start()
    wait_for(lambda: server._listeners)
    with RemoteClient.connect(server._listeners[0].address) as client:
        client.env("stub").reset()
    server.close()
    thread.join(5)
    assert not thread.is_alive()


def raw_connection(address):
    transport = transport_connect(address)

    def request(*message):
        transport.send(*wire.encode(list(message)))
        document, buffers = transport.recv()
        return wire.restore(wire.parse(document), buffers)

    return transport, request


def test_protocol_handshake_and_malformed_requests(server):
    server.register("stub", StubEnv())
    address = server.listen("tcp://127.0.0.1:0")
    transport, request = raw_connection(address)
    reply = request(1, "", "hello", [], {"version": PROTOCOL_VERSION})
    assert reply[:2] == [1, 0] and reply[2]["version"] == PROTOCOL_VERSION and "stub" in reply[2]["resources"]
    assert request(2, "", "hello", [], {"version": 1})[1:3][0] == 1  # unsupported version -> error
    assert request(3, "", "nope", [], {})[2][0] == "AttributeError"
    assert request(4, "", "subscribe", ["missing"], {})[2][0] == "KeyError"
    assert request(5, "stub", "reset", [], {})[1] == 0
    transport.send(*wire.encode([6, "stub", "step", "not-a-list", {}]))  # invalid request: dropped connection
    with pytest.raises(ConnectionError):
        transport.recv()
    transport.close()
    sock = socket.create_connection(("127.0.0.1", int(address.rsplit(":", 1)[1])))
    sock.sendall(struct.pack("<II", 0, 3) + b"{x}")  # malformed frame
    assert sock.recv(10) == b""
    sock.close()
    with RemoteClient.connect(address) as client:  # other connections are unaffected
        client.env("stub").reset()


def test_unix_socket_cleanup_and_conflicts(tmp_path):
    path = str(tmp_path / "s.sock")
    stale = socket.socket(socket.AF_UNIX)
    stale.bind(path)
    stale.close()  # leaves a stale socket file behind
    with RemoteServer() as server:
        server.register("stub", StubEnv())
        address = server.listen("unix://" + path)
        with RemoteServer() as other, pytest.raises(OSError, match="listening"):
            other.listen(address)
        with RemoteClient.connect(address) as client:
            client.env("stub").reset()
    import os
    assert not os.path.exists(path)


# ---------------------------------------------------------------------------- observers

def test_subscription_streams_results_of_every_client(server, transport):
    server.register("stub", StubEnv())
    with connect(server, transport) as controller, connect(server, transport) as observer:
        env = controller.env("stub")
        with observer.subscribe("stub") as stream:
            reset = env.reset(seed=2)
            event = stream.next(timeout=5)
            assert (event.kind, event.sequence) == ("reset", 1)
            same_tree(reset, event.value)
            step = env.step(np.ones(2, np.float32))
            event = stream.next(timeout=5)
            assert (event.kind, event.sequence) == ("step", 2)
            same_tree(step, event.value)
            observer.env("stub").step(np.ones(2, np.float32))  # the observer's own calls count too
            assert next(iter(stream)).sequence == 3
            with pytest.raises(TimeoutError):
                stream.next(timeout=0.05)
        with pytest.raises(StopIteration):
            stream.next()
        wait_for(lambda: server._resources["stub"].subscribers == ())
        with pytest.raises(RemoteError, match="KeyError"):
            observer.subscribe("stub/missing")


def test_subscription_latest_wins(server, transport):
    server.register("stub", StubEnv())
    with connect(server, transport) as client:
        env = client.env("stub")
        with client.subscribe("stub") as slow, client.subscribe("stub") as other:
            env.reset()
            for _ in range(20):
                env.step(np.zeros(2, np.float32))
            wait_for(lambda: slow._latest is not None and slow._latest[0][1] == 21)
            assert slow.next(timeout=1).sequence == 21  # intermediate results were skipped
            with pytest.raises(TimeoutError):
                slow.next(timeout=0.05)
            assert other.next(timeout=1).sequence == 21
        wait_for(lambda: server._resources["stub"].subscribers == ())


def test_stalled_subscriber_never_blocks_the_server(server):
    hosted = LiveImageEnv(image_shape=(256, 256, 3))
    server.register("live", hosted)
    address = server.listen("tcp://127.0.0.1:0")
    transport, request = raw_connection(address)
    request(1, "", "hello", [], {"version": PROTOCOL_VERSION})
    request(2, "", "subscribe", ["live"], {})  # ... and never read again
    with RemoteClient.connect(address, timeout=10) as client:
        env = client.env("live")
        for _ in range(100):  # ~20 MB of events the stalled subscriber cannot take
            env.step(np.zeros(3, np.float32))
    assert hosted.steps == 100
    transport.close()


def test_connection_loss_ends_subscriptions(server, transport):
    server.register("stub", StubEnv())
    client = connect(server, transport)
    stream = client.subscribe("stub")
    for connection in list(server._connections):
        connection.close()
    with pytest.raises(ConnectionError):
        stream.next(timeout=5)
    stream.close()
    client.close()
