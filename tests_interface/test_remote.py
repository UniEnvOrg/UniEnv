"""Behavioral conformance, run unchanged over memory and real WebSocket sockets."""

from __future__ import annotations

from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import struct
import subprocess
import sys
import time
from threading import Event, get_ident
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import numpy as np
import pytest

from unienv_interface.backends import NumpyComputeBackend as B
from unienv_interface.env_base.env import Env
from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.space.spaces import BatchedSpace
from unienv_interface.space.spaces.graph import GraphInstance
from unienv_interface.world import World, WorldEnv, WorldNode
from unienv_interface.world.nodes.combined_node import CombinedWorldNode
from unienv_interface.remote import (
    Codec, CodecError, MemoryTransport, RemoteClient, RemoteError,
    RemoteServer, RemoteWorldEnv, UncertainOutcomeError,
)

if TYPE_CHECKING:
    from unienv_interface.remote.protocol import Descriptor, StepResult
    from unienv_interface.remote.proxies import RemoteWorldNode
    from unienv_interface.remote.server import _Domain, _Session


def box(shape: Sequence[int]) -> BoxSpace:
    return BoxSpace(B, -np.inf, np.inf, np.float32, shape=shape)


def assert_tree(actual: Any, expected: Any) -> None:
    if isinstance(expected, GraphInstance):
        assert isinstance(actual, GraphInstance)
        assert_tree(vars(actual), vars(expected))
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        if expected.dtype == object:
            assert actual.shape == expected.shape
            for a, b in zip(actual.flat, expected.flat):
                assert_tree(a, b)
        else:
            np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_tree(a, b)
    else:
        assert actual == expected


class CounterEnv(Env):
    backend, device = B, None
    render_mode = "rgb_array"

    def __init__(self, batch_size: Optional[int] = None) -> None:
        self.batch_size = batch_size
        self.shape = (2,) if batch_size is None else (batch_size, 2)
        self.action_space = box(self.shape)
        self.observation_space = DictSpace(B, {"position": box(self.shape)})
        self.context_space = box(self.shape)
        self.value = np.zeros(self.shape, np.float32)
        self.closed_count = 0
        self.threads: Set[int] = set()
        self.block_started, self.block_release = Event(), Event()

    def reset(self, *, mask: Optional[np.ndarray] = None, seed: Optional[int] = None,
              **kwargs: Any) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Optional[int]]]:
        self.threads.add(get_ident())
        if mask is None:
            self.value[:] = seed or 0
            value = self.value
        else:
            self.value[mask] = seed or 0
            value = self.value[mask]
        return value.copy(), {"position": value}, {"seed": seed}

    def step(self, action: np.ndarray) -> StepResult:
        self.threads.add(get_ident())
        if np.all(action == 99):
            self.block_started.set()
            self.block_release.wait(3)
        self.value += action
        reward = float(self.value.sum()) if self.batch_size is None else self.value.sum(axis=-1)
        terminated = bool(np.all(self.value > 3)) if self.batch_size is None else np.all(self.value > 3, axis=-1)
        truncated = False if self.batch_size is None else np.zeros(self.batch_size, dtype=bool)
        return {"position": self.value}, reward, terminated, truncated, {"pair": (b"abc", [1, None])}

    def render(self) -> np.ndarray:
        self.threads.add(get_ident())
        return np.full((3, 4, 3), int(self.value.sum()), dtype=np.uint8)

    def close(self) -> None:
        self.threads.add(get_ident())
        self.closed_count += 1


class TickWorld(World):
    backend, device = B, None
    world_timestep = 0.125
    world_subtimestep = 0.125
    batch_size = None

    def __init__(self) -> None:
        self.tick = 0
        self.log: List[Tuple[object, ...]] = []
        self.threads: Set[int] = set()
        self.closed_count = 0
        self.width = 1

    def record(self, *entry: object) -> None:
        self.threads.add(get_ident())
        self.log.append(entry)

    def reset(self, *, seed: Optional[int] = None, mask: Optional[np.ndarray] = None,
              **kwargs: Any) -> None:
        self.record("world.reset")
        self.tick = seed or 0

    def reload(self, *, seed: Optional[int] = None, mask: Optional[np.ndarray] = None,
               **kwargs: Any) -> None:
        self.record("world.reload")
        self.width += 1
        self.tick = seed or 0

    def after_reset(self, **kwargs: Any) -> None:
        self.record("world.after_reset")

    def after_reload(self, **kwargs: Any) -> None:
        self.record("world.after_reload")

    def step(self) -> float:
        self.record("world.step")
        self.tick += 1
        return self.world_timestep

    def close(self) -> None:
        self.record("world.close")
        self.closed_count += 1


class TickNode(WorldNode):
    reset_priorities = {1, 0}
    reload_priorities = {1, 0}
    after_reset_priorities = {0}
    after_reload_priorities = {0}
    pre_environment_step_priorities = {2, 0}
    post_environment_step_priorities = {0, -1}
    control_timestep = 0.5
    update_timestep = 0.25
    has_reward = True
    has_termination_signal = True
    has_truncation_signal = True
    render_mode = "rgb_array"

    def __init__(self, world: TickWorld, name: str = "sensor") -> None:
        self.world, self.name = world, name
        self.action_space, self.observation_space, self.context_space = box((1,)), box((1,)), box((1,))
        self.closed_count = 0
        self.action = np.zeros(1, np.float32)

    def reset(self, *, priority: int = 0, **kwargs: Any) -> None:
        self.world.record(self.name, "reset", priority)

    def reload(self, *, priority: int = 0, **kwargs: Any) -> None:
        self.world.record(self.name, "reload", priority)

    def after_reload(self, *, priority: int = 0, mask: Optional[np.ndarray] = None) -> None:
        self.world.record(self.name, "after_reload", priority)
        self.observation_space = box((self.world.width,))

    def after_reset(self, *, priority: int = 0, mask: Optional[np.ndarray] = None) -> None:
        self.world.record(self.name, "after_reset", priority)

    def set_next_action(self, action: np.ndarray) -> None:
        self.world.record(self.name, "action")
        self.action = action

    def pre_environment_step(self, dt: float, *, priority: int = 0) -> None:
        self.world.record(self.name, "pre", dt, priority)

    def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
        self.world.record(self.name, "post", dt, priority)

    def get_observation(self) -> np.ndarray:
        return np.full(self.observation_space.shape, self.world.tick, np.float32)

    def get_context(self) -> np.ndarray:
        return np.array([7], np.float32)

    def get_info(self) -> Dict[str, int]:
        return {"tick": self.world.tick}

    def get_reward(self) -> float:
        return float(self.world.tick)

    def get_termination(self) -> bool:
        return self.world.tick > 7

    def get_truncation(self) -> bool:
        return self.world.tick > 12

    def render(self) -> np.ndarray:
        return np.full((2, 2, 3), self.world.tick, np.uint8)

    def close(self) -> None:
        self.closed_count += 1


class DeviceNode(WorldNode):
    backend, device = B, None
    reset_priorities = {0}
    reload_priorities = {0}
    after_reset_priorities = {0}
    after_reload_priorities = {0}
    post_environment_step_priorities = {0}
    has_reward = True
    has_termination_signal = True
    has_truncation_signal = True

    def __init__(self, name: str = "device", width: int = 1, nodes: Sequence[WorldNode] = ()) -> None:
        self.name, self.nodes = name, list(nodes)
        self.action_space, self.observation_space, self.context_space = box((width,)), box((width,)), box((width,))
        self.value = np.zeros(width, np.float32)
        self.action = np.zeros(width, np.float32)
        self.closed_count = 0
        self.threads: Set[int] = set()
        self.block_started, self.block_release = Event(), Event()

    def reset(self, *, priority: int = 0, seed: Optional[int] = None, **kwargs: Any) -> None:
        self.threads.add(get_ident())
        self.value[:] = seed or 0
        self.action[:] = 0
        for child in self.nodes:
            child.reset(priority=priority, seed=seed, **kwargs)

    def set_next_action(self, action: np.ndarray) -> None:
        self.threads.add(get_ident())
        assert isinstance(action, np.ndarray)
        if np.all(action == 99):
            self.block_started.set()
            assert self.block_release.wait(5)
        self.action = action

    def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
        self.threads.add(get_ident())
        self.value += self.action

    def get_observation(self) -> np.ndarray:
        self.threads.add(get_ident())
        return self.value

    def get_context(self) -> np.ndarray:
        self.threads.add(get_ident())
        return np.zeros_like(self.value)

    def get_reward(self) -> float:
        self.threads.add(get_ident())
        return float(self.value.sum())

    def get_termination(self) -> bool:
        self.threads.add(get_ident())
        return bool(np.all(self.value > 10))

    def get_truncation(self) -> bool:
        self.threads.add(get_ident())
        return False

    def get_info(self) -> Dict[str, Any]:
        self.threads.add(get_ident())
        return {"name": self.name}

    def close(self) -> None:
        self.threads.add(get_ident())
        self.closed_count += 1
        for child in self.nodes:
            child.close()


@pytest.fixture(params=["memory", "websocket"])
def hosting(request: pytest.FixtureRequest) -> Iterator[Tuple[RemoteServer, Callable[..., RemoteClient]]]:
    with ExitStack() as stack:
        server = stack.enter_context(RemoteServer())
        endpoint: Optional[str] = None

        def connect(**options: object) -> RemoteClient:
            nonlocal endpoint
            if request.param == "websocket":
                pytest.importorskip("websockets")
                endpoint = endpoint or server.listen()
                client = RemoteClient.connect(endpoint, **options)
            else:
                client = server.connect(**options)
            return stack.enter_context(client)

        yield server, connect


@pytest.mark.parametrize("batch_size", [None, 1, 3])
def test_env_equivalence_and_immutable_observations(hosting, batch_size):
    server, connect = hosting
    local, hosted = CounterEnv(batch_size), CounterEnv(batch_size)
    server.register("env", hosted)
    client = connect()
    remote = client.env("env")
    assert remote.batch_size == batch_size
    assert remote.action_space.shape == local.shape
    assert_tree(remote.reset(seed=2), local.reset(seed=2))
    for _ in range(4):
        result = remote.step(np.ones(local.shape, np.float32))
        assert_tree(result, local.step(np.ones(local.shape, np.float32)))
    preserved = result[0]["position"].copy()
    remote.step(np.ones(local.shape, np.float32))
    np.testing.assert_array_equal(result[0]["position"], preserved)
    local.step(np.ones(local.shape, np.float32))
    assert_tree(remote.render(), local.render())
    if batch_size is not None:
        mask = np.arange(batch_size) == 0
        assert_tree(remote.reset(mask=mask, seed=10), local.reset(mask=mask, seed=10))
        assert remote.observation_space["position"].shape == local.shape
    remote.close()
    assert hosted.closed_count == 0
    server.close()
    assert hosted.closed_count == 1
    assert len(hosted.threads) == 1


def make_composed() -> WorldEnv:
    world = TickWorld()
    nodes = CombinedWorldNode("root", [TickNode(world, "a"), TickNode(world, "b")], direct_return=False)
    return WorldEnv(world, nodes)


@pytest.mark.parametrize("component_mode", [False, True])
def test_composition_lifecycle_spaces_and_snapshots(hosting, component_mode):
    server, connect = hosting
    local, hosted = make_composed(), make_composed()
    server.register("env", hosted)
    controller, observer = connect(), connect()
    descriptor = controller.descriptors["env"]
    node_id = descriptor["node_id"]
    if component_mode:
        remote = RemoteWorldEnv(controller.world(descriptor["world_id"]), controller.node(node_id))
    else:
        remote = controller.env("env")
    subscription = observer.subscribe(node_id, ["observation", "info", "render"])
    assert_tree(remote.reset(seed=1), local.reset(seed=1))
    event = subscription.next(timeout=2)
    assert event["kind"] in {"reset", "reload"}
    assert remote.observation_space["a"].shape == (2,)
    action = {"a": np.ones(1, np.float32), "b": np.ones(1, np.float32)}
    for _ in range(3):
        assert_tree(remote.step(action), local.step(action))
        snapshot = subscription.next(timeout=2)
        assert snapshot["sequence"] == event["sequence"] + 1
        assert_tree(snapshot["data"]["observation"], local.node.get_observation())
        event = snapshot
    assert hosted.world.log == local.world.log
    assert len(hosted.world.threads) == 1
    assert_tree(remote.reset(seed=5), local.reset(seed=5))
    assert_tree(remote.reload(), local.reload())
    assert remote.observation_space["a"].shape == (3,)
    assert_tree(remote.render(), local.render())
    assert remote.get_node(["a"]).name == "a"
    assert remote.get_node(["missing"]) is None
    assert remote.get_node([]) is not None


def test_local_combination_of_remote_children(hosting):
    server, connect = hosting
    hosted = make_composed()
    server.register("env", hosted)
    client = connect()
    root = client.node(client.descriptors["env"]["node_id"])
    remote = RemoteWorldEnv(root.world, [root.get_node("a"), root.get_node("b")])
    remote.reset()
    assert remote.observation_space["a"].shape == (2,)
    result = remote.step({"a": np.ones(1, np.float32), "b": np.ones(1, np.float32)})
    np.testing.assert_array_equal(result[0]["a"], np.full(2, 4, np.float32))


def test_observer_latest_fields_and_control_ownership(hosting):
    server, connect = hosting
    server.register("env", CounterEnv())
    controller, observer = connect(), connect()
    remote = controller.env("env")
    subscription = observer.subscribe("env", ["observation"])
    remote.reset()
    first = subscription.next(timeout=2)
    for _ in range(20):
        remote.step(np.ones(2, np.float32))
    latest = subscription.next(timeout=2)
    while latest["sequence"] < 21:
        latest = subscription.next(timeout=2)
    assert latest["sequence"] > first["sequence"] + 1
    assert set(latest["data"]) == {"observation"}
    np.testing.assert_array_equal(first["data"]["observation"]["position"], np.zeros(2, np.float32))
    with pytest.raises(RemoteError) as error:
        observer.env("env").step(np.zeros(2, np.float32))
    assert error.value.code == "ownership_conflict"
    remote.close()
    observer.env("env").reset()
    subscription.close()
    with pytest.raises(StopIteration):
        next(subscription)


def test_component_boundary_abort_and_mode_conflict(hosting):
    server, connect = hosting
    hosted = make_composed()
    server.register("env", hosted)
    client = connect()
    descriptor = client.descriptors["env"]
    world = client.world(descriptor["world_id"])
    root = client.node(descriptor["node_id"])
    remote = RemoteWorldEnv(world, root)
    remote.reset()
    with pytest.raises(RemoteError) as error:
        client.env("env").step({})
    assert error.value.code == "ownership_conflict"
    with pytest.raises(RemoteError, match="explicit matching"):
        world.step()
    observer = connect()
    subscription = observer.subscribe(world.resource_id)
    with client.operation(world.resource_id):
        world.step()
        with pytest.raises(TimeoutError):
            subscription.next(timeout=0.02)
    assert subscription.next(timeout=2)["data"] == {"dt": 0.125}
    with pytest.raises(ValueError):
        with client.operation(world.resource_id):
            world.step()
            raise ValueError("interrupt")
    with pytest.raises(RemoteError) as error:
        remote.step({})
    assert error.value.code == "faulted"
    remote.reset()
    remote.step({"a": np.ones(1, np.float32), "b": np.ones(1, np.float32)})


def test_timeout_reports_uncertainty_without_retry(hosting):
    server, connect = hosting
    hosted = CounterEnv()
    server.register("env", hosted)
    client = connect()
    remote = client.env("env")
    remote.reset()
    client.timeout = 0.25
    try:
        with pytest.raises(UncertainOutcomeError) as error:
            remote.step(np.full(2, 99, np.float32))
        assert error.value.uncertain
        assert hosted.block_started.is_set()
        assert client.closed
    finally:
        hosted.block_release.set()


def test_disconnect_faults_component_operation(hosting):
    server, connect = hosting
    world = TickWorld()
    server.register("world", world)
    client = connect()
    client._request("begin", "world", kind="step", operation_id="interrupted", mutation=True)
    client.close()
    replacement = connect()
    # Disconnect cleanup is serialized behind any accepted work. Discovery is
    # used as a worker barrier once the server's reader has observed closure.
    for session in list(server._sessions):
        if session.id == client.session_id:
            session.reader.join(timeout=2)
    replacement.discover()
    with pytest.raises(RemoteError) as error:
        with replacement.operation("world"):
            pass
    assert error.value.code == "faulted"
    with replacement.operation("world", "reset"):
        replacement.world("world").reset()
    with replacement.operation("world"):
        replacement.world("world").step()


def test_errors_and_allowlist(hosting):
    server, connect = hosting
    server.register("env", CounterEnv())
    client = connect()
    with pytest.raises(RemoteError) as error:
        client.call("env", "__getattribute__", "value")
    assert error.value.code == "unsupported_operation"
    with pytest.raises(RemoteError) as error:
        client.subscribe("env", ["private"])
    assert error.value.code == "invalid_request"
    with pytest.raises(RemoteError) as error:
        client.describe("missing")
    assert error.value.code == "not_found"
    with pytest.raises(CodecError):
        client.env("env").step(object())
    # Local serialization failure doesn't send a mutation or break the session.
    client.env("env").reset()


@pytest.mark.parametrize("value", [
    np.arange(30, dtype=np.int16).reshape(5, 6)[:, ::2], np.array(4, dtype=np.uint64),
    np.array([complex(1, 2)], dtype=np.complex64), np.ones((0, 4), dtype=np.float32),
    np.array([1, 2], dtype=">i4"), {"t": "literal", "x": (b"123", [True, None, 4, 2.5])},
])
def test_codec_roundtrip(value):
    codec = Codec()
    assert_tree(codec.decode(codec.encode(value)), value)


def test_codec_nonfinite_and_rejections():
    codec = Codec()
    values = codec.decode(codec.encode([float("nan"), float("inf"), -float("inf")]))
    assert np.isnan(values[0]) and values[1] == np.inf and values[2] == -np.inf
    for value in (object(), {1: "not a string key"}, np.array([object()], dtype=object)):
        with pytest.raises(CodecError):
            codec.encode(value)
    with pytest.raises(CodecError):
        Codec(20).encode(np.ones(50))
    with pytest.raises(CodecError):
        codec.decode(b"\x00\x00\x00\xff{}")
    for node in (
        {"t": "array", "dtype": "<f4", "shape": [10**18], "offset": 0, "length": 4},
        {"t": "array", "dtype": "O", "shape": [1], "offset": 0, "length": 4},
        {"t": "bytes", "offset": -1, "length": 4},
        {"t": "unknown"},
    ):
        header = json.dumps(node).encode()
        with pytest.raises(CodecError):
            codec.decode(struct.pack("!I", len(header)) + header + b"1234")


def test_version_mismatch_and_invalid_message():
    with RemoteServer() as server:
        client, connection = MemoryTransport.pair()
        server.attach(connection)
        codec = Codec()
        client.send(codec.encode({"version": 99, "type": "request", "request_id": "r1", "op": "hello"}))
        assert codec.decode(client.recv())["error"]["code"] == "version_mismatch"
        client.send(codec.encode([]))
        assert codec.decode(client.recv())["error"]["code"] == "invalid_request"
        client.close()


def test_base_import_does_not_import_websockets():
    subprocess.run([sys.executable, "-c", "import sys; import unienv_interface.remote; assert 'websockets' not in sys.modules"], check=True)


def test_wrapper_shares_world_ownership_and_cleanup(hosting):
    server, connect = hosting
    base = make_composed()

    class TrackingWrapper(Wrapper):
        closed_count = 0

        def close(self):
            self.closed_count += 1
            super().close()

    wrapped = TrackingWrapper(base)
    server.register("base", base)
    server.register("wrapped", wrapped)
    first, second = connect(), connect()
    remote = first.env("wrapped")
    remote.reset()
    assert first.descriptors["wrapped"]["world_id"] == first.descriptors["base"]["world_id"]
    with pytest.raises(RemoteError) as error:
        second.env("base").reset()
    assert error.value.code == "ownership_conflict"
    first.close()
    server.close()
    assert base.world.closed_count == 1
    assert wrapped.closed_count == 1
    assert all(node.closed_count == 1 for node in base.node.nodes)


def test_mutated_unserializable_result_faults_domain(hosting):
    server, connect = hosting

    class InvalidInfoEnv(CounterEnv):
        def step(self, action):
            result = super().step(action)
            result[-1]["unsupported"] = object()
            return result

    hosted = InvalidInfoEnv()
    server.register("env", hosted)
    client = connect()
    env = client.env("env")
    env.reset()
    with pytest.raises(RemoteError) as error:
        env.step(np.ones(2, np.float32))
    assert error.value.code == "serialization_error"
    assert error.value.uncertain
    np.testing.assert_array_equal(hosted.value, np.ones(2, np.float32))
    with pytest.raises(RemoteError) as error:
        env.step(np.ones(2, np.float32))
    assert error.value.code == "faulted"
    env.reset()


def test_subscription_error_is_reliable_and_does_not_fail_control(hosting):
    server, connect = hosting

    class BrokenRenderEnv(CounterEnv):
        renders = 0

        def render(self):
            self.renders += 1
            return object()

    hosted = BrokenRenderEnv()
    server.register("env", hosted)
    controller, observer = connect(), connect()
    normal = observer.subscribe("env")
    controller.env("env").reset()
    normal.next(timeout=2)
    assert hosted.renders == 0
    broken = observer.subscribe("env", ["render"])
    result = controller.env("env").step(np.ones(2, np.float32))
    np.testing.assert_array_equal(result[0]["position"], np.ones(2, np.float32))
    with pytest.raises(RemoteError) as error:
        broken.next(timeout=2)
    assert error.value.code == "snapshot_error"
    controller.env("env").step(np.ones(2, np.float32))
    assert hosted.renders == 1


def test_independent_worlds_progress_on_separate_workers(hosting):
    server, connect = hosting
    blocked, independent = CounterEnv(), CounterEnv()
    server.register("blocked", blocked)
    server.register("independent", independent)
    first, second = connect(), connect()
    first.env("blocked").reset()
    second.env("independent").reset()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(first.env("blocked").step, np.full(2, 99, np.float32))
        try:
            assert blocked.block_started.wait(2)
            result = second.env("independent").step(np.ones(2, np.float32))
            assert not future.done()
            np.testing.assert_array_equal(result[0]["position"], np.ones(2, np.float32))
        finally:
            blocked.block_release.set()
        future.result(timeout=2)
    assert blocked.threads.isdisjoint(independent.threads)


@pytest.mark.parametrize("backend_name", ["pytorch", "jax"])
def test_client_backend_conversion(hosting, backend_name):
    if backend_name == "pytorch":
        torch = pytest.importorskip("torch")
        from unienv_interface.backends.pytorch import PyTorchComputeBackend as backend
        device = "cpu"
    else:
        jax = pytest.importorskip("jax")
        from unienv_interface.backends.jax import JaxComputeBackend as backend
        device = jax.devices("cpu")[0]
    server, connect = hosting
    server.register("env", CounterEnv(1))
    client = connect(backend=backend, device=device)
    remote = client.env("env")
    remote.reset(seed=1)
    action = backend.from_numpy(np.ones((1, 2), np.float32), device=device)
    result = remote.step(action)
    assert backend.is_backendarray(result[0]["position"])
    np.testing.assert_array_equal(backend.to_numpy(result[0]["position"]), np.full((1, 2), 2, np.float32))
    assert remote.action_space.backend is backend
    with client.subscribe("env", ["observation"]) as stream:
        remote.step(action)
        assert backend.is_backendarray(stream.next(timeout=2)["data"]["observation"]["position"])


def test_backend_conversion_rejects_precision_loss():
    from unienv_interface.remote.codec import map_arrays

    class ReducingBackend:
        simplified_name = "test"

        @staticmethod
        def __array_namespace_info__():
            return np.__array_namespace_info__()

        @staticmethod
        def from_numpy(value, **kwargs):
            return value.astype(np.float32)

    with pytest.raises(CodecError, match="changed dtype"):
        map_arrays(np.ones(2, np.float64), ReducingBackend)


def test_registration_failure_does_not_leave_partial_resources():
    with RemoteServer() as server:
        reserved = TickWorld()
        server.register("env/node", reserved)
        with pytest.raises(ValueError, match="already registered"):
            server.register("env", make_composed())
        with server.connect() as client:
            assert [d["id"] for d in client.discover()] == ["env/node"]


def test_message_limit_and_pending_send_deadline():
    codec = Codec(64)
    with pytest.raises(CodecError, match="size"):
        codec.decode(bytes(65))

    class BlockedTransport:
        def __init__(self):
            self.closed = Event()

        def send(self, payload):
            self.closed.wait(2)

        def recv(self):
            self.closed.wait(2)
            raise ConnectionError("closed")

        def close(self):
            self.closed.set()

    transport = BlockedTransport()
    with pytest.raises(TimeoutError):
        RemoteClient(transport, timeout=0.02)
    assert transport.closed.is_set()


def test_graph_and_structured_batch_values(hosting):
    from unienv_interface.space.spaces import GraphSpace
    graph_space = GraphSpace(B, box((2,)), max_nodes=4, max_edges=6)
    graph = GraphInstance(n_nodes=np.array(2, np.int32), nodes_features=np.ones((2, 2), np.float32))
    batch = np.empty((2,), dtype=object)
    batch[0], batch[1] = {"graph": graph}, {"graph": graph}

    class StructuredEnv(CounterEnv):
        def __init__(self):
            super().__init__()
            self.observation_space = BatchedSpace(DictSpace(B, {"graph": graph_space}), (2,))

        def reset(self, **kwargs):
            return None, batch, {}

    server, connect = hosting
    server.register("structured", StructuredEnv())
    controller, observer = connect(), connect()
    stream = observer.subscribe("structured", ["observation"])
    actual = controller.env("structured").reset()[1]
    assert_tree(actual, batch)
    assert_tree(stream.next(timeout=2)["data"]["observation"], batch)
    graph.nodes_features[:] = 99
    np.testing.assert_array_equal(actual[0]["graph"].nodes_features, np.ones((2, 2), np.float32))


def test_invalid_container_shapes_and_records():
    codec = Codec()
    for node in [
        {"t": "object_array", "shape": [10000000000], "v": []},
        {"t": "object_array", "shape": [-1], "v": []},
        {"t": "graph", "v": {"t": "dict", "v": [["class", "os.system"]]}},
    ]:
        header = json.dumps(node).encode()
        with pytest.raises(CodecError):
            codec.decode(struct.pack("!I", len(header)) + header)


def test_negotiated_limits_isolate_slow_or_small_observers(hosting):
    server, connect = hosting

    class LargeRenderEnv(CounterEnv):
        def render(self):
            return np.zeros((128, 128, 3), np.uint8)

    server.register("env", LargeRenderEnv())
    controller = connect()
    observer = connect(max_message_size=4096)
    assert observer.codec.max_message_size == 4096
    stream = observer.subscribe("env", ["render"])
    controller.env("env").reset()
    with pytest.raises(RemoteError) as error:
        stream.next(timeout=2)
    assert error.value.code == "snapshot_error"
    assert not controller.closed and not observer.closed
    controller.env("env").step(np.ones(2, np.float32))


def test_oversized_mutation_response_reports_uncertainty(hosting):
    server, connect = hosting

    class LargeInfoEnv(CounterEnv):
        def step(self, action):
            result = super().step(action)
            result[-1]["large"] = bytes(5000)
            return result

    server.register("env", LargeInfoEnv())
    client = connect(max_message_size=4096)
    client.env("env").reset()
    with pytest.raises(RemoteError) as error:
        client.env("env").step(np.ones(2, np.float32))
    assert error.value.code == "serialization_error"
    assert error.value.uncertain
    assert not client.closed
    client.env("env").reset()


def test_hello_capabilities_and_unknown_result_fields(hosting, monkeypatch):
    from unienv_interface.remote.server import SERVER_VERSION
    server, connect = hosting
    original = server._request
    results = []

    def request(session, message):
        response = original(session, message)
        if message["op"] == "hello":
            results.append(response["result"].copy())
            response["result"]["future_capability"] = {"enabled": True}
        return response

    monkeypatch.setattr(server, "_request", request)
    client = connect()
    assert results == [{"session_id": client.session_id, "max_message_size": client.codec.max_message_size,
                        "server": "unienv", "server_version": SERVER_VERSION, "protocol_versions": [1]}]
    assert isinstance(SERVER_VERSION, str) and SERVER_VERSION
    assert client.discover() == []


@pytest.mark.parametrize("left, right, equal", [
    (float("nan"), float("nan"), True),
    (np.float32("nan"), float("nan"), True),
    ({"low": [[float("nan"), -np.inf]], "shape": (1, 2)},
     {"shape": (1, 2), "low": [[float("nan"), -np.inf]]}, True),
    ((float("nan"), {"value": [1]}), (float("nan"), {"value": [2]}), False),
    ({"low": [float("nan")]}, {"high": [float("nan")]}, False),
    ([float("nan")], [float("nan"), 1], False),
    ([float("nan")], (float("nan"),), False),
    ({}, [], False),
    (float("nan"), 0, False),
    (float("nan"), None, False),
    (np.inf, -np.inf, False),
    (10**400, 10**400 + 1, False),
    (np.int32(2), 2.0, True),
    (np.float32(1), 1.0000000000000002, False),
    (np.array(1, dtype=np.float32), 1.0000000000000002, False),
    (np.int64(2**53 + 1), float(2**53), False),
    (np.array(2**53 + 1, dtype=np.int64), float(2**53), False),
    (np.float32(1), 1.0, True),
    (np.array(float("nan"), dtype=np.float32), float("nan"), True),
])
def test_nan_aware_descriptor_comparison(left, right, equal):
    from unienv_interface.remote.proxies import _descriptors_equal
    assert bool(_descriptors_equal(left, right)) is equal
    assert bool(_descriptors_equal(right, left)) is equal


@pytest.mark.parametrize("nested", [False, True])
def test_nan_space_identity_survives_wire_refresh(hosting, nested):
    from unienv_interface.space.spaces import DynamicBoxSpace
    server, connect = hosting
    hosted = CounterEnv()
    space = DynamicBoxSpace(B, -np.inf, np.inf, (1,), (4,), np.float32, fill_value=float("nan"))
    hosted.observation_space = DictSpace(B, {"position": space}) if nested else space
    server.register("env", hosted)
    client = connect()
    remote = client.env("env")
    observation = remote.observation_space
    dynamic = observation["position"] if nested else observation
    assert np.isnan(dynamic.fill_value)
    client.describe("env")
    assert remote.observation_space is observation
    remote.reset()
    assert remote.observation_space is observation
    assert (remote.observation_space["position"] if nested else remote.observation_space) is dynamic


def test_mixed_precision_fill_value_rebuilds_space(hosting):
    from unienv_interface.space.spaces import DynamicBoxSpace
    server, connect = hosting
    hosted = CounterEnv()
    hosted.observation_space = DynamicBoxSpace(
        B, -np.inf, np.inf, (1,), (4,), np.float64, fill_value=np.float32(1))
    server.register("env", hosted)
    client = connect()
    remote = client.env("env")
    observation = remote.observation_space
    assert observation.fill_value.item() == 1.0
    updated = 1.0000000000000002
    server._resources["env"].domain.executor.submit(
        setattr, hosted.observation_space, "fill_value", updated).result()
    descriptor = client.describe("env")
    assert descriptor["observation_space"]["fill_value"] == updated
    assert remote.observation_space is not observation
    assert remote.observation_space.fill_value == updated
    assert observation.fill_value.item() == 1.0


def test_proxy_space_identity_and_descriptor_changes(hosting):
    server, connect = hosting
    hosted = CounterEnv()
    hosted.context_space = None
    server.register("env", hosted)
    client = connect()
    remote = client.env("env")
    observation, action = remote.observation_space, remote.action_space
    assert remote.context_space is None
    remote.reset()
    remote.step(np.ones(2, np.float32))
    client.discover()
    client.describe("env")
    with client.subscribe("env") as stream:
        remote.reset()
        stream.next(timeout=2)
    assert remote.observation_space is observation
    assert remote.action_space is action

    def change_spaces():
        hosted.observation_space = box((3,))
        hosted.context_space = box((2,))

    server._resources["env"].domain.executor.submit(change_spaces).result()
    client.describe("env")
    assert remote.observation_space is not observation
    assert remote.observation_space.shape == (3,)
    assert remote.context_space.shape == (2,)
    # Compare the current space, not merely the previous incoming descriptor.
    remote.action_space = box((4,))
    client.describe("env")
    assert remote.action_space.shape == (2,)


@pytest.mark.parametrize("kind, proxy_name, local_methods", [
    ("env", "RemoteEnv", {"get_node"}),
    ("world", "RemoteWorld", set()),
    ("node", "RemoteWorldNode", {"get_node", "get_nodes_by_fn"}),
])
def test_method_table_matches_proxies_and_docs(kind, proxy_name, local_methods):
    from unienv_interface.remote import proxies
    from unienv_interface.remote.protocol import METHODS
    proxy = getattr(proxies, proxy_name)
    public_methods = {name for name, method in vars(proxy).items() if not name.startswith("_") and callable(method)}
    assert public_methods - local_methods == set(METHODS[kind])
    document = (Path(__file__).resolve().parents[1] / "docs/reference/remote-protocol.md").read_text()
    row = next(line for line in document.splitlines() if line.startswith(f"| `{kind}` |"))
    documented_methods = {part for index, part in enumerate(row.split("|")[2].split("`")) if index % 2}
    assert documented_methods == set(METHODS[kind])


@pytest.mark.parametrize("method, uncertain", [("step", False), ("render", True)])
def test_server_error_uncertainty_is_authoritative(hosting, monkeypatch, method, uncertain):
    server, connect = hosting
    server.register("env", CounterEnv())
    client = connect()
    original = server._execute

    def execute(session, resource, request):
        if request["op"] == "call":
            raise RemoteError("execution_error", "Server decision", uncertain=uncertain)
        return original(session, resource, request)

    monkeypatch.setattr(server, "_execute", execute)
    with pytest.raises(RemoteError) as error:
        client.call("env", method)
    assert error.value.uncertain is uncertain
    assert not client.closed


@pytest.mark.parametrize("close_kind", ["release", "proxy", "composer"])
def test_release_retains_subscriptions(hosting, close_kind):
    server, connect = hosting
    server.register("env", make_composed())
    controller, observer = connect(), connect()
    node_id = observer.descriptors["env"]["node_id"]
    stream = observer.subscribe(node_id, ["observation"])
    if close_kind == "release":
        observer.release("env")
    elif close_kind == "proxy":
        observer.env("env").close()
    else:
        node = observer.node(node_id)
        RemoteWorldEnv(node.world, node).close()
    controller.env("env").reset()
    assert stream.next(timeout=2)["data"]["observation"]
    controller_stream = controller.subscribe(node_id, ["observation"])
    controller.release("env")
    observer.env("env").reset()
    assert controller_stream.next(timeout=2)["data"]["observation"]


def test_release_can_close_only_domain_subscriptions(hosting):
    server, connect = hosting
    server.register("env", make_composed())
    server.register("other", CounterEnv())
    client = connect()
    node_id = client.descriptors["env"]["node_id"]
    streams = [client.subscribe("env"), client.subscribe(node_id)]
    other = client.subscribe("other")
    client.release(node_id, close_subscriptions=True)
    for stream in streams:
        with pytest.raises(StopIteration):
            stream.next(timeout=2)
    client.env("other").reset()
    assert other.next(timeout=2)["resource_id"] == "other"


def test_discover_fans_out_before_waiting(hosting, monkeypatch):
    server, connect = hosting
    blocked, independent = CounterEnv(), CounterEnv()
    server.register("blocked", blocked)
    server.register("independent", independent)
    first, second = connect(), connect()
    captured = Event()
    original = server._descriptors

    def descriptors(domain):
        result = original(domain)
        if domain is server._resources["independent"].domain:
            independent.threads.add(get_ident())
            captured.set()
        return result

    monkeypatch.setattr(server, "_descriptors", descriptors)
    with ThreadPoolExecutor(max_workers=2) as executor:
        step = executor.submit(first.env("blocked").step, np.full(2, 99, np.float32))
        try:
            assert blocked.block_started.wait(2)
            discovery = executor.submit(second.discover)
            assert captured.wait(2)
            assert not discovery.done()
        finally:
            blocked.block_release.set()
        step.result(timeout=2)
        assert {item["id"] for item in discovery.result(timeout=2)} == {"blocked", "independent"}
    second.env("independent").reset()
    assert len(independent.threads) == 1
    assert independent.threads.isdisjoint(blocked.threads)


def test_failed_registration_preserves_existing_domain_members():
    with RemoteServer() as server:
        hosted = make_composed()
        server.register("world", hosted.world)
        server.register("env/node", TickWorld())
        resources, objects, domains = server._resources.copy(), server._objects.copy(), server._domains.copy()
        world_owners = server._world_owners.copy()
        members = {key: list(domain.resources) for key, domain in domains.items()}
        with pytest.raises(ValueError, match="already registered"):
            server.register("env", hosted)
        assert server._resources == resources
        assert server._objects == objects
        assert server._domains == domains
        assert server._world_owners == world_owners
        assert all(domain.resources == members[key] for key, domain in domains.items())
        server.register("valid", hosted)
        assert server.register("alias", hosted) == "valid"
        with server.connect() as client:
            assert client.descriptors["valid"]["world_id"] == "world"


def connect_pending(server: RemoteServer) -> RemoteClient:
    class NoDiscoveryClient(RemoteClient):
        def discover(self) -> List[Descriptor]:
            return []

    transport, connection = MemoryTransport.pair()
    server.attach(connection)
    return NoDiscoveryClient(transport)


def wait_until(predicate: Callable[[], bool], timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "Condition did not become true before deadline"
        time.sleep(0.01)


def assert_during(predicate: Callable[[], bool], duration: float = 0.6) -> None:
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        assert predicate()
        time.sleep(0.01)


def is_pending(domain: _Domain) -> bool:
    return domain.executor.submit(lambda: not domain.instantiated).result(timeout=2)


@pytest.mark.parametrize("operation", ["describe", "call", "begin", "subscribe"])
def test_factory_first_access_on_worker(operation):
    created, threads = [], set()

    class WorkerNode(TickNode):
        @property
        def nodes(self) -> Tuple[WorldNode, ...]:
            threads.add(get_ident())
            return ()

    def factory() -> WorldEnv:
        threads.add(get_ident())
        world = TickWorld()
        env = WorldEnv(world, WorkerNode(world))
        created.append(env)
        return env

    with RemoteServer(idle_timeout=None) as server:
        assert server.register("env", factory) == "env"
        domain = server._resources["env"].domain
        assert not created
        with connect_pending(server) as client:
            assert not created  # hello alone doesn't instantiate.
            if operation == "describe":
                client.describe("env/node")
            elif operation == "call":
                client._request("call", "env", method="reset", operation_id="first", args=[], kwargs={})
            elif operation == "begin":
                client._request("begin", "env/world", kind="reset", operation_id="first")
            else:
                client.subscribe("env/node", ["observation"])
            assert len(created) == 1
            client.describe("env")
            assert len(created) == 1
            assert client.descriptors["env"]["world_id"] == "env/world"
            assert client.descriptors["env"]["node_id"] == "env/node"
        worker = domain.executor.submit(get_ident).result()
    assert threads == {worker}
    assert created[0].world.threads == {worker}
    assert get_ident() != worker
    assert created[0].world.closed_count == 1


def test_factory_environment_lifecycle(hosting):
    server, connect = hosting
    created, threads = [], set()

    def factory() -> CounterEnv:
        threads.add(get_ident())
        created.append(CounterEnv())
        return created[-1]

    server.register("env", factory)
    assert not created
    client = connect()  # Discovery instantiates on both transports.
    assert len(created) == 1
    remote, local = client.env("env"), CounterEnv()
    assert_tree(remote.reset(seed=2), local.reset(seed=2))
    assert_tree(remote.step(np.ones(2, np.float32)), local.step(np.ones(2, np.float32)))
    remote.close()
    assert created[0].closed_count == 0
    server.close()
    assert created[0].closed_count == 1
    assert created[0].threads == threads


@pytest.mark.parametrize("transport", ["memory", "websocket"])
def test_factory_idle_eviction_and_revival(transport):
    created = []

    def factory() -> WorldEnv:
        env = make_composed()
        env.node.nodes[0].name = "a/b"
        env.node.nodes[0].observation_space = box((len(created) + 1,))
        created.append(env)
        return env

    with RemoteServer(idle_timeout=0.2) as server:
        server.register("env", factory)
        if transport == "websocket":
            pytest.importorskip("websockets")
            client = RemoteClient.connect(server.listen())
        else:
            client = server.connect()
        with client:
            node_id = "env/node/a%2Fb"
            remote = client.node(node_id)
            old_space = remote.observation_space
            old_revision = client.descriptors[node_id]["revision"]
            ids = set(client.descriptors)
            domain = server._resources["env"].domain
            assert server._world_owners == {id(created[0].world): domain}
            wait_until(lambda: is_pending(domain))
            assert server._world_owners == {}
            assert created[0].world.closed_count == 1
            assert all(node.closed_count == 1 for node in created[0].node.nodes)
            assert set(server._resources) == ids
            result = remote.get_observation()
            assert result.shape == (2,)
            assert len(created) == 2
            assert server._world_owners == {id(created[1].world): domain}
            assert set(client.descriptors) == ids
            assert client.descriptors[node_id]["revision"] == old_revision + 1
            assert remote.observation_space is not old_space
            assert remote.observation_space.shape == (2,)
            client.release(node_id)
            wait_until(lambda: is_pending(domain))
            assert created[1].world.closed_count == 1
        reaper = server._reaper
    assert not reaper.is_alive()
    assert len(created) == 2
    assert all(env.world.closed_count == 1 for env in created)
    assert server._world_owners == {}


@pytest.mark.parametrize("guard", ["subscription", "claim", "activity", "eager", "disabled"])
def test_idle_eviction_guards(guard):
    hosted = CounterEnv()
    with RemoteServer(idle_timeout=None if guard == "disabled" else 0.2) as server:
        server.register("env", hosted if guard == "eager" else lambda: hosted)
        with server.connect() as client:
            domain = server._resources["env"].domain
            if guard == "subscription":
                stream = client.subscribe("env", ["observation"])
            elif guard == "claim":
                client.acquire("env")

            def still_alive() -> bool:
                if guard == "activity":
                    client.describe("env")
                return hosted.closed_count == 0

            assert_during(still_alive)
            if guard == "subscription":
                stream.close()
            elif guard == "claim":
                client.release("env")
            if guard not in {"eager", "disabled"}:
                wait_until(lambda: is_pending(domain))
                assert hosted.closed_count == 1
    assert hosted.closed_count == 1


def test_close_pending_factory_and_validate_id():
    calls = []
    with RemoteServer(idle_timeout=0.2) as server:
        factory = lambda: calls.append(True)
        for invalid in (None, "", 3):
            with pytest.raises(ValueError, match="nonempty string"):
                server.register(invalid, factory)
        server.register("env", factory)
        with pytest.raises(ValueError, match="already registered"):
            server.register("env", factory)
        reaper = server._reaper
        assert reaper.daemon
    assert calls == []
    assert not reaper.is_alive()
    assert not server._domains


@pytest.mark.parametrize("failure", ["raise", "type", "tree", "wrapped_tree"])
def test_factory_failure_stays_pending_and_retries(failure):
    calls, created, wrappers = [], [], []

    class ClosingWrapper(Wrapper):
        closed_count = 0

        def close(self) -> None:
            self.closed_count += 1
            super().close()

    def factory() -> object:
        calls.append(get_ident())
        if len(calls) == 1:
            if failure == "raise":
                raise RuntimeError("Construction failed")
            if failure == "type":
                return object()
            env = make_composed()
            created.append(env)
            env.node.nodes[1].name = env.node.nodes[0].name
            if failure == "wrapped_tree":
                wrappers.append(ClosingWrapper(env))
                wrappers.append(ClosingWrapper(wrappers[-1]))
                return wrappers[-1]
            return env
        created.append(make_composed())
        return created[-1]

    with RemoteServer() as server:
        server.register("env", factory)
        domain = server._resources["env"].domain
        with connect_pending(server) as client:
            with pytest.raises(RemoteError) as error:
                client.describe("env")
            assert error.value.code == "execution_error"
            assert not error.value.uncertain
            assert is_pending(domain)
            assert set(server._resources) == {"env"}
            assert server._objects == {}
            assert server._world_owners == {}
            for env in created:
                assert env.world.closed_count == 1
                assert env.world.threads == {calls[0]}
                assert all(node.closed_count == 1 for node in env.node.nodes)
            assert all(wrapper.closed_count == 1 for wrapper in wrappers)
            client.describe("env/node/a")
            assert len(calls) == 2
            client.env("env").reset()
    assert all(env.world.closed_count == 1 for env in created)
    assert all(wrapper.closed_count == 1 for wrapper in wrappers)


def test_concurrent_factory_first_access():
    entered, release = Event(), Event()
    calls = []

    def factory() -> CounterEnv:
        calls.append(True)
        entered.set()
        assert release.wait(3)
        return CounterEnv()

    with RemoteServer(idle_timeout=None) as server:
        server.register("env", factory)
        with connect_pending(server) as first, connect_pending(server) as second:
            with ThreadPoolExecutor(max_workers=2) as executor:
                one = executor.submit(first.describe, "env")
                try:
                    assert entered.wait(2)
                    two = executor.submit(second.describe, "env")
                finally:
                    release.set()
                assert one.result(timeout=2) == two.result(timeout=2)
            assert calls == [True]


def test_access_waits_for_idle_disposal():
    closing, release, accessing = Event(), Event(), Event()
    created = []

    class SlowCloseEnv(CounterEnv):
        def close(self) -> None:
            closing.set()
            assert release.wait(3)
            super().close()

    def factory() -> SlowCloseEnv:
        if created:
            assert created[-1].closed_count == 1
        created.append(SlowCloseEnv())
        return created[-1]

    with RemoteServer(idle_timeout=0.2) as server:
        server.register("env", factory)
        with server.connect() as client:
            with ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    assert closing.wait(3)

                    def describe() -> Descriptor:
                        accessing.set()
                        return client.describe("env")

                    future = executor.submit(describe)
                    assert accessing.wait(2)
                    assert not future.done()
                    assert len(created) == 1
                finally:
                    release.set()
                assert future.result(timeout=2)["revision"] == 1
                assert len(created) == 2
    assert all(env.closed_count == 1 for env in created)


def test_factory_changed_topology_rejected_before_commit():
    created = []

    def factory() -> WorldEnv:
        env = make_composed()
        if len(created) == 1:
            env.node.nodes[0].name = "changed"
        created.append(env)
        return env

    with RemoteServer(idle_timeout=0.2) as server:
        server.register("env", factory)
        with server.connect() as client:
            domain = server._resources["env"].domain
            resources = server._resources.copy()
            wait_until(lambda: is_pending(domain))
            with pytest.raises(RemoteError, match="topology"):
                client.describe("env/node/a")
            assert is_pending(domain)
            assert server._resources == resources
            assert server._world_owners == {}
            assert domain.revision == 0
            assert [env.world.closed_count for env in created] == [1, 1]
            assert all(node.closed_count == 1 for node in created[1].node.nodes)
            worker = domain.executor.submit(get_ident).result()
            assert created[1].world.threads == {worker}
            assert client.describe("env/node/a")["revision"] == 1
            assert len(created) == 3
            assert server._world_owners == {id(created[2].world): domain}
    assert [env.world.closed_count for env in created] == [1, 1, 1]


@pytest.mark.parametrize("first_lazy", [False, True], ids=["eager-lazy", "lazy-lazy"])
def test_factory_rejects_shared_wrapped_environment(first_lazy):
    base = CounterEnv()
    first = Wrapper(Wrapper(base))
    rejected = []

    class ClosingWrapper(Wrapper):
        closed_count = 0

        def close(self) -> None:
            self.closed_count += 1
            super().close()

    def factory() -> Wrapper:
        rejected.append(ClosingWrapper(base))
        return rejected[-1]

    with RemoteServer(idle_timeout=None) as server:
        server.register("first", (lambda: first) if first_lazy else first)
        server.register("second", factory)
        with connect_pending(server) as controller, connect_pending(server) as other:
            controller.describe("first")
            controller.env("first").reset()
            owner = server._resources["first"].domain
            pending = server._resources["second"].domain
            resources, objects = server._resources.copy(), server._objects.copy()
            assert id(base) not in objects
            assert server._world_owners == {id(base): owner}
            for _ in range(2):
                with pytest.raises(RemoteError, match="another domain") as error:
                    other.describe("second")
                assert error.value.code == "execution_error"
                assert not error.value.uncertain
                assert is_pending(pending)
                assert pending.owner is None
                assert server._resources == resources
                assert server._objects == objects
                assert server._world_owners == {id(base): owner}
                assert base.closed_count == 0
            controller.env("first").step(np.ones(2, np.float32))
            worker = owner.executor.submit(get_ident).result()
            assert base.threads == {worker}
    assert base.closed_count == 1
    assert base.threads == {worker}
    assert all(wrapper.closed_count == 0 for wrapper in rejected)
    assert server._world_owners == {}


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_worldenv_tree_retains_single_underlying_world_owner(lazy):
    env = make_composed()
    with RemoteServer(idle_timeout=None) as server:
        if not lazy:
            server.register("world", env.world)
        server.register("env", (lambda: Wrapper(env)) if lazy else Wrapper(env))
        if not lazy:
            server.register("base", env)
        with server.connect() as client:
            domain = server._resources["env"].domain
            assert len(server._domains) == 1
            assert server._world_owners == {id(env.world): domain}
            assert all(resource.domain is domain for resource in server._resources.values())
            assert client.descriptors["env"]["world_id"] == ("env/world" if lazy else "world")
            client.env("env").reset()
    assert env.world.closed_count == 1
    assert all(node.closed_count == 1 for node in env.node.nodes)
    assert server._world_owners == {}


@pytest.mark.parametrize("root_kind", ["env", "wrapper", "node"])
def test_rejected_partial_factory_tree_preserves_registered_objects(root_kind):
    hosted, created, threads = make_composed(), [], []

    def factory() -> object:
        threads.append(get_ident())
        env = make_composed()
        created.append(env)
        env.node.nodes.append(hosted.node.nodes[0])
        if root_kind == "node":
            return env.node
        return Wrapper(env) if root_kind == "wrapper" else env

    with RemoteServer(idle_timeout=None) as server:
        server.register("hosted", hosted)
        server.register("invalid", factory)
        with connect_pending(server) as client:
            client.describe("hosted")
            domain = server._resources["hosted"].domain
            for _ in range(2):
                with pytest.raises(RemoteError, match="server-side world"):
                    client.describe("invalid")
                rejected = created[-1]
                assert rejected.world.closed_count == 1
                assert rejected.world.threads == {threads[-1]}
                assert all(node.closed_count == 1 for node in rejected.node.nodes[:2])
                assert hosted.world.closed_count == 0
                assert all(node.closed_count == 0 for node in hosted.node.nodes)
                assert server._world_owners == {id(hosted.world): domain}
                assert is_pending(server._resources["invalid"].domain)
            client.env("hosted").reset()
    assert all(env.world.closed_count == 1 for env in [hosted, *created])
    assert all(node.closed_count == 1 for node in hosted.node.nodes)


@pytest.mark.parametrize("pause_at", ["validation", "cleanup", "close"])
def test_rejected_cleanup_reserves_world_against_concurrent_factory(monkeypatch, pause_at):
    invalid = make_composed()
    invalid.node.nodes[1].name = invalid.node.nodes[0].name
    shared, survivor = invalid.world, TickWorld()
    validation_started, finish_validation = Event(), Event()
    cleanup_started, finish_cleanup, competing_factory_started = Event(), Event(), Event()
    products = []

    def factory() -> World:
        products.append(shared if not products else survivor)
        competing_factory_started.set()
        return products[-1]

    with RemoteServer(idle_timeout=None) as server:
        server.register("invalid", lambda: invalid)
        server.register("survivor", factory)
        original_register = server._register
        original_cleanup = server._dispose_rejected
        original_close = shared.close

        def register(resource_id: str, obj: object, target: Optional[_Domain] = None) -> str:
            if resource_id == "invalid" and pause_at == "validation":
                validation_started.set()
                assert finish_validation.wait(5)
            return original_register(resource_id, obj, target=target)

        def cleanup(objects: List[object]) -> None:
            if pause_at != "close" and any(obj is invalid for obj in objects):
                cleanup_started.set()
                assert finish_cleanup.wait(5)
            original_cleanup(objects)

        def close() -> None:
            cleanup_started.set()
            assert finish_cleanup.wait(5)
            original_close()

        monkeypatch.setattr(server, "_register", register)
        monkeypatch.setattr(server, "_dispose_rejected", cleanup)
        if pause_at == "close":
            monkeypatch.setattr(shared, "close", close)

        with connect_pending(server) as first, connect_pending(server) as second:
            with ThreadPoolExecutor(max_workers=2) as executor:
                rejected = executor.submit(first.describe, "invalid")
                try:
                    if pause_at == "validation":
                        assert validation_started.wait(3)
                        competing = executor.submit(second._request, "acquire", "survivor", mode="components")
                        assert competing_factory_started.wait(3)
                        assert not competing.done()
                        finish_validation.set()
                    assert cleanup_started.wait(3)
                    with server._lock:
                        assert id(shared) in server._disposal_reservations
                    with pytest.raises(RemoteError, match="reserved for disposal") as error:
                        if pause_at == "validation":
                            competing.result(timeout=3)
                        else:
                            second._request("acquire", "survivor", mode="components")
                    assert error.value.code == "execution_error"
                    domain = server._resources["survivor"].domain
                    assert is_pending(domain)
                    assert domain.owner is None
                    assert shared.closed_count == 0
                    with server._lock:
                        # The competing rejection must not release the first cleanup's reservation.
                        assert id(shared) in server._disposal_reservations
                    second._request("acquire", "survivor", mode="components")
                    assert domain.instantiated and domain.owner.id == second.session_id
                    assert server._world_owners == {id(survivor): domain}
                    assert survivor.closed_count == 0
                finally:
                    finish_validation.set()
                    finish_cleanup.set()
                with pytest.raises(RemoteError, match="Duplicate child node name"):
                    rejected.result(timeout=3)
            assert shared.closed_count == 1
            assert survivor.closed_count == 0
            assert server._disposal_reservations == set()
            second.describe("survivor")
            with second.operation("survivor", "reset"):
                second.world("survivor").reset()
    assert shared.closed_count == survivor.closed_count == 1
    assert all(node.closed_count == 1 for node in invalid.node.nodes)
    assert shared.threads.isdisjoint(survivor.threads)


def test_rejected_cleanup_preserves_world_acquired_before_validation():
    invalid = make_composed()
    invalid.node.nodes[1].name = invalid.node.nodes[0].name
    shared = invalid.world
    factory_started, finish_factory = Event(), Event()

    def factory() -> WorldEnv:
        factory_started.set()
        assert finish_factory.wait(5)
        return invalid

    with RemoteServer(idle_timeout=None) as server:
        server.register("invalid", factory)
        server.register("survivor", lambda: shared)
        with connect_pending(server) as first, connect_pending(server) as second:
            with ThreadPoolExecutor(max_workers=1) as executor:
                rejected = executor.submit(first.describe, "invalid")
                try:
                    assert factory_started.wait(3)
                    second._request("acquire", "survivor", mode="components")
                finally:
                    finish_factory.set()
                with pytest.raises(RemoteError, match="another domain"):
                    rejected.result(timeout=3)
            domain = server._resources["survivor"].domain
            assert domain.instantiated and domain.owner.id == second.session_id
            assert server._world_owners == {id(shared): domain}
            assert server._disposal_reservations == set()
            assert shared.closed_count == 0
            assert all(node.closed_count == 1 for node in invalid.node.nodes)
            second.describe("survivor")
            with second.operation("survivor", "reset"):
                second.world("survivor").reset()
    assert shared.closed_count == 1
    assert len(shared.threads) == 1


def test_rejected_cleanup_releases_reservations_after_error(monkeypatch):
    invalid = make_composed()
    invalid.node.nodes[1].name = invalid.node.nodes[0].name
    survivor, products = make_composed(), []

    def factory() -> WorldEnv:
        products.append(invalid if not products else survivor)
        return products[-1]

    with RemoteServer(idle_timeout=None) as server:
        server.register("env", factory)
        original_cleanup = server._dispose_rejected

        def cleanup(objects: List[object]) -> None:
            with server._lock:
                assert id(invalid.world) in server._disposal_reservations
            original_cleanup(objects)
            raise RuntimeError("Cleanup hook failed")

        monkeypatch.setattr(server, "_dispose_rejected", cleanup)
        with connect_pending(server) as client:
            with pytest.raises(RemoteError, match="Duplicate child node name"):
                client.describe("env")
            assert invalid.world.closed_count == 1
            assert server._disposal_reservations == set()
            assert is_pending(server._resources["env"].domain)
            client.describe("env")
            client.env("env").reset()
    assert invalid.world.closed_count == survivor.world.closed_count == 1


def record_requests(server: RemoteServer, monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    original = server._request

    def request(session: _Session, message: Dict[str, Any]) -> Dict[str, Any]:
        requests.append(message.copy())
        return original(session, message)

    monkeypatch.setattr(server, "_request", request)
    return requests


def read_node_fields(node: RemoteWorldNode) -> Dict[str, Any]:
    return {"context": node.get_context(), "observation": node.get_observation(),
            "reward": node.get_reward(), "terminated": node.get_termination(),
            "truncated": node.get_truncation(), "info": node.get_info()}


@pytest.mark.parametrize("component_mode", [False, True])
def test_controller_boundary_getters_use_cache(hosting, monkeypatch, component_mode):
    server, connect = hosting
    hosted = make_composed()
    server.register("env", hosted)
    requests = record_requests(server, monkeypatch)
    client = connect()
    node = client.node("env/node")
    remote = RemoteWorldEnv(node.world, node) if component_mode else client.env("env")
    remote.reset()
    action = {"a": np.ones(1, np.float32), "b": np.ones(1, np.float32)}
    remote.step(action)
    nodes = [node, node.get_node("a"), node.get_node("b")]
    requests.clear()
    values = [read_node_fields(member) for member in nodes]
    assert requests == []
    for member in nodes:
        snapshot = client._snapshots[member.resource_id]
        assert snapshot["sequence"] == 2
        assert snapshot["revision"] == member.descriptor["revision"]
        assert "render" not in snapshot["data"]
        assert "snapshot" not in member.descriptor
    if not component_mode:
        assert set(client._snapshots["env"]["data"]) == {"observation", "reward", "terminated", "truncated", "info"}
    remote.render()
    assert len(requests) == 1 and requests[0]["method"] == "render"
    # Explicit client.call is a live RPC, including in the presence of a cache.
    if not component_mode:
        client.release("env")
    getters = {"context": "get_context", "observation": "get_observation", "reward": "get_reward",
               "terminated": "get_termination", "truncated": "get_truncation", "info": "get_info"}
    for member, cached in zip(nodes, values):
        live = {field: client.call(member.resource_id, method) for field, method in getters.items()}
        assert_tree(cached, live)


def test_controller_getters_are_live_inside_operations(hosting, monkeypatch):
    server, connect = hosting

    class ActionNode(TickNode):
        def get_observation(self) -> np.ndarray:
            return self.action.copy()

    world = TickWorld()
    hosted = ActionNode(world)
    server.register("node", hosted)
    requests = record_requests(server, monkeypatch)
    client = connect()
    node = client.node("node")
    remote = RemoteWorldEnv(node.world, node)
    remote.reset()
    np.testing.assert_array_equal(node.get_observation(), np.zeros(1))
    with client.operation(node.resource_id):
        assert client._snapshots == {}
        node.set_next_action(np.array([7], np.float32))
        requests.clear()
        values = read_node_fields(node)
        assert len(requests) == 6
        assert all(request["op"] == "call" for request in requests)
        np.testing.assert_array_equal(values["observation"], np.array([7], np.float32))
    requests.clear()
    np.testing.assert_array_equal(node.get_observation(), np.array([7], np.float32))
    assert requests == []


@pytest.mark.parametrize("component_mode", [False, True])
def test_controller_cache_reset_reload_and_fault_invalidation(hosting, monkeypatch, component_mode):
    server, connect = hosting
    hosted = make_composed()
    server.register("env", hosted)
    client = connect()
    node = client.node("env/node")
    remote = RemoteWorldEnv(node.world, node) if component_mode else client.env("env")
    remote.reset()
    action = {"a": np.ones(1, np.float32), "b": np.ones(1, np.float32)}
    remote.step(action)
    old = client._snapshots[node.resource_id]
    original = server._execute
    invalidated = []

    def execute(session: _Session, resource: object, request: Dict[str, Any]) -> Any:
        if request["op"] == "begin" or (request["op"] == "call" and request.get("method") in {"reset", "reload", "step"}):
            with client._state_lock:
                assert client._snapshots == {}
            invalidated.append(request["op"])
        return original(session, resource, request)

    monkeypatch.setattr(server, "_execute", execute)
    remote.reset(seed=8)
    assert client._snapshots[node.resource_id]["revision"] > old["revision"]
    np.testing.assert_array_equal(node.get_observation()["a"], np.full(2, 8, np.float32))
    if not component_mode:
        assert set(client._snapshots["env"]["data"]) == {"context", "observation", "info"}
    remote.reload()
    assert node.get_observation()["a"].shape == (3,)
    original_step = hosted.world.step

    def fail() -> float:
        raise RuntimeError("Step failed")

    monkeypatch.setattr(hosted.world, "step", fail)
    with pytest.raises(RemoteError, match="Step failed"):
        remote.step(action)
    assert client._snapshots == {}
    monkeypatch.setattr(hosted.world, "step", original_step)
    remote.reset(seed=11)
    np.testing.assert_array_equal(node.get_observation()["a"], np.full(3, 11, np.float32))
    assert invalidated


@pytest.mark.parametrize("close_kind", ["client", "detach"])
def test_controller_cache_abort_revision_and_close_invalidation(hosting, monkeypatch, close_kind):
    server, connect = hosting
    hosted = make_composed()
    server.register("env", hosted)
    requests = record_requests(server, monkeypatch)
    client = connect()
    node = client.node("env/node")
    remote = RemoteWorldEnv(node.world, node)
    remote.reset()
    with pytest.raises(ValueError, match="interrupt"):
        with client.operation(node.resource_id):
            node.world.step()
            raise ValueError("interrupt")
    assert client._snapshots == {}
    with pytest.raises(RemoteError) as error:
        node.get_observation()
    assert error.value.code == "faulted"
    remote.reset()
    domain = server._resources[node.resource_id].domain
    domain.executor.submit(setattr, domain, "revision", domain.revision + 1).result()
    client.describe(node.resource_id)
    assert client._snapshots == {}
    requests.clear()
    node.get_observation()
    assert len(requests) == 1
    remote.reset()
    assert client._snapshots
    if close_kind == "detach":
        session = next(session for session in server._sessions if session.id == client.session_id)
        session.close()
        wait_until(lambda: client.closed)
    else:
        client.close()
    assert client._snapshots == {}
    with pytest.raises(ConnectionError):
        node.get_observation()


@pytest.mark.parametrize("observe", [False, True])
def test_controller_cache_release_handoff_and_reacquire(hosting, monkeypatch, observe):
    server, connect = hosting
    server.register("env", make_composed())
    requests = record_requests(server, monkeypatch)
    first, second = connect(), connect()
    node = first.node("env/node")
    remote = RemoteWorldEnv(node.world, node)
    stream = first.subscribe(node.resource_id) if observe else None
    remote.reset()
    if stream is not None:
        stream.next(timeout=2)
    first.release(node.resource_id)
    assert first._snapshots == {}
    expected = second.env("env").reset(seed=9)[1]["a"]
    if stream is not None:
        assert stream.next(timeout=2)["sequence"] == 2
        assert first._snapshots == {}  # Observer delivery never reclaims controller caching.
    requests.clear()
    with pytest.raises(RemoteError) as error:
        node.get_observation()
    assert error.value.code == "ownership_conflict"
    assert len(requests) == 1
    second.release("env")
    first.acquire(node.resource_id)
    assert first._snapshots == {}
    requests.clear()
    np.testing.assert_array_equal(node.get_observation()["a"], expected)
    assert len(requests) == 1
    with first.operation(node.resource_id):
        pass
    requests.clear()
    node.get_observation()
    assert requests == []


def test_controller_cache_drops_fields_on_factory_revival(monkeypatch):
    created = []

    def factory() -> WorldEnv:
        env = make_composed()
        env.world.tick = 100 + len(created)
        env.node.nodes[0].observation_space = box((len(created) + 1,))
        created.append(env)
        return env

    with RemoteServer(idle_timeout=0.2) as server:
        server.register("env", factory)
        requests = record_requests(server, monkeypatch)
        with server.connect() as client:
            node = client.node("env/node/a")
            client.env("env").reset()
            revision = client._snapshots[node.resource_id]["revision"]
            client.release("env")
            assert client._snapshots == {}
            domain = server._resources["env"].domain
            wait_until(lambda: is_pending(domain))
            requests.clear()
            observation = node.get_observation()
            assert len(requests) == 1
            assert len(created) == 2
            np.testing.assert_array_equal(observation, np.full(2, 101, np.float32))
            assert node.descriptor["revision"] > revision
            assert client._snapshots == {}


def test_controller_cache_freezes_values_and_ignores_older_boundaries(hosting, monkeypatch):
    server, connect = hosting
    hosted = CounterEnv()
    server.register("env", hosted)
    client = connect()
    remote = client.env("env")
    responses = []
    original = server._request

    def request(session: _Session, message: Dict[str, Any]) -> Dict[str, Any]:
        response = original(session, message)
        if message.get("method") in {"reset", "step"}:
            responses.append(response["result"]["descriptors"])
        return response

    monkeypatch.setattr(server, "_request", request)
    with client.subscribe("env") as stream:
        remote.reset()
        stream.next(timeout=2)
        previous = client._snapshots["env"]
        remote.step(np.ones(2, np.float32))
        stream.next(timeout=2)
        current = client._snapshots["env"]
        assert current["sequence"] == previous["sequence"] + 1
        np.testing.assert_array_equal(previous["data"]["observation"]["position"], np.zeros(2))
        client._update(responses[0])
        assert client._snapshots["env"] is current
        assert client.descriptors["env"]["sequence"] == current["sequence"]


def test_controller_cache_getter_waits_for_other_user_thread_boundary(hosting, monkeypatch):
    server, connect = hosting
    hosted = make_composed()
    server.register("env", hosted)
    requests = record_requests(server, monkeypatch)
    client = connect()
    node = client.node("env/node")
    child = node.get_node("a")
    remote = RemoteWorldEnv(node.world, node)
    remote.reset()
    started, resume, reading = Event(), Event(), Event()
    original = hosted.world.step

    def step() -> float:
        started.set()
        assert resume.wait(5)
        return original()

    def read() -> Any:
        reading.set()
        return child.get_observation()

    monkeypatch.setattr(hosted.world, "step", step)
    requests.clear()
    with ThreadPoolExecutor(max_workers=2) as executor:
        stepping = executor.submit(remote.step, {"a": np.ones(1, np.float32), "b": np.ones(1, np.float32)})
        try:
            assert started.wait(3)
            getter = executor.submit(read)
            assert reading.wait(3)
            assert not getter.done()
        finally:
            resume.set()
        result = stepping.result(timeout=3)
        assert_tree(getter.result(timeout=3), result[0]["a"])
    assert not any(request.get("method") == "get_observation" and request["resource_id"] == child.resource_id
                   for request in requests)


def test_controller_cache_capture_failure_falls_back_to_live_read(hosting, monkeypatch):
    server, connect = hosting
    hosted = make_composed()
    server.register("env", hosted)
    requests = record_requests(server, monkeypatch)
    client = connect()
    node = client.node("env/node/a")
    original = hosted.node.nodes[0].get_info

    def fail() -> Dict[str, Any]:
        raise RuntimeError("Info unavailable")

    # An unavailable optional cache must not make an otherwise successful boundary fail.
    monkeypatch.setattr(hosted.node.nodes[0], "get_info", fail)
    with client.operation(node.world.resource_id, "reset"):
        node.world.reset()
    assert node.resource_id not in client._snapshots
    monkeypatch.setattr(hosted.node.nodes[0], "get_info", original)
    requests.clear()
    assert node.get_info() == {"tick": 0}
    assert len(requests) == 1


def test_controller_cache_optional_fields_respect_message_limit(hosting, monkeypatch):
    server, connect = hosting
    world = TickWorld()
    hosted = TickNode(world)
    hosted.observation_space = box((10000,))
    server.register("node", hosted)
    requests = record_requests(server, monkeypatch)
    client = connect(max_message_size=12000)
    node = client.node("node")
    with client.operation(node.resource_id, "reset"):
        node.world.reset()
    assert client._snapshots == {}
    requests.clear()
    assert node.get_info() == {"tick": 0}
    assert len(requests) == 1


def test_controller_field_selection_preserves_observer_publication(hosting, monkeypatch):
    server, connect = hosting
    server.register("env", make_composed())
    requests = record_requests(server, monkeypatch)
    controller, observer = connect(), connect()
    node = controller.node("env/node")
    selected = controller.subscribe(node.resource_id, ["observation"])
    normal = observer.subscribe(node.resource_id)
    remote = RemoteWorldEnv(node.world, node)
    remote.reset()
    assert set(selected.next(timeout=2)["data"]) == {"observation"}
    assert set(normal.next(timeout=2)["data"]) == {"observation", "context", "info", "reward", "terminated", "truncated"}
    assert set(controller._snapshots[node.resource_id]["data"]) == {"observation"}
    requests.clear()
    node.get_observation()
    assert requests == []
    node.get_reward()
    node.render()
    assert [request["method"] for request in requests] == ["get_reward", "render"]


@pytest.mark.parametrize("backend_name", ["pytorch", "jax"])
def test_controller_cached_getters_convert_backend(hosting, monkeypatch, backend_name):
    if backend_name == "pytorch":
        pytest.importorskip("torch")
        from unienv_interface.backends.pytorch import PyTorchComputeBackend as backend
        device = "cpu"
    else:
        jax = pytest.importorskip("jax")
        from unienv_interface.backends.jax import JaxComputeBackend as backend
        device = jax.devices("cpu")[0]
    server, connect = hosting
    server.register("env", make_composed())
    requests = record_requests(server, monkeypatch)
    client = connect(backend=backend, device=device)
    client.env("env").reset(seed=3)
    node = client.node("env/node/a")
    requests.clear()
    observation = node.get_observation()
    assert requests == []
    assert backend.is_backendarray(observation)
    np.testing.assert_array_equal(backend.to_numpy(observation), np.full(2, 3, np.float32))


def test_controller_cache_keeps_complete_publication_on_response_error(hosting, monkeypatch):
    server, connect = hosting
    server.register("env", make_composed())
    controller, observer = connect(), connect()
    node = controller.node("env/node")
    remote = RemoteWorldEnv(node.world, node)
    remote.reset()
    stream = observer.subscribe(node.resource_id, ["observation"])
    original = server._freeze_result

    def freeze_result(session: _Session, request: Dict[str, Any], result: Any) -> Any:
        if request["op"] == "complete":
            raise CodecError("Completion response failed")
        return original(session, request, result)

    monkeypatch.setattr(server, "_freeze_result", freeze_result)
    with pytest.raises(RemoteError, match="Completion response failed"):
        with controller.operation(node.resource_id):
            node.world.step()
    assert controller._snapshots == {}
    assert stream.next(timeout=2)["sequence"] == 2


def test_detached_node_lifecycle_snapshots_and_proxy(hosting, monkeypatch):
    server, connect = hosting
    hosted = DeviceNode("arm")
    server.register("arm", hosted)
    server.register("world", TickWorld())
    requests = record_requests(server, monkeypatch)
    controller, observer = connect(), connect()
    descriptor = controller.describe("arm")
    assert descriptor["world_id"] is None
    assert descriptor["domain_id"] == "arm"
    assert descriptor["kind"] == "node"
    assert set(controller.descriptors) == {"arm", "world"}
    node = controller.node("arm")
    assert node.world is None
    assert node.backend is controller.backend and node.device == controller.device
    with pytest.raises(TypeError, match="detached nodes"):
        RemoteWorldEnv(node.world, node)
    with pytest.raises(ValueError, match="Detached nodes"):
        RemoteWorldEnv(controller.world("world"), node)
    with pytest.raises(RemoteError, match="components mode"):
        controller.acquire("arm", mode="env")
    controller.acquire("arm")
    stream = observer.subscribe("arm")
    for kind in ("reset", "reload"):
        with controller.operation("arm", kind):
            getattr(node, kind)(seed=3)
            getattr(node, "after_" + kind)()
        event = stream.next(timeout=2)
        assert event["kind"] == kind
        assert event["descriptor"]["world_id"] is None
        np.testing.assert_array_equal(event["data"]["observation"], np.array([3], np.float32))
    with controller.operation("arm"):
        node.set_next_action(np.array([2], np.float32))
        node.post_environment_step(0.01)
        np.testing.assert_array_equal(node.get_observation(), np.array([5], np.float32))
    event = stream.next(timeout=2)
    assert event["sequence"] == 3
    assert event["data"]["reward"] == 5
    requests.clear()
    assert_tree(read_node_fields(node), event["data"])
    assert requests == []
    assert controller.call("arm", "get_info") == {"name": "arm"}
    controller.release("arm")
    assert controller._snapshots == {}
    observer.acquire("arm")
    assert observer.node("arm").get_reward() == 5
    observer.release("arm")
    stream.close()
    node.close()
    assert hosted.closed_count == 0
    server.close()
    assert hosted.closed_count == 1
    assert len(hosted.threads) == 1


def test_detached_nodes_have_independent_workers_ownership_and_faults(hosting):
    server, connect = hosting
    arm, hand = DeviceNode("arm"), DeviceNode("hand")
    server.register("arm", arm)
    server.register("hand", hand)
    first, second = connect(), connect()
    arm_proxy, hand_proxy = first.node("arm"), second.node("hand")
    first.acquire("arm")
    second.acquire("hand")
    assert set(server._domains) == {("node", id(arm)), ("node", id(hand))}
    assert server._world_owners == {}
    for client, node in ((first, arm_proxy), (second, hand_proxy)):
        with client.operation(node.resource_id, "reset"):
            node.reset()

    def blocked() -> None:
        with first.operation("arm"):
            arm_proxy.set_next_action(np.array([99], np.float32))
            arm_proxy.post_environment_step(0.01)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(blocked)
        try:
            assert arm.block_started.wait(3)
            with second.operation("hand"):
                hand_proxy.set_next_action(np.array([2], np.float32))
                hand_proxy.post_environment_step(0.01)
            assert not future.done()
            assert hand_proxy.get_reward() == 2
        finally:
            arm.block_release.set()
        future.result(timeout=3)
    with pytest.raises(ValueError, match="interrupt"):
        with first.operation("arm"):
            raise ValueError("interrupt")
    arm_domain, hand_domain = server._resources["arm"].domain, server._resources["hand"].domain
    assert arm_domain.faulted and not hand_domain.faulted
    with pytest.raises(RemoteError) as error:
        arm_proxy.get_observation()
    assert error.value.code == "faulted"
    with second.operation("hand"):
        hand_proxy.post_environment_step(0.01)
    assert hand_proxy.get_reward() == 4
    with pytest.raises(RemoteError, match="detached-root node"):
        with first.operation("arm", "reset"):
            arm_proxy.after_reset()
    with first.operation("arm", "reset"):
        arm_proxy.reset(seed=1)
    assert not arm_domain.faulted
    assert arm_proxy.get_reward() == 1
    first.release("arm")
    second.acquire("arm")
    with pytest.raises(RemoteError) as error:
        first.acquire("arm")
    assert error.value.code == "ownership_conflict"
    server.close()
    assert arm.closed_count == hand.closed_count == 1
    assert len(arm.threads) == len(hand.threads) == 1
    assert arm.threads.isdisjoint(hand.threads)


def test_detached_node_children_share_root_domain(hosting):
    server, connect = hosting
    arm, hand = DeviceNode("arm/grip"), DeviceNode("hand")
    hosted = DeviceNode("devices", nodes=[arm, hand])
    server.register("devices", hosted)
    assert server.register("alias", arm) == "devices/arm%2Fgrip"
    assert len(server._domains) == 1
    assert server._world_owners == {}
    controller, observer = connect(), connect()
    root = controller.node("devices")
    child = root.get_node("arm/grip")
    assert child.resource_id == "devices/arm%2Fgrip"
    assert all(d["world_id"] is None and d["domain_id"] == "devices" for d in controller.descriptors.values())
    assert all(r.domain is server._resources["devices"].domain for r in server._resources.values())
    assert root.world is child.world is None
    stream = observer.subscribe(child.resource_id, ["observation"])
    with pytest.raises(RemoteError, match="detached-root node"):
        with controller.operation(child.resource_id, "reset"):
            child.reset()
    with controller.operation(child.resource_id, "reset"):
        root.reset(seed=4)
    np.testing.assert_array_equal(stream.next(timeout=2)["data"]["observation"], np.array([4], np.float32))
    with controller.operation(child.resource_id):
        child.set_next_action(np.array([3], np.float32))
        child.post_environment_step(0.01)
    np.testing.assert_array_equal(stream.next(timeout=2)["data"]["observation"], np.array([7], np.float32))
    assert child.get_reward() == 7
    controller.release(child.resource_id)
    assert controller._snapshots == {}
    server.close()
    assert hosted.closed_count == arm.closed_count == hand.closed_count == 1


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("backend", [None, WorldNode.backend], ids=["none", "undeclared"])
def test_detached_node_requires_backend_before_commit(lazy, backend):
    class MissingBackendNode(DeviceNode):
        device = WorldNode.device

    MissingBackendNode.backend = backend
    hosted = MissingBackendNode()
    with RemoteServer(idle_timeout=None) as server:
        if lazy:
            server.register("device", lambda: hosted)
            with connect_pending(server) as client:
                with pytest.raises(RemoteError, match="Detached nodes must declare a backend"):
                    client.describe("device")
                assert is_pending(server._resources["device"].domain)
            assert hosted.closed_count == 1
        else:
            with pytest.raises(ValueError, match="Detached nodes must declare a backend"):
                server.register("device", hosted)
            assert server._resources == {} and server._domains == {}
            assert hosted.closed_count == 0
        assert server._objects == {} and server._world_owners == {}
    if not lazy:
        hosted.close()
    assert hosted.closed_count == 1


@pytest.mark.parametrize("declared_device", [False, True])
def test_detached_node_backend_and_device_coercion(hosting, declared_device):
    class BackendOnlyNode(DeviceNode):
        device = "cpu" if declared_device else WorldNode.device

    server, connect = hosting
    hosted = BackendOnlyNode()
    server.register("device", hosted)
    with connect() as client:
        node = client.node("device")
        assert node.backend is B and node.device is None
        assert server._backend_device(hosted) == (B, "cpu" if declared_device else None)
        with client.operation("device", "reset"):
            node.reset()
        with client.operation("device"):
            node.set_next_action(np.array([2], np.float32))
            node.post_environment_step(0.01)
        np.testing.assert_array_equal(node.get_observation(), np.array([2], np.float32))


@pytest.mark.parametrize("transport", ["memory", "websocket"])
def test_detached_factory_eviction_revival_and_topology_validation(transport):
    created, factory_threads = [], set()

    def factory() -> DeviceNode:
        factory_threads.add(get_ident())
        name = "changed" if len(created) == 1 else "hand/grip"
        root = DeviceNode("arm", nodes=[DeviceNode(name, width=len(created) + 1)])
        created.append(root)
        return root

    with RemoteServer(idle_timeout=0.2) as server:
        server.register("arm", factory)
        assert created == []
        if transport == "websocket":
            pytest.importorskip("websockets")
            client = RemoteClient.connect(server.listen())
        else:
            client = connect_pending(server)
            assert created == []
        with client:
            client.describe("arm/hand%2Fgrip")
            assert len(created) == 1
            root, child = client.node("arm"), client.node("arm/hand%2Fgrip")
            assert root.world is child.world is None
            stream = client.subscribe(child.resource_id)
            with client.operation("arm", "reset"):
                root.reset(seed=2)
            stream.next(timeout=2)
            assert child.get_observation().shape == (1,)
            revision = root.descriptor["revision"]
            stream.close()
            client.release("arm")
            domain = server._resources["arm"].domain
            ids = set(server._resources)
            wait_until(lambda: is_pending(domain))
            assert created[0].closed_count == created[0].nodes[0].closed_count == 1
            with pytest.raises(RemoteError, match="topology"):
                client.describe(child.resource_id)
            assert is_pending(domain)
            assert domain.revision == revision
            assert created[1].closed_count == created[1].nodes[0].closed_count == 1
            client.describe(child.resource_id)
            assert len(created) == 3
            assert set(server._resources) == ids
            assert child.descriptor["revision"] == revision + 1
            assert child.observation_space.shape == (3,)
            assert server._world_owners == {}
            with client.operation(child.resource_id, "reset"):
                root.reset(seed=6)
            np.testing.assert_array_equal(child.get_observation(), np.full(3, 6, np.float32))
            client.release(child.resource_id)
            wait_until(lambda: is_pending(domain))
    assert len(created) == 3
    assert all(root.closed_count == root.nodes[0].closed_count == 1 for root in created)
    assert all(root.threads == root.nodes[0].threads == factory_threads for root in created)


@pytest.mark.parametrize("consumptive", [False, True], ids=["reusable-buffer", "consumptive-opt-out"])
def test_controller_capture_preserves_reusable_env_results(hosting, consumptive):
    class ReusableNode(TickNode):
        def __init__(self, world: TickWorld) -> None:
            super().__init__(world)
            self.buffer = np.zeros(2, np.float32)
            self.reads = 0
            if consumptive:
                self.remote_snapshot_fields = ()

        def after_reload(self, *, priority: int = 0, mask: Optional[np.ndarray] = None) -> None:
            super().after_reload(priority=priority, mask=mask)
            self.observation_space = box((2,))

        def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
            super().post_environment_step(dt, priority=priority)
            if not consumptive:
                self.buffer[:] = self.world.tick

        def get_observation(self) -> np.ndarray:
            self.reads += 1
            self.buffer += 1
            return self.buffer

    local_world, hosted_world = TickWorld(), TickWorld()
    local = WorldEnv(local_world, ReusableNode(local_world))
    hosted = WorldEnv(hosted_world, ReusableNode(hosted_world))
    server, connect = hosting
    server.register("env", hosted)
    client = connect()
    remote = client.env("env")
    assert not client._subscriptions
    for method, args in (("reset", ()), ("step", (np.ones(1, np.float32),)), ("step", (np.ones(1, np.float32),))):
        expected = getattr(local, method)(*args)
        actual = getattr(remote, method)(*args)
        assert_tree(actual, expected)
        fields = (("context", "observation", "info") if method == "reset"
                  else ("observation", "reward", "terminated", "truncated", "info"))
        assert_tree(client._snapshots["env"]["data"], dict(zip(fields, actual)))
        if consumptive:
            assert client._snapshots["env/node"]["data"] == {}
            assert hosted.node.reads == local.node.reads
        else:
            # The optional read really refreshed the reused server buffer, but
            # neither the mandatory value nor the Env snapshot can change with it.
            assert hosted.node.reads == 2 * local.node.reads
            assert not np.array_equal(hosted.node.buffer, actual[1 if method == "reset" else 0])


@pytest.mark.parametrize("component_mode", [False, True])
def test_boundary_capture_reuses_overlapping_observer_fields(hosting, component_mode):
    class CountingNode(TickNode):
        def __init__(self, world: TickWorld) -> None:
            super().__init__(world)
            self.observation_reads = 0

        def get_observation(self) -> np.ndarray:
            self.observation_reads += 1
            return super().get_observation()

    server, connect = hosting
    world = TickWorld()
    hosted = CountingNode(world)
    server.register("env", WorldEnv(world, hosted))
    controller, observer = connect(), connect()
    node = controller.node("env/node")
    remote = RemoteWorldEnv(node.world, node) if component_mode else controller.env("env")
    first = observer.subscribe(node.resource_id, ["observation"])
    second = observer.subscribe(node.resource_id, ["observation", "info"])
    for method, args in (("reset", ()), ("step", (np.ones(1, np.float32),))):
        before = hosted.observation_reads
        result = getattr(remote, method)(*args)
        assert hosted.observation_reads - before == 2  # one lifecycle read, one shared boundary capture
        expected = result[1 if method == "reset" else 0]
        assert_tree(first.next(timeout=2)["data"]["observation"], expected)
        assert_tree(second.next(timeout=2)["data"]["observation"], expected)
        assert_tree(node.get_observation(), expected)
        assert hosted.observation_reads - before == 2


def test_boundary_capture_does_not_retry_failed_getters(hosting):
    class FailingNode(DeviceNode):
        reads = 0

        def get_observation(self) -> np.ndarray:
            self.reads += 1
            raise RuntimeError("Sensor unavailable")

    server, connect = hosting
    hosted = FailingNode()
    server.register("node", hosted)
    controller, observer = connect(), connect()
    node = controller.node("node")
    streams = [observer.subscribe("node", ["observation"]), observer.subscribe("node", ["observation", "info"])]
    with controller.operation("node", "reset"):
        node.reset()
    for stream in streams:
        with pytest.raises(RemoteError, match="Sensor unavailable"):
            stream.next(timeout=2)
    assert hosted.reads == 1
    assert "node" not in controller._snapshots


@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_cached_getter_results_do_not_alias_cache(hosting, monkeypatch, backend_name):
    if backend_name == "pytorch":
        pytest.importorskip("torch")
        from unienv_interface.backends.pytorch import PyTorchComputeBackend as backend
        device = "cpu"
    else:
        backend, device = B, None

    class InfoNode(DeviceNode):
        def get_info(self) -> Dict[str, Any]:
            return {"position": self.value, "labels": [self.name]}

    server, connect = hosting
    server.register("node", InfoNode())
    requests = record_requests(server, monkeypatch)
    client = connect(backend=backend, device=device)
    node = client.node("node")
    with client.operation("node", "reset"):
        node.reset(seed=3)
    requests.clear()
    observation = node.get_observation()
    observation[:] = 99
    info = node.get_info()
    info["position"][:] = 88
    info["labels"].append("changed")
    np.testing.assert_array_equal(backend.to_numpy(node.get_observation()), np.array([3], np.float32))
    fresh = node.get_info()
    np.testing.assert_array_equal(backend.to_numpy(fresh["position"]), np.array([3], np.float32))
    assert fresh["labels"] == ["device"]
    np.testing.assert_array_equal(client._snapshots["node"]["data"]["observation"], np.array([3], np.float32))
    assert requests == []
