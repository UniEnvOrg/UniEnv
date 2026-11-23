"""Remote policies: a ``Policy`` hosted by a ``RemoteServer``, over every transport."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.policy import FuncPolicy, FuncPolicyBasedPolicy, Policy, PolicyWrapper
from unienv_interface.remote import RemoteError, RemotePolicy, RemoteServer
from unienv_interface.space import BoxSpace

from tests_common.remote_envs import TRANSPORTS, ProbeEnv, connect

OBS = BoxSpace(B, -10.0, 10.0, np.float32, shape=(3,))
ACT = BoxSpace(B, -10.0, 10.0, np.float32, shape=(2,))
CTX = BoxSpace(B, 0.0, 1.0, np.float32, shape=(1,))


class LatestObservationPolicy(Policy):
    """Acts with the first two entries of the latest observation (times ``scale``); records its calls."""

    backend, device = B, None
    observation_space, action_space, context_space = OBS, ACT, CTX
    metadata = {"policy": "latest-observation"}

    def __init__(self, batch_size: Optional[int] = None) -> None:
        self.batch_size = batch_size
        if batch_size is not None:
            self.observation_space = BoxSpace(B, -10.0, 10.0, np.float32, shape=(batch_size, 3))
            self.action_space = BoxSpace(B, -10.0, 10.0, np.float32, shape=(batch_size, 2))
            self.context_space = BoxSpace(B, 0.0, 1.0, np.float32, shape=(batch_size, 1))
        self.rng = B.random.random_number_generator(0)
        self.context: Any = None
        self.observation: Any = None
        self.resets: List[Any] = []
        self.transitions: List[Any] = []
        self.dts: List[float] = []
        self.closed = False

    def reset(self, context, observation, *, seed=None, mask=None, **kwargs):
        if mask is None:
            self.context, self.observation = context, observation
        else:
            self.observation = np.where(mask[:, None], observation, self.observation)
        self.resets.append((seed, mask, kwargs))

    def observe_transition(self, action_taken, new_observation, dt):
        assert type(dt) is float
        self.transitions.append(action_taken)
        self.dts.append(dt)
        self.observation = new_observation

    def get_action(self, scale: float = 1.0, **kwargs):
        return (self.observation[..., :2] * scale).astype(np.float32)

    def visualize_action(self, environment_render, observation, action, **kwargs):
        frame = environment_render.copy()
        frame[0, 0] = 255
        return frame

    def close(self):
        self.closed = True


class CountingFuncPolicy(FuncPolicy):
    """State: steps since reset; action: [steps, first observation entry]."""

    backend, device, batch_size = B, None, None
    observation_space, action_space, context_space = OBS, ACT, None

    def initial(self, context, observation, *, seed=None, **kwargs):
        return {"steps": 0, "first": float(observation[0])}

    def reset(self, state, context, observation, *, seed=None, mask=None, **kwargs):
        return self.initial(context, observation, seed=seed)

    def observe_transition(self, state, action_taken, new_observation, dt):
        return {**state, "steps": state["steps"] + 1}

    def get_action(self, state, **kwargs):
        return state, np.array([state["steps"], state["first"]], dtype=np.float32)


@pytest.fixture(params=TRANSPORTS)
def transport(request):
    if request.param == "websocket":
        pytest.importorskip("websockets")
    return request.param


@pytest.fixture
def server():
    with RemoteServer() as server:
        yield server


def test_policy_round_trip(server, transport):
    hosted = LatestObservationPolicy()
    server.register("pi", hosted)
    with connect(server, transport) as client:
        policy = client.policy("pi")
        assert isinstance(policy, RemotePolicy) and isinstance(policy, Policy)
        assert client.resources["pi"]["kind"] == "policy" and client.resource("pi") is policy
        assert policy.batch_size is None and policy.metadata == {"policy": "latest-observation"}
        for local, remote in ((OBS, policy.observation_space), (ACT, policy.action_space),
                              (CTX, policy.context_space)):
            assert isinstance(remote, BoxSpace) and remote.shape == local.shape and remote.dtype == local.dtype

        context = np.array([0.5], np.float32)
        observation = np.array([1.0, -2.0, 3.0], np.float32)
        assert policy.reset(context, observation, seed=7, task="pick") is None
        assert hosted.resets == [(7, None, {"task": "pick"})]
        np.testing.assert_array_equal(hosted.context, context)

        action = policy.get_action()
        assert isinstance(action, np.ndarray) and action.dtype == np.float32
        np.testing.assert_array_equal(action, [1.0, -2.0])
        np.testing.assert_array_equal(policy.get_action(scale=2.0), [2.0, -4.0])  # kwargs pass through
        np.testing.assert_array_equal(policy.get_action(), action)  # no new transition, same state

        policy.observe_transition(action, np.array([4.0, 5.0, 6.0], np.float32), 0.05)
        np.testing.assert_array_equal(hosted.transitions[0], action)
        assert hosted.dts == [0.05]
        np.testing.assert_array_equal(policy.get_action(), [4.0, 5.0])

        frame = np.zeros((4, 4, 3), np.uint8)
        drawn = policy.visualize_action(frame, observation, action)
        assert drawn.shape == frame.shape and drawn[0, 0].tolist() == [255, 255, 255] and drawn[1, 1].sum() == 0

        policy.close()  # the proxy does not close the hosted policy
        assert not hosted.closed
    server.close()
    assert hosted.closed


def test_policy_drives_an_env(server, transport):
    """The usual control loop, with only the policy remote."""
    server.register("pi", LatestObservationPolicy())
    env = ProbeEnv()
    with connect(server, transport) as client:
        policy = client.policy("pi")
        context, obs, info = env.reset(seed=0)
        policy.reset(None, np.array([0.1, 0.2, 0.3], np.float32))
        for _ in range(3):
            action = policy.get_action()
            assert policy.action_space.contains(action)
            policy.observe_transition(action, np.array([0.3, 0.2, 0.1], np.float32), 0.05)
        np.testing.assert_allclose(policy.get_action(), [0.3, 0.2])


def test_batched_policy_masked_reset(server, transport):
    hosted = LatestObservationPolicy(batch_size=2)
    server.register("pi", hosted)
    with connect(server, transport) as client:
        policy = client.policy("pi")
        assert policy.batch_size == 2 and policy.observation_space.shape == (2, 3)
        policy.reset(np.zeros((2, 1), np.float32), np.array([[1, 1, 1], [2, 2, 2]], np.float32))
        mask = np.array([False, True])
        policy.reset(np.zeros((2, 1), np.float32), np.array([[9, 9, 9], [3, 3, 3]], np.float32), mask=mask)
        seed, sent_mask, _ = hosted.resets[-1]
        assert isinstance(sent_mask, np.ndarray) and sent_mask.dtype == np.bool_
        np.testing.assert_array_equal(sent_mask, mask)
        np.testing.assert_array_equal(policy.get_action(), [[1, 1], [3, 3]])  # row 0 kept its state


def test_func_policy_is_hosted_through_func_policy_based_policy(server):
    with pytest.raises(TypeError, match="FuncPolicyBasedPolicy"):
        server.register("bare", CountingFuncPolicy())
    server.register("pi", FuncPolicyBasedPolicy(CountingFuncPolicy(), seed=0))
    with server.connect() as client:
        policy = client.policy("pi")
        obs = np.array([5.0, 0.0, 0.0], np.float32)
        policy.reset(None, obs)
        np.testing.assert_array_equal(policy.get_action(), [0.0, 5.0])
        policy.observe_transition(np.zeros(2, np.float32), obs, 0.05)
        policy.observe_transition(np.zeros(2, np.float32), obs, 0.05)
        np.testing.assert_array_equal(policy.get_action(), [2.0, 5.0])


def test_wrapped_policy(server):
    class Doubled(PolicyWrapper):
        def get_action(self, **kwargs):
            return 2 * self.policy.get_action(**kwargs)

    inner = LatestObservationPolicy()
    server.register("pi", Doubled(inner))
    with server.connect() as client:
        policy = client.policy("pi")
        policy.reset(None, np.array([1.0, 2.0, 3.0], np.float32))
        np.testing.assert_array_equal(policy.get_action(), [2.0, 4.0])
        assert policy.metadata == {"policy": "latest-observation"}  # forwarded by the wrapper


def test_only_policy_methods_are_exposed(server):
    server.register("pi", LatestObservationPolicy())
    server.register("env", ProbeEnv())
    with server.connect() as client:
        assert sorted(client.resources["pi"]["methods"]) == [
            "get_action", "observe_transition", "reset", "visualize_action"]
        with pytest.raises(RemoteError, match="does not expose 'close'"):
            client.call("pi", "close")
        with pytest.raises(RemoteError, match="does not expose 'sample_action'"):
            client.call("pi", "sample_action")
        with pytest.raises(TypeError, match="is a policy, not a env"):
            client.env("pi")
        with pytest.raises(TypeError, match="is a env, not a policy"):
            client.policy("env")


def test_remote_errors_are_raised(server):
    server.register("pi", LatestObservationPolicy())
    with server.connect() as client:
        policy = client.policy("pi")
        with pytest.raises(RemoteError, match="TypeError"):
            policy.get_action()  # no reset yet: observation is None


def test_policy_backend_conversion(server):
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as T

    hosted = LatestObservationPolicy()
    server.register("pi", hosted)
    with server.connect(backend=T, device="cpu") as client:
        policy = client.policy("pi")
        assert policy.backend is T
        policy.reset(None, torch.tensor([1.0, 2.0, 3.0]))
        assert isinstance(hosted.observation, np.ndarray)  # rebuilt on the policy's backend
        action = policy.get_action()
        assert isinstance(action, torch.Tensor)  # returned on the client's
        assert action.tolist() == [1.0, 2.0]
