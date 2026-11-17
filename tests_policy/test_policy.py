"""``unienv_policy``: the stateful / functional policy interfaces and their wrappers."""
from typing import Optional

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace
from unienv_policy import FuncPolicy, FuncPolicyBasedPolicy, FuncPolicyWrapper, Policy, PolicyWrapper

OBS = BoxSpace(B, -10.0, 10.0, np.float32, shape=(3,))
ACT = BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,))
CTX = BoxSpace(B, 0.0, 1.0, np.float32, shape=(1,))


def test_top_level_exports():
    import unienv_policy
    for name in ("Policy", "PolicyWrapper", "FuncPolicy", "FuncPolicyWrapper", "FuncPolicyBasedPolicy"):
        assert getattr(unienv_policy, name) is getattr(unienv_policy.base, name)


class CountingPolicy(Policy):
    """Acts with the clipped first two observation entries; counts resets and transitions."""
    backend = B
    device = None
    observation_space = OBS
    action_space = ACT
    context_space = CTX

    def __init__(self, seed: Optional[int] = 0):
        self.rng = B.random.random_number_generator(seed)
        self.resets = []
        self.transitions = 0
        self.closed = False

    def reset(self, context, observation, *, seed=None, mask=None, **kwargs):
        self.resets.append((context, observation, seed, mask))

    def observe_transition(self, action_taken, new_observation):
        self.transitions += 1

    def compute_action(self, observation, **kwargs):
        return np.clip(observation[:2], -1.0, 1.0)

    def close(self):
        self.closed = True


def test_policy_defaults_and_sampling():
    class Minimal(Policy):
        backend = B
        device = None
        observation_space = OBS
        action_space = ACT

        def compute_action(self, observation, **kwargs):
            return np.zeros(2, np.float32)

    policy = Minimal()
    policy.rng = B.random.random_number_generator(0)
    assert policy.context_space is None and policy.sample_context() is None
    assert ACT.contains(policy.sample_action()) and OBS.contains(policy.sample_observation())
    assert policy.reset(None, policy.sample_observation()) is None
    assert policy.observe_transition(np.zeros(2, np.float32), np.zeros(3, np.float32)) is None
    frame = np.zeros((4, 4, 3), np.uint8)
    assert policy.visualize_action(frame, None, None) is frame
    assert policy.unwrapped is policy and policy.prev_wrapper_layer is None
    with Minimal() as entered:
        assert isinstance(entered, Minimal)
    with pytest.raises(TypeError):
        Policy()  # compute_action is abstract


def test_policy_wrapper_delegates_and_overrides():
    inner = CountingPolicy()
    wrapper = PolicyWrapper(inner)
    assert wrapper.unwrapped is inner and wrapper.prev_wrapper_layer is inner
    assert wrapper.backend is B and wrapper.device is None and wrapper.batch_size is None
    assert wrapper.observation_space is OBS and wrapper.action_space is ACT and wrapper.context_space is CTX
    assert wrapper.metadata == {}

    obs = np.array([0.5, -3.0, 2.0], np.float32)
    wrapper.reset(np.zeros(1, np.float32), obs, seed=3, mask=None)
    assert inner.resets[-1][2] == 3
    np.testing.assert_array_equal(wrapper.compute_action(obs), np.array([0.5, -1.0], np.float32))
    wrapper.observe_transition(np.zeros(2, np.float32), obs)
    assert inner.transitions == 1

    other = BoxSpace(B, -2.0, 2.0, np.float32, shape=(2,))
    wrapper.action_space = other
    wrapper.context_space = None
    wrapper.metadata = {"name": "wrapped"}
    assert wrapper.action_space is other and inner.action_space is ACT
    assert wrapper.context_space is None and inner.context_space is CTX
    assert wrapper.metadata == {"name": "wrapped"}

    rng = B.random.random_number_generator(9)
    wrapper.rng = rng  # no rng of its own: goes to the wrapped policy
    assert inner.rng is rng and wrapper.rng is rng
    wrapper.close()
    assert inner.closed


def test_wrapper_attr_lookup_walks_the_stack():
    inner = CountingPolicy()
    inner.special = 1
    outer = PolicyWrapper(PolicyWrapper(inner))
    assert outer.has_wrapper_attr("special") and outer.get_wrapper_attr("special") == 1
    outer.set_wrapper_attr("special", 2)
    assert inner.special == 2
    assert not outer.has_wrapper_attr("missing")
    with pytest.raises(AttributeError):
        outer.get_wrapper_attr("missing")
    with pytest.raises(AttributeError):
        outer.set_wrapper_attr("missing", 0)
    assert "CountingPolicy" in str(outer)


class RunningSumFuncPolicy(FuncPolicy):
    """State: running sum of observations per env; action: the first two entries of the sum, clipped."""
    backend = B
    device = None
    observation_space = OBS
    action_space = ACT

    def __init__(self):
        self.closed_with = None

    def initial(self, context, observation, *, batch_size, seed=None, **kwargs):
        return {"sum": np.array(observation, np.float32, copy=True), "seed": seed}

    def reset(self, state, context, observation, *, seed=None, mask=None, **kwargs):
        total = state["sum"].copy()
        if mask is None:
            total = np.array(observation, np.float32, copy=True)
        else:
            total[mask] = observation
        return {**state, "sum": total}

    def observe_transition(self, state, action_taken, new_observation):
        return {**state, "sum": state["sum"] + new_observation}

    def compute_action(self, state, observation, **kwargs):
        return state, np.clip(state["sum"][..., :2], -1.0, 1.0)

    def close(self, state):
        self.closed_with = state


def test_func_policy_based_policy_runs_the_functional_policy():
    func = RunningSumFuncPolicy()
    policy = FuncPolicyBasedPolicy(func, batch_size=2, seed=5)
    assert policy.batch_size == 2 and policy.backend is B
    assert policy.action_space is ACT and policy.observation_space is OBS and policy.context_space is None
    with pytest.raises(AssertionError):
        policy.compute_action(np.zeros((2, 3), np.float32))  # not reset yet

    obs = np.array([[0.25, 0.0, 0.0], [0.5, 0.5, 0.0]], np.float32)
    with pytest.raises(AssertionError):
        policy.reset(None, obs, mask=np.array([True, False]))  # no mask on the first reset
    policy.reset(None, obs)
    assert policy.state["seed"] == 5  # the constructor's seed
    policy.observe_transition(np.zeros((2, 2), np.float32), obs)
    np.testing.assert_allclose(policy.compute_action(obs), [[0.5, 0.0], [1.0, 1.0]])

    policy.reset(None, np.full((1, 3), 0.125, np.float32), mask=np.array([False, True]))
    np.testing.assert_allclose(policy.compute_action(obs), [[0.5, 0.0], [0.125, 0.125]])

    final_state = policy.state
    policy.close()
    assert func.closed_with is final_state and policy.state is None
    policy.reset(None, obs, seed=7)  # usable again after close: starts from `initial`
    assert policy.state["seed"] == 7


def test_func_policy_wrapper_and_attr_lookup():
    func = RunningSumFuncPolicy()
    func.gain = 1.0
    wrapper = FuncPolicyWrapper(func)
    assert wrapper.unwrapped is func and wrapper.prev_wrapper_layer is func
    assert wrapper.observation_space is OBS and wrapper.backend is B
    obs = np.array([0.5, 0.25, 0.0], np.float32)
    state = wrapper.initial(None, obs, batch_size=None, seed=1)
    state, action = wrapper.compute_action(wrapper.observe_transition(state, np.zeros(2, np.float32), obs), obs)
    np.testing.assert_allclose(action, [1.0, 0.5])

    policy = FuncPolicyBasedPolicy(wrapper)
    assert policy.get_wrapper_attr("gain") == 1.0
    policy.set_wrapper_attr("gain", 2.0)
    assert func.gain == 2.0
    with pytest.raises(AttributeError):
        policy.set_wrapper_attr("missing", 0)
    policy.metadata = {"kind": "adapter"}
    assert policy.metadata == {"kind": "adapter"} and func.metadata == {}
