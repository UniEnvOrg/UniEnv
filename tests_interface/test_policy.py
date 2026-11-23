"""``unienv_interface.policy``: the stateful / functional policy interfaces and their wrappers."""
import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace
from unienv_interface.policy import FuncPolicy, FuncPolicyBasedPolicy, FuncPolicyWrapper, Policy, PolicyWrapper

OBS = BoxSpace(B, -10.0, 10.0, np.float32, shape=(3,))
ACT = BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,))
CTX = BoxSpace(B, 0.0, 1.0, np.float32, shape=(1,))
N = 2
BATCHED_OBS = BoxSpace(B, -10.0, 10.0, np.float32, shape=(N, 3))
BATCHED_ACT = BoxSpace(B, -1.0, 1.0, np.float32, shape=(N, 2))


def test_top_level_exports():
    import unienv_interface
    import unienv_interface.policy as policy_module
    for name in ("Policy", "PolicyWrapper", "FuncPolicy", "FuncPolicyWrapper", "FuncPolicyBasedPolicy"):
        assert getattr(unienv_interface, name) is getattr(policy_module, name)


class ReactivePolicy(Policy):
    """Acts with the clipped first two entries of the latest observation; records what it was told."""
    backend = B
    device = None
    observation_space = OBS
    action_space = ACT
    context_space = CTX

    def __init__(self):
        self.rng = B.random.random_number_generator(0)
        self.resets = []
        self.transitions = 0
        self.dts = []
        self.latest = None
        self.closed = False

    def reset(self, context, observation, *, seed=None, mask=None, **kwargs):
        self.resets.append((context, observation, seed, mask))
        self.context = context
        self.latest = observation

    def observe_transition(self, action_taken, new_observation, dt):
        self.transitions += 1
        self.dts.append(dt)
        self.latest = new_observation

    def get_action(self, **kwargs):
        return np.clip(self.latest[:2], -1.0, 1.0)

    def close(self):
        self.closed = True


def test_lifecycle_methods_are_abstract():
    class OnlyGetAction(Policy):
        def get_action(self, **kwargs):
            return None

    with pytest.raises(TypeError):
        OnlyGetAction()  # reset and observe_transition are where the state comes from

    class FuncWithoutReset(FuncPolicy):
        def initial(self, context, observation, *, seed=None, **kwargs):
            return None

        def observe_transition(self, state, action_taken, new_observation, dt):
            return state

        def get_action(self, state, **kwargs):
            return state, None

    with pytest.raises(TypeError):
        FuncWithoutReset()


def test_state_comes_from_reset_and_observe_transition():
    policy = ReactivePolicy()
    assert policy.batch_size is None
    policy.reset(np.zeros(1, np.float32), np.array([0.5, -3.0, 2.0], np.float32), seed=1)
    first = policy.get_action()
    np.testing.assert_array_equal(first, [0.5, -1.0])
    np.testing.assert_array_equal(policy.get_action(), first)  # reading the state does not change it
    policy.observe_transition(first, np.array([-0.25, 0.75, 0.0], np.float32), 0.05)
    np.testing.assert_array_equal(policy.get_action(), [-0.25, 0.75])
    assert policy.transitions == 1 and policy.dts == [0.05]


def test_policy_helpers():
    policy = ReactivePolicy()
    assert ACT.contains(policy.sample_action()) and OBS.contains(policy.sample_observation())
    assert CTX.contains(policy.sample_context())
    frame = np.zeros((4, 4, 3), np.uint8)
    assert policy.visualize_action(frame, None, None) is frame
    assert policy.unwrapped is policy and policy.prev_wrapper_layer is None
    with ReactivePolicy() as entered:
        assert isinstance(entered, ReactivePolicy)
    assert entered.closed


def test_policy_wrapper_delegates_and_overrides():
    inner = ReactivePolicy()
    wrapper = PolicyWrapper(inner)
    assert wrapper.unwrapped is inner and wrapper.prev_wrapper_layer is inner
    assert wrapper.backend is B and wrapper.device is None and wrapper.batch_size is None
    assert wrapper.observation_space is OBS and wrapper.action_space is ACT and wrapper.context_space is CTX
    assert wrapper.metadata == {}

    obs = np.array([0.5, -3.0, 2.0], np.float32)
    wrapper.reset(np.zeros(1, np.float32), obs, seed=3)
    assert inner.resets[-1][2] == 3
    np.testing.assert_array_equal(wrapper.get_action(), [0.5, -1.0])
    wrapper.observe_transition(np.zeros(2, np.float32), np.zeros(3, np.float32), 0.05)
    assert inner.transitions == 1 and inner.dts == [0.05]  # the wrapper passes dt on
    np.testing.assert_array_equal(wrapper.get_action(), [0.0, 0.0])

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
    inner = ReactivePolicy()
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
    assert "ReactivePolicy" in str(outer)


class RunningSumFuncPolicy(FuncPolicy):
    """Batched (N=2). State: running sum of observations per env, and a draw counter for get_action."""
    backend = B
    device = None
    batch_size = N
    observation_space = BATCHED_OBS
    action_space = BATCHED_ACT

    def __init__(self):
        self.closed_with = None

    def initial(self, context, observation, *, seed=None, **kwargs):
        return {"sum": np.array(observation, np.float32, copy=True), "seed": seed, "draws": 0}

    def reset(self, state, context, observation, *, seed=None, mask=None, **kwargs):
        total = np.array(observation, np.float32, copy=True)
        if mask is not None:
            total = np.where(mask[:, None], total, state["sum"])  # full batch in; keep the other envs
        return {**state, "sum": total}

    def observe_transition(self, state, action_taken, new_observation, dt):
        return {**state, "sum": state["sum"] + new_observation, "time": state.get("time", 0.0) + dt}

    def get_action(self, state, **kwargs):
        # bookkeeping only (like an RNG key): the sum is unchanged
        return {**state, "draws": state["draws"] + 1}, np.clip(state["sum"][..., :2], -1.0, 1.0)

    def close(self, state):
        self.closed_with = state


def test_func_policy_based_policy_runs_the_functional_policy():
    func = RunningSumFuncPolicy()
    policy = FuncPolicyBasedPolicy(func, seed=5)
    assert policy.batch_size == N  # the functional policy's
    assert policy.action_space is BATCHED_ACT and policy.observation_space is BATCHED_OBS
    assert policy.context_space is None
    with pytest.raises(AssertionError):
        policy.get_action()  # not reset yet
    with pytest.raises(AssertionError):
        policy.observe_transition(np.zeros((N, 2), np.float32), np.zeros((N, 3), np.float32), 0.05)

    obs = np.array([[0.25, 0.0, 0.0], [0.5, 0.5, 0.0]], np.float32)
    with pytest.raises(AssertionError):
        policy.reset(None, obs, mask=np.array([True, False]))  # no mask on the first reset
    policy.reset(None, obs)
    assert policy.state["seed"] == 5  # the constructor's seed
    np.testing.assert_allclose(policy.get_action(), [[0.25, 0.0], [0.5, 0.5]])
    policy.observe_transition(np.zeros((N, 2), np.float32), obs, 0.05)
    np.testing.assert_allclose(policy.get_action(), [[0.5, 0.0], [1.0, 1.0]])
    assert policy.state["draws"] == 2
    assert policy.state["time"] == 0.05  # dt reaches the functional policy

    # masked reset: the full batch comes in, only env 1 starts over
    fresh = np.array([[9.0, 9.0, 9.0], [0.125, 0.125, 0.0]], np.float32)
    policy.reset(None, fresh, mask=np.array([False, True]))
    np.testing.assert_allclose(policy.get_action(), [[0.5, 0.0], [0.125, 0.125]])

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
    assert wrapper.observation_space is BATCHED_OBS and wrapper.backend is B and wrapper.batch_size == N
    obs = np.array([[0.5, 0.25, 0.0], [0.0, 0.0, 0.0]], np.float32)
    state = wrapper.initial(None, obs, seed=1)
    state = wrapper.observe_transition(state, np.zeros((N, 2), np.float32), obs, 0.05)
    state = wrapper.reset(state, None, np.zeros((N, 3), np.float32), mask=np.array([False, True]))
    state, action = wrapper.get_action(state)
    np.testing.assert_allclose(action, [[1.0, 0.5], [0.0, 0.0]])

    policy = FuncPolicyBasedPolicy(wrapper)
    assert policy.batch_size == N
    assert policy.get_wrapper_attr("gain") == 1.0
    policy.set_wrapper_attr("gain", 2.0)
    assert func.gain == 2.0
    with pytest.raises(AttributeError):
        policy.set_wrapper_attr("missing", 0)
    policy.metadata = {"kind": "adapter"}
    assert policy.metadata == {"kind": "adapter"} and func.metadata == {}
