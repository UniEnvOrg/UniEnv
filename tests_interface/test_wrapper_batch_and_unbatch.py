"""Tests for ``BatchifyWrapper`` / ``UnBatchifyWrapper``."""

import numpy as np
import pytest

from unienv_interface.wrapper import BatchifyWrapper, UnBatchifyWrapper
from tests_common.env_stubs import ArrayStepsStubEnv as StubEnv
from tests_common.trees import assert_tree_equal


def test_batchify_batches_every_step_output():
    env = StubEnv()
    wrapper = BatchifyWrapper(env)
    assert wrapper.batch_size == 1
    assert wrapper.action_space.shape == (1, 2)
    assert wrapper.observation_space["position"].shape == (1, 2)
    assert wrapper.context_space.shape == (1, 2)

    context, obs, _ = wrapper.reset(seed=3)
    assert wrapper.context_space.contains(context)
    assert wrapper.observation_space.contains(obs)
    np.testing.assert_array_equal(obs["position"][0], env.position)

    obs, reward, terminated, truncated, _ = wrapper.step(np.full((1, 2), 0.5, np.float32))
    np.testing.assert_allclose(env.last_action, [0.5, 0.5])
    assert wrapper.observation_space.contains(obs)
    for value in (reward, terminated, truncated):
        assert isinstance(value, np.ndarray) and value.shape == (1,)
    assert terminated.dtype == bool and truncated.dtype == bool
    np.testing.assert_allclose(reward, [-np.abs(env.position).sum()])


def test_unbatchify_unbatches_every_step_output():
    env = StubEnv(batch_size=1)
    wrapper = UnBatchifyWrapper(env)
    assert wrapper.batch_size is None
    assert wrapper.action_space.shape == (2,)
    context, obs, _ = wrapper.reset(seed=3)
    assert wrapper.context_space.contains(context)
    assert wrapper.observation_space.contains(obs)
    obs, reward, terminated, truncated, _ = wrapper.step(np.full(2, 0.5, np.float32))
    np.testing.assert_allclose(env.last_action, [[0.5, 0.5]])
    assert wrapper.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert reward == pytest.approx(float(-np.abs(env.position).sum()))
    assert terminated is False and truncated is False


def test_batchify_then_unbatchify_is_transparent():
    reference = StubEnv()
    wrapped = UnBatchifyWrapper(BatchifyWrapper(StubEnv()))
    ref_out = reference.reset(seed=7)
    out = wrapped.reset(seed=7)
    assert_tree_equal(out[:2], ref_out[:2])
    for value in (0.25, -0.5, 1.0):
        action = np.full(2, value, np.float32)
        ref_out = reference.step(action)
        out = wrapped.step(action)
        assert_tree_equal(out[0], ref_out[0])
        assert out[1] == pytest.approx(ref_out[1])
        assert (out[2], out[3]) == (ref_out[2], ref_out[3])


def test_batchify_masked_reset():
    env = StubEnv()
    wrapper = BatchifyWrapper(env)
    wrapper.reset(seed=0)
    wrapper.step(np.full((1, 2), 0.5, np.float32))
    position = env.position.copy()

    _, obs, _ = wrapper.reset(mask=np.array([False]), seed=1)
    assert obs["position"].shape == (0, 2)
    np.testing.assert_array_equal(env.position, position)  # nothing reset

    context, obs, _ = wrapper.reset(mask=np.array([True]), seed=1)
    assert obs["position"].shape == (1, 2)
    np.testing.assert_array_equal(context, [[1.0, 1.0]])
    np.testing.assert_array_equal(obs["velocity"], 0.0)
