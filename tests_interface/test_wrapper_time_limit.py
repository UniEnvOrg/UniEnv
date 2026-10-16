"""Tests for ``TimeLimitWrapper``."""

import numpy as np
import pytest

from unienv_interface.wrapper import TimeLimitWrapper
from tests_common.envs import StubEnv


def test_unbatched_truncates_exactly_at_limit():
    wrapper = TimeLimitWrapper(StubEnv(), time_limit=3)
    wrapper.reset(seed=0)
    flags = [wrapper.step(np.zeros(2, np.float32))[3] for _ in range(3)]
    assert flags == [False, False, True]
    wrapper.reset(seed=0)
    assert wrapper.step(np.zeros(2, np.float32))[3] is False


def test_unbatched_termination_is_not_overridden():
    wrapper = TimeLimitWrapper(StubEnv(terminate_above=-10.0), time_limit=1)
    wrapper.reset(seed=0)
    _, _, terminated, truncated, _ = wrapper.step(np.zeros(2, np.float32))
    assert terminated is True
    assert truncated is False


def test_batched_truncation_and_masked_reset():
    env = StubEnv(batch_size=3)
    wrapper = TimeLimitWrapper(env, time_limit=2)
    wrapper.reset(seed=0)
    action = np.zeros((3, 2), np.float32)
    _, _, _, truncated, _ = wrapper.step(action)
    np.testing.assert_array_equal(truncated, [False, False, False])
    _, _, _, truncated, _ = wrapper.step(action)
    np.testing.assert_array_equal(truncated, [True, True, True])

    wrapper.reset(mask=np.array([True, False, False]), seed=1)
    _, _, _, truncated, _ = wrapper.step(action)
    np.testing.assert_array_equal(truncated, [False, True, True])
    _, _, _, truncated, _ = wrapper.step(action)
    np.testing.assert_array_equal(truncated, [True, True, True])
    wrapper.reset(seed=2)
    _, _, _, truncated, _ = wrapper.step(action)
    np.testing.assert_array_equal(truncated, [False, False, False])


def test_batched_termination_is_not_overridden():
    env = StubEnv(batch_size=2, terminate_above=0.5)
    wrapper = TimeLimitWrapper(env, time_limit=1)
    wrapper.reset(seed=0)
    env.position[:] = np.array([[0.9, 0.9], [-0.9, -0.9]], np.float32)
    _, _, terminated, truncated, _ = wrapper.step(np.zeros((2, 2), np.float32))
    np.testing.assert_array_equal(terminated, [True, False])
    # same semantics as the unbatched case: a terminated instance is not also truncated
    np.testing.assert_array_equal(truncated, [False, True])
