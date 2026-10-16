"""Tests for the stateful ``Wrapper`` base classes in ``unienv_interface.env_base.wrapper``."""

import numpy as np
import pytest

from unienv_interface.env_base.wrapper import ActionWrapper, ContextObservationWrapper, Wrapper
from tests_common.envs import StubEnv


class _Plain(Wrapper):
    pass


def test_rng_falls_through_to_inner_env_until_overridden(stub_env):
    wrapper = _Plain(_Plain(stub_env))
    assert wrapper.rng is stub_env.rng
    own = np.random.default_rng(123)
    wrapper.rng = own
    assert wrapper.rng is own
    assert wrapper.env.rng is stub_env.rng


def test_rng_with_jax_key_does_not_call_bool():
    jax = pytest.importorskip("jax")
    env = StubEnv()
    env.rng = jax.random.key(0)
    wrapper = _Plain(env)
    assert wrapper.rng is env.rng
    wrapper.rng = jax.random.key(1)
    assert not bool(jax.numpy.all(jax.random.key_data(wrapper.rng) == jax.random.key_data(env.rng)))
