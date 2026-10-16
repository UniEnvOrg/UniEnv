"""Repository-wide pytest configuration shared by tests_interface, tests_data and tests_utils."""

import os

import pytest

try:
    from hypothesis import HealthCheck, settings

    settings.register_profile(
        "default",
        deadline=None,
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
    )
    settings.register_profile("ci", parent=settings.get_profile("default"), max_examples=100)
    settings.register_profile("dev", parent=settings.get_profile("default"), max_examples=15)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
except ImportError:  # hypothesis is a dev dependency; property tests importorskip it
    pass

from tests_common.envs import StubEnv, StubFuncEnv


@pytest.fixture
def stub_env():
    env = StubEnv()
    yield env
    env.close()


@pytest.fixture(params=[None, 3], ids=["unbatched", "batched3"])
def any_stub_env(request):
    """StubEnv both unbatched and batched (batch_size=3)."""
    env = StubEnv(batch_size=request.param)
    yield env
    env.close()


@pytest.fixture(params=[None, 3], ids=["unbatched", "batched3"])
def any_stub_func_env(request):
    return StubFuncEnv(batch_size=request.param)
