"""Tests for ``ControlFrequencyLimitWrapper`` using a fake monotonic clock."""

import numpy as np
import pytest

from unienv_interface.wrapper import ControlFrequencyLimitWrapper
from unienv_interface.wrapper import control_frequency_limit as cfl
from tests_common.envs import StubEnv


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, dt):
        self.sleeps.append(dt)
        self.now += dt


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(cfl.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(cfl.time, "sleep", fake.sleep)
    return fake


def test_tight_loop_runs_at_requested_rate(clock):
    wrapper = ControlFrequencyLimitWrapper(StubEnv(), control_dt=0.1)
    wrapper.reset(seed=0)
    start = clock.now
    for _ in range(10):
        wrapper.step(np.zeros(2, np.float32))
    assert clock.now - start == pytest.approx(1.0)
    assert clock.sleeps == pytest.approx([0.1] * 10)


def test_sleep_accounts_for_time_spent_between_steps(clock):
    wrapper = ControlFrequencyLimitWrapper(StubEnv(), control_dt=0.1)
    wrapper.reset(seed=0)
    clock.now += 0.03  # caller computes for 30 ms
    wrapper.step(np.zeros(2, np.float32))
    assert clock.sleeps[-1] == pytest.approx(0.07)
    clock.now += 0.25  # caller is late: no sleep
    wrapper.step(np.zeros(2, np.float32))
    assert len(clock.sleeps) == 1
    clock.now += 0.05
    wrapper.step(np.zeros(2, np.float32))
    assert clock.sleeps[-1] == pytest.approx(0.05)


def test_control_dt_defaults_to_render_fps(clock):
    wrapper = ControlFrequencyLimitWrapper(StubEnv())
    assert wrapper.control_dt == pytest.approx(1 / StubEnv.render_fps)
    env = StubEnv()
    env.render_fps = None
    with pytest.raises(AssertionError):
        ControlFrequencyLimitWrapper(env)
