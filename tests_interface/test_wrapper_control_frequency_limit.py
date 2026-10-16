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


LOGGER = "unienv_interface.wrapper.control_frequency_limit"


def _late_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER]


def test_late_actions_are_logged_rate_limited(clock, caplog, capsys):
    caplog.set_level("WARNING", logger=LOGGER)
    wrapper = ControlFrequencyLimitWrapper(StubEnv(), control_dt=0.1, late_warning_interval=5.0)
    wrapper.reset(seed=0)
    clock.now += 0.35
    wrapper.step(np.zeros(2, np.float32))
    messages = _late_messages(caplog)
    assert messages == ["Action is late by 0.250s (control_dt=0.100s)"]
    assert caplog.records[0].levelname == "WARNING"
    for extra in (0.15, 0.6, 0.2):  # late again, within the 5 s window: not logged
        clock.now += extra
        wrapper.step(np.zeros(2, np.float32))
    assert len(_late_messages(caplog)) == 1
    clock.now += 4.5  # 5.45 s after the first message
    wrapper.step(np.zeros(2, np.float32))
    messages = _late_messages(caplog)
    assert len(messages) == 2
    assert messages[1].startswith("Action is late by 4.400s (control_dt=0.100s); 3 more late action(s), up to 0.500s late")
    assert capsys.readouterr().out == ""  # no print to stdout any more


def test_on_time_steps_do_not_log(clock, caplog):
    caplog.set_level("WARNING", logger=LOGGER)
    wrapper = ControlFrequencyLimitWrapper(StubEnv(), control_dt=0.1)
    wrapper.reset(seed=0)
    for _ in range(5):
        wrapper.step(np.zeros(2, np.float32))
    assert _late_messages(caplog) == []


def test_late_warnings_can_be_disabled(clock, caplog):
    caplog.set_level("WARNING", logger=LOGGER)
    wrapper = ControlFrequencyLimitWrapper(StubEnv(), control_dt=0.1, late_warning_interval=None)
    wrapper.reset(seed=0)
    clock.now += 1.0
    wrapper.step(np.zeros(2, np.float32))
    assert _late_messages(caplog) == []


def test_zero_interval_logs_every_late_action(clock, caplog):
    caplog.set_level("WARNING", logger=LOGGER)
    wrapper = ControlFrequencyLimitWrapper(StubEnv(), control_dt=0.1, late_warning_interval=0.0)
    wrapper.reset(seed=0)
    for _ in range(3):
        clock.now += 0.3
        wrapper.step(np.zeros(2, np.float32))
    assert len(_late_messages(caplog)) == 3
