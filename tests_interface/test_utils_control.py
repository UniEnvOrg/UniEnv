"""Tests for ``unienv_interface.utils.control_util``."""

from fractions import Fraction

import pytest

from unienv_interface.utils.control_util import find_best_timestep, is_integer_multiple


def test_find_best_timestep_float_periods():
    dt, steps = find_best_timestep([0.01, 0.02, 0.05])
    assert dt == pytest.approx(0.01)
    assert steps == [1, 2, 5]


@pytest.mark.parametrize("periods,expected_dt,expected_steps", [
    ([0.1], 0.1, [1]),
    ([0.02, 0.03], 0.01, [2, 3]),
    ([1 / 3, 0.5], 1 / 6, [2, 3]),
    ([4, 6, 10], 2, [2, 3, 5]),
    ((p for p in [0.002, 0.01]), 0.002, [1, 5]),
])
def test_find_best_timestep_cases(periods, expected_dt, expected_steps):
    dt, steps = find_best_timestep(periods)
    assert dt == pytest.approx(expected_dt)
    assert steps == expected_steps


def test_find_best_timestep_fraction():
    dt, steps = find_best_timestep([0.25, 0.5], return_fraction=True)
    assert dt == Fraction(1, 4)
    assert steps == [1, 2]


@pytest.mark.parametrize("bad", [[], [0.1, 0.0], [-0.1]])
def test_find_best_timestep_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        find_best_timestep(bad)


@pytest.mark.parametrize("value,base,expected", [
    (0.1, 0.02, True), (0.3, 0.1, True), (0.07, 0.01, True), (1.0, 1 / 3, True), (0.02, 0.02, True),
    (0.05, 0.02, False), (0.01, 0.02, False), (0.25, 0.1, False), (0.0, 0.1, False),
])
def test_is_integer_multiple(value, base, expected):
    assert is_integer_multiple(value, base) is expected
