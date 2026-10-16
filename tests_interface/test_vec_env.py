"""Tests for ``SyncVecEnv`` / ``AsyncVecEnv`` and ``data_stack``."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.env_base.vec_env import data_stack


# ---------------------------------------------------------------------------
# data_stack
# ---------------------------------------------------------------------------

def test_data_stack_scalars_and_arrays():
    out = data_stack([1, 2, 3], B)
    assert isinstance(out, np.ndarray) and out.tolist() == [1, 2, 3]
    out = data_stack([True, False], B)
    assert out.dtype == B.default_boolean_dtype
    out = data_stack([0.5, 1.5], B)
    assert out.dtype == B.default_floating_dtype
    out = data_stack([np.float32(0.5), np.float32(1.5)], B)
    np.testing.assert_allclose(out, [0.5, 1.5])
    out = data_stack([np.zeros(2), np.ones(2)], B)
    assert out.shape == (2, 2)
    assert data_stack([], B) == []


def test_data_stack_list_of_infos_becomes_dict_of_stacked_values():
    infos = [
        {"step": 1, "success": True, "pos": np.zeros(2), "nested": {"x": 0.5}},
        {"step": 2, "success": False, "pos": np.ones(2), "nested": {"x": 1.5}},
    ]
    out = data_stack(infos, B)
    assert isinstance(out, dict)
    np.testing.assert_array_equal(out["step"], [1, 2])
    np.testing.assert_array_equal(out["success"], [True, False])
    assert out["pos"].shape == (2, 2)
    np.testing.assert_allclose(out["nested"]["x"], [0.5, 1.5])


def test_data_stack_strings_do_not_recurse():
    out = data_stack([{"name": "a", "step": 0}, {"name": "bb", "step": 1}], B)
    assert out["name"] == ["a", "bb"]
    assert data_stack(["x", "y"], B) == ["x", "y"]
    assert data_stack([None, None], B) == [None, None]


def test_data_stack_mismatched_info_keys_are_kept_per_env():
    infos = [{"a": 1}, {"b": 2}]
    out = data_stack(infos, B)
    assert out == [{"a": 1}, {"b": 2}]
