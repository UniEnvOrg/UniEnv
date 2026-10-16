"""Tests for ``unienv_interface.utils.dict_util``."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.utils.dict_util import flatten_keys_in_mapping, nested_get, unflatten_keys_in_mapping


def _box(n=1):
    return BoxSpace(B, -1.0, 1.0, np.float32, shape=(n,))


def test_flatten_unflatten_round_trip_for_dicts():
    nested = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    flat = flatten_keys_in_mapping(nested)
    assert flat == {"a/b": 1, "a/c/d": 2, "e": 3}
    assert unflatten_keys_in_mapping(flat) == nested
    flat_dot = flatten_keys_in_mapping(nested, nested_separator=".")
    assert flat_dot == {"a.b": 1, "a.c.d": 2, "e": 3}
    assert unflatten_keys_in_mapping(flat_dot, nested_separator=".") == nested


def test_flatten_unflatten_round_trip_for_dict_space():
    space = DictSpace(B, {"robot": DictSpace(B, {"qpos": _box(3), "qvel": _box(3)}), "goal": _box(2)})
    flat = flatten_keys_in_mapping(space)
    assert isinstance(flat, DictSpace)
    assert set(flat.keys()) == {"robot/qpos", "robot/qvel", "goal"}
    assert flat["robot/qpos"].shape == (3,)
    back = unflatten_keys_in_mapping(flat)
    assert isinstance(back, DictSpace) and isinstance(back["robot"], DictSpace)
    assert set(back["robot"].keys()) == {"qpos", "qvel"}
    assert back["goal"] == space["goal"]
    # samples of the flattened space live in the flattened key structure
    _, sample = flat.sample(np.random.default_rng(0))
    assert flat.contains(sample)
    assert space.contains(unflatten_keys_in_mapping(sample))


def test_unflatten_keeps_non_nested_keys():
    assert unflatten_keys_in_mapping({"x": 1, "y/z": 2}) == {"x": 1, "y": {"z": 2}}


def test_nested_get():
    data = {"a": {"b": {"c": 5}}}
    assert nested_get(data, ["a", "b", "c"]) == 5
    assert nested_get(data, []) is data
    with pytest.raises(KeyError):
        nested_get(data, ["a", "x"])
