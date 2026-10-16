"""Tests for nested-key handling in DictIncludeKeyTransformation / DictExcludeKeyTransformation."""

import itertools

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.transformations import (
    DictExcludeKeyTransformation, DictIncludeKeyTransformation,
    json_to_transformation, transformation_to_json,
)


def _box(v):
    return BoxSpace(B, float(v), float(v) + 1.0, np.float32, shape=(1,))


def _space():
    return DictSpace(B, {
        "a": {"x": _box(0), "y": _box(1), "z": _box(2), "n": {"p": _box(3), "q": _box(4)}},
        "b": _box(5),
    })


@pytest.mark.parametrize("keys,expected", [
    (["a/x", "a/y"], {"a": {"x", "y"}}),
    (["a/x", "a/n/p", "a/n/q"], {"a": {"x", "n"}, "a/n": {"p", "q"}}),
    (["a", "a/x"], {"a": {"x", "y", "z", "n"}}),
    (["a/x", "b"], {"": {"a", "b"}, "a": {"x"}}),
])
def test_include_sibling_nested_keys(keys, expected):
    space = _space()
    _, data = space.sample(B.random.random_number_generator(0))
    for order in itertools.permutations(keys):
        t = DictIncludeKeyTransformation(order)
        target = t.get_target_space_from_source(space)
        out = t.transform(space, data)
        assert target.contains(out)
        for path, keys_at_path in expected.items():
            node_space, node_data = target, out
            for part in filter(None, path.split("/")):
                node_space, node_data = node_space[part], node_data[part]
            assert set(node_space.keys()) == keys_at_path
            assert set(node_data.keys()) == keys_at_path
    # The source must not be modified.
    assert set(space["a"].keys()) == {"x", "y", "z", "n"}
    assert set(data["a"].keys()) == {"x", "y", "z", "n"}


def test_exclude_nested_keys():
    space = _space()
    _, data = space.sample(B.random.random_number_generator(0))
    t = DictExcludeKeyTransformation(["a/x", "a/n/p", "b"])
    target = t.get_target_space_from_source(space)
    out = t.transform(space, data)
    assert target.contains(out)
    assert set(out.keys()) == {"a"} and set(out["a"].keys()) == {"y", "z", "n"}
    assert set(out["a"]["n"].keys()) == {"q"}
    assert set(data["a"].keys()) == {"x", "y", "z", "n"}


@pytest.mark.parametrize("cls,arg", [
    (DictIncludeKeyTransformation, "enabled_keys"),
    (DictExcludeKeyTransformation, "excluded_keys"),
])
def test_missing_keys(cls, arg):
    space = _space()
    with pytest.raises(ValueError):
        cls(**{arg: ["missing"]}).get_target_space_from_source(space)
    with pytest.raises(ValueError):
        cls(**{arg: ["a/missing"]}).get_target_space_from_source(space)
    t = cls(**{arg: ["missing", "b"]}, ignore_missing_keys=True)
    target = t.get_target_space_from_source(space)
    assert "missing" not in target.keys()
    restored = json_to_transformation(transformation_to_json(t))
    assert isinstance(restored, cls) and restored.ignore_missing_keys
    assert restored.get_target_space_from_source(space) == target
