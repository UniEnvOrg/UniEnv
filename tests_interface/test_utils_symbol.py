"""Tests for ``unienv_interface.utils.symbol_util``."""

import json

import numpy as np
import pytest

from unienv_interface.utils.symbol_util import (
    deserialize_function,
    get_class_from_full_name,
    get_full_class_name,
    get_full_function_name,
    get_function_from_full_name,
    serialize_function,
)
from unienv_interface.space import BoxSpace
from tests_common.envs import StubEnv


def module_level_function(x):
    return x + 1


class Outer:
    class Inner:
        @staticmethod
        def static_fn(x):
            return x * 2

    @staticmethod
    def static_fn(x):
        return x * 3


def test_class_name_round_trip():
    assert get_full_class_name(BoxSpace) == "unienv_interface.space.spaces.box.BoxSpace"
    assert get_class_from_full_name(get_full_class_name(BoxSpace)) is BoxSpace


def test_nested_class_name_round_trip():
    name = get_full_class_name(Outer.Inner)
    assert name.endswith("Outer.Inner")
    assert get_class_from_full_name(name) is Outer.Inner


def test_class_remap_for_moved_classes():
    pytest.importorskip("unienv_data")
    cls = get_class_from_full_name("unienv_data.storages.common.FlattenedStorage")
    assert cls.__name__ == "FlattenedStorage"


def test_module_level_function_serialized_by_name():
    data = serialize_function(module_level_function)
    assert data["mode"] == "name"
    assert deserialize_function(json.loads(json.dumps(data))) is module_level_function


@pytest.mark.parametrize("fn", [Outer.static_fn, Outer.Inner.static_fn, StubEnv.render, np.linalg.norm])
def test_nested_qualnames_serialized_by_name(fn):
    full_name = get_full_function_name(fn)
    assert get_function_from_full_name(full_name) is fn
    data = serialize_function(fn)
    assert data == {"mode": "name", "value": full_name}
    assert deserialize_function(data) is fn


def test_lambda_and_local_functions_fall_back_to_pickle():
    offset = 5

    def local(x):
        return x + offset

    for fn in (lambda x: x - 1, local):
        data = serialize_function(fn)
        assert data["mode"] == "pickle"
        restored = deserialize_function(json.loads(json.dumps(data)))
        assert restored(10) == fn(10)


def test_bound_methods_are_not_serialized_by_name():
    env = StubEnv()
    env.position[:] = 7.0
    data = serialize_function(env.render)
    assert data["mode"] == "pickle"
    restored = deserialize_function(data)
    np.testing.assert_array_equal(restored(), env.render())


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        deserialize_function({"mode": "bogus", "value": "x"})


def test_missing_symbol_raises():
    with pytest.raises((ImportError, AttributeError)):
        get_function_from_full_name("unienv_interface.utils.symbol_util.does_not_exist")
    with pytest.raises((ImportError, AttributeError)):
        get_class_from_full_name("no_such_module_xyz.Thing")
