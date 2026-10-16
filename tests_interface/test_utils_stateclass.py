"""Tests for ``unienv_interface.utils.stateclass``."""

import dataclasses

import pytest

from unienv_interface.utils.stateclass import StateClass, field, stateclass


class Point(StateClass):
    x: int
    y: int = 0
    tag: str = field(pytree_node=False, default="p")


def test_stateclass_subclass_is_frozen_dataclass_with_replace():
    p = Point(1)
    assert dataclasses.is_dataclass(p)
    assert (p.x, p.y, p.tag) == (1, 0, "p")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.x = 3
    q = p.replace(y=5)
    assert (q.x, q.y) == (1, 5) and (p.x, p.y) == (1, 0)
    assert q == Point(1, 5)


def test_field_metadata():
    fields = {f.name: f for f in dataclasses.fields(Point)}
    assert fields["tag"].metadata["pytree_node"] is False
    assert fields["x"].metadata.get("pytree_node", True) is True
    assert field(metadata={"extra": 1}).metadata == {"extra": 1, "pytree_node": True}


def test_stateclass_decorator_options_and_idempotence():
    @stateclass(frozen=False)
    class Mutable:
        value: int

    m = Mutable(1)
    m.value = 2
    assert m.replace(value=3).value == 3
    assert stateclass(Mutable) is Mutable

    @stateclass
    class Frozen:
        value: int

    with pytest.raises(dataclasses.FrozenInstanceError):
        Frozen(1).value = 2
