"""Spaces as JSON Schemas and their data as JSON: schemas validate the data, round trips are exact."""

import numpy as np
import pytest

jsonschema = pytest.importorskip("jsonschema")

from unienv_interface.backends.numpy import NumpyComputeBackend as B  # noqa: E402
from unienv_interface.space import (BatchedSpace, BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace,  # noqa: E402
                                    TextSpace, TupleSpace, UnionSpace)
from unienv_interface.space.space_utils import (batch_utils as sbu, bound_violations, from_json,  # noqa: E402
                                                space_to_json_schema, to_json)

ARM = BoxSpace(B, np.array([-1, -2, 0], np.float32), np.array([1, 2, 1], np.float32), np.float32, shape=(3,),
               metadata={"kind": "action", "names": ["shoulder", "elbow", "gripper"], "units": ["rad", "rad", None]})

SPACES = {
    "named box": ARM,
    "scalar int": BoxSpace(B, -3, 7, np.int64, shape=()),
    "matrix": BoxSpace(B, -1.0, 1.0, np.float64, shape=(2, 3)),
    "per-entry bounds": BoxSpace(B, np.array([0.0, -5.0]), np.array([1.0, 5.0]), np.float64, shape=(2,)),
    "binary": BinarySpace(B, shape=(2, 2)),
    "text": TextSpace(B, 6, min_length=1, charset="abc"),
    "dynamic": DynamicBoxSpace(B, 0.0, 1.0, shape_low=(1,), shape_high=(4,), dtype=np.float32),
    "dict": DictSpace(B, {"arm": ARM, "note": TextSpace(B, 4), "nested": DictSpace(B, {"k": BoxSpace(B, 0, 9, np.int32, shape=(2,))})}),
    "tuple": TupleSpace(B, [BoxSpace(B, 0.0, 1.0, np.float32, shape=(2,)), TextSpace(B, 3)]),
    "union": UnionSpace(B, [BoxSpace(B, 0.0, 1.0, np.float32, shape=()), TextSpace(B, 4)]),
    "batched named": sbu.batch_space(ARM, 2),
    "batched text": BatchedSpace(TextSpace(B, 4), batch_shape=(3,)),
}


@pytest.mark.parametrize("name", list(SPACES))
def test_schema_validates_data_and_round_trips(name):
    space = SPACES[name]
    schema = space_to_json_schema(space)
    jsonschema.Draft202012Validator.check_schema(schema)
    rng = B.random.random_number_generator(0)
    for _ in range(30):
        rng, sample = space.sample(rng)
        value = to_json(space, sample)
        jsonschema.validate(value, schema)
        restored = from_json(space, value)
        assert space.contains(restored), (name, value)
        assert to_json(space, restored) == value


def test_named_vectors_are_objects_with_units_and_bounds():
    schema = space_to_json_schema(ARM)
    assert schema["type"] == "object" and schema["required"] == ["shoulder", "elbow", "gripper"]
    assert schema["properties"]["elbow"] == {"type": "number", "minimum": -2.0, "maximum": 2.0, "description": "units: rad"}
    assert schema["description"] == "kind: action; units: rad, rad, -"
    assert to_json(ARM, np.array([0.5, -1, 0.25], np.float32)) == {"shoulder": 0.5, "elbow": -1.0, "gripper": 0.25}
    plain = space_to_json_schema(ARM, named=False)
    assert plain["type"] == "array" and plain["prefixItems"][1] == {"type": "number", "minimum": -2.0, "maximum": 2.0}
    assert to_json(ARM, np.zeros(3, np.float32), named=False) == [0.0, 0.0, 0.0]


def test_errors_name_the_path():
    space = SPACES["dict"]
    good = to_json(space, space.sample(B.random.random_number_generator(1))[1])
    with pytest.raises(ValueError, match=r"arm: missing \['gripper'\]"):
        from_json(space, {**good, "arm": {"shoulder": 0, "elbow": 0}})
    with pytest.raises(ValueError, match=r"nested/k: expected 2 entries, got 3"):
        from_json(space, {**good, "nested": {"k": [1, 2, 3]}})
    with pytest.raises(ValueError, match=r"nested/k: expected integers"):
        from_json(space, {**good, "nested": {"k": [1, 2.5]}})
    with pytest.raises(ValueError, match=r"unexpected \['extra'\]"):
        from_json(space, {**good, "extra": 1})
    with pytest.raises(ValueError, match="expected a string"):
        from_json(SPACES["text"], 3)


def test_bound_violations():
    action = from_json(ARM, {"shoulder": 1.5, "elbow": 0.0, "gripper": -0.1})
    assert bound_violations(ARM, action) == ["shoulder: 1.5 is above the maximum 1", "gripper: -0.1 is below the minimum 0"]
    assert bound_violations(ARM, from_json(ARM, {"shoulder": 0, "elbow": 0, "gripper": 0})) == []
    scalar = SPACES["scalar int"]
    assert bound_violations(scalar, from_json(scalar, 9)) == ["<value>: 9 is above the maximum 7"]
    assert bound_violations(SPACES["dict"], {"arm": action, "note": "ok", "nested": {"k": np.array([0, 3])}}) == [
        "arm/shoulder: 1.5 is above the maximum 1", "arm/gripper: -0.1 is below the minimum 0"]


def test_from_json_uses_the_space_backend_and_device():
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as T
    device = "cuda" if torch.cuda.is_available() else "cpu"
    space = BoxSpace(T, -1.0, 1.0, torch.float32, shape=(2, 2), device=device)
    value = from_json(space, [[0, 0.5], [1, -1]])
    assert isinstance(value, torch.Tensor) and value.dtype == torch.float32 and value.device.type == device
    assert to_json(space, value) == [[0.0, 0.5], [1.0, -1.0]]


def test_rounding_and_non_finite_values():
    space = BoxSpace(B, -np.inf, np.inf, np.float64, shape=(3,))
    assert to_json(space, np.array([1 / 3, np.inf, np.nan]), digits=3) == [0.333, None, None]
    assert "minimum" not in space_to_json_schema(space)["items"]
