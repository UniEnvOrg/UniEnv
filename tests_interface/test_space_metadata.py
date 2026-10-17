"""``Space.metadata``: storage, validation, equality, serialization and batch-utility propagation."""

from __future__ import annotations

import copy
import json
import pickle
from typing import Callable, Dict

import numpy as np
import pytest

from unienv_interface.backends import NumpyComputeBackend as B
from unienv_interface.space import Space, SpaceMetadata
from unienv_interface.space.spaces import (
    BatchedSpace, BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, GraphSpace,
    TextSpace, TupleSpace, UnionSpace,
)
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.space.space_utils.metadata_utils import (
    child_spaces, find_spaces_by_metadata, spaces_equal_with_metadata,
)
from unienv_interface.space.space_utils.serialization_utils import json_to_space, space_to_json

IMAGE_MD = {"kind": "image", "format": "rgb", "layout": "HWC", "source": "front", "fps": 30}


def box(shape=(2,), metadata=None, low=-1.0, high=1.0, dtype=np.float32) -> BoxSpace:
    return BoxSpace(B, low, high, dtype, shape=shape, metadata=metadata)


SPACE_FACTORIES: Dict[str, Callable[..., Space]] = {
    "box": lambda md: box((2, 3), md),
    "box-array-bounds": lambda md: BoxSpace(B, np.zeros((2, 3)), np.arange(1, 7).reshape(2, 3), np.float64, metadata=md),
    "binary": lambda md: BinarySpace(B, shape=(2, 3), metadata=md),
    "dynamic-box": lambda md: DynamicBoxSpace(B, -1.0, 1.0, (2, 0), (2, 5), np.float32, metadata=md),
    "dict": lambda md: DictSpace(B, {"a": box(), "b": BinarySpace(B, shape=(2,))}, metadata=md),
    "tuple": lambda md: TupleSpace(B, [box(), BinarySpace(B, shape=(2,))], metadata=md),
    "text": lambda md: TextSpace(B, 5, charset="abc", metadata=md),
    "union": lambda md: UnionSpace(B, [box(), BinarySpace(B, shape=(2,))], metadata=md),
    "graph": lambda md: GraphSpace(B, box((3,)), box((1,)), is_edge=True, max_nodes=4, batch_shape=(2,), metadata=md),
    "batched": lambda md: BatchedSpace(TextSpace(B, 5), batch_shape=(2,), metadata=md),
}


@pytest.fixture(params=sorted(SPACE_FACTORIES))
def factory(request) -> Callable[..., Space]:
    return SPACE_FACTORIES[request.param]


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

def test_every_space_accepts_metadata(factory):
    plain = factory(None)
    assert plain.metadata == {}
    assert len(plain.metadata) == 0 and not plain.metadata
    annotated = factory(IMAGE_MD)
    assert isinstance(annotated.metadata, SpaceMetadata)
    assert annotated.metadata == IMAGE_MD
    assert dict(annotated.metadata) == IMAGE_MD


def test_metadata_is_ignored_by_equality_and_subspace_checks(factory):
    plain, annotated = factory(None), factory(IMAGE_MD)
    other = factory({"kind": "depth"})
    assert plain == annotated and annotated == other
    assert annotated.is_subspaceeq(plain) and plain.is_subspaceeq(annotated)
    assert not annotated.is_subspace(plain)
    assert not spaces_equal_with_metadata(plain, annotated)
    assert not spaces_equal_with_metadata(annotated, other)
    assert spaces_equal_with_metadata(annotated, factory(dict(IMAGE_MD)))


def test_metadata_is_read_only_and_copied():
    source = {"kind": "image", "intrinsics": {"fx": 1.0, "size": [4, 3]}, "tags": ["a", "b"]}
    space = box(metadata=source)
    source["kind"] = "depth"
    source["intrinsics"]["fx"] = 2.0
    source["tags"].append("c")
    assert space.metadata["kind"] == "image"
    assert space.metadata["intrinsics"]["fx"] == 1.0
    assert space.metadata["tags"] == ("a", "b")
    assert space.metadata == {"kind": "image", "intrinsics": {"fx": 1.0, "size": [4, 3]}, "tags": ["a", "b"]}
    with pytest.raises(TypeError):
        space.metadata["kind"] = "depth"
    with pytest.raises(TypeError):
        space.metadata["intrinsics"]["fx"] = 3.0
    with pytest.raises(AttributeError):
        space.metadata._data = {}
    # A plain JSON copy can be edited freely
    as_dict = space.metadata.to_dict()
    assert as_dict == {"kind": "image", "intrinsics": {"fx": 1.0, "size": [4, 3]}, "tags": ["a", "b"]}
    as_dict["tags"].append("c")
    assert space.metadata["tags"] == ("a", "b")


@pytest.mark.parametrize("bad", [
    {"kind": object()},
    {"kind": {1, 2}},
    {"array": np.zeros(3)},
    {1: "integer key"},
    {"nested": {2: "integer key"}},
    {"bytes": b"raw"},
])
def test_non_json_metadata_is_rejected(bad):
    with pytest.raises(TypeError, match="metadata"):
        box(metadata=bad)


def test_metadata_must_be_a_mapping():
    with pytest.raises(TypeError, match="mapping"):
        box(metadata=[("kind", "image")])


def test_numpy_scalars_become_python_scalars():
    space = box(metadata={"fps": np.int64(30), "scale": np.float32(0.5), "flag": np.bool_(True)})
    assert type(space.metadata["fps"]) is int and space.metadata["fps"] == 30
    assert type(space.metadata["scale"]) is float
    assert space.metadata["flag"] is True
    json.dumps(space.metadata.to_dict())


def test_metadata_mapping_is_hashable_and_compares_with_plain_mappings():
    a = SpaceMetadata({"kind": "image", "size": [1, 2]})
    assert a == SpaceMetadata({"size": (1, 2), "kind": "image"})
    assert hash(a) == hash(SpaceMetadata({"size": (1, 2), "kind": "image"}))
    assert a == {"kind": "image", "size": [1, 2]}
    assert {"kind": "image", "size": [1, 2]} == a
    assert a != {"kind": "image"}
    assert a != {"kind": object()}
    assert a != 3
    assert "kind" in a and a.get("missing", 7) == 7
    assert "SpaceMetadata" in repr(a)


# ---------------------------------------------------------------------------
# with_metadata
# ---------------------------------------------------------------------------

def test_with_metadata_returns_a_new_space(factory):
    original = factory({"kind": "image", "fps": 30})
    updated = original.with_metadata({"fps": 15}, source="wrist")
    assert updated is not original
    assert type(updated) is type(original)
    assert updated == original
    assert updated.metadata == {"kind": "image", "fps": 15, "source": "wrist"}
    assert original.metadata == {"kind": "image", "fps": 30}
    assert original.with_metadata(replace=True).metadata == {}
    assert original.with_metadata({"label": "x"}, replace=True).metadata == {"label": "x"}
    # The copy is a working space
    _, sample = updated.sample(np.random.default_rng(0))
    if not isinstance(updated, GraphSpace):
        assert updated.contains(sample)


def test_with_metadata_on_dict_space_does_not_share_children_dict():
    original = DictSpace(B, {"a": box()})
    updated = original.with_metadata(kind="observation")
    updated["b"] = box()
    assert set(original.keys()) == {"a"}
    assert set(updated.keys()) == {"a", "b"}
    assert updated.spaces["a"] is original.spaces["a"]


def test_with_metadata_on_box_shares_bounds_and_keeps_fast_paths():
    original = box((3,), {"kind": "proprio"})
    updated = original.with_metadata(units="m")
    assert updated._low is original._low
    rng = np.random.default_rng(0)
    for space in (original, updated):
        rng, sample = space.sample(rng)
        assert space.contains(sample)


def test_spaces_without_stored_metadata_report_empty_metadata():
    """Spaces pickled before metadata existed have no ``_metadata`` attribute."""
    space = box()
    assert "_metadata" not in space.__dict__
    assert space.metadata == {}
    legacy = DictSpace(B, {"a": box()})
    legacy.__dict__.pop("_metadata", None)
    restored = pickle.loads(pickle.dumps(legacy))
    assert restored.metadata == {}


def test_reinitialising_in_place_resets_metadata():
    space = BinarySpace(B, shape=(2,), metadata={"kind": "image"})
    space.__init__(B, shape=(3,))
    assert space.metadata == {}
    space.__init__(B, shape=(3,), metadata={"kind": "mask"})
    assert space.metadata == {"kind": "mask"}


# ---------------------------------------------------------------------------
# Copying, pickling, backend conversion
# ---------------------------------------------------------------------------

def test_pickle_and_copy_keep_metadata(factory):
    space = factory(IMAGE_MD)
    for clone in (pickle.loads(pickle.dumps(space)), copy.deepcopy(space), copy.copy(space)):
        assert clone.metadata == IMAGE_MD
        assert spaces_equal_with_metadata(clone, space)


def test_to_keeps_metadata_recursively():
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    space = DictSpace(B, {
        "image": box((4, 4, 3), IMAGE_MD, 0, 255, np.uint8),
        "text": TextSpace(B, 3, metadata={"kind": "instruction"}),
        "nested": TupleSpace(B, [BinarySpace(B, shape=(1,), metadata={"kind": "flag"})], metadata={"label": "t"}),
        "dyn": DynamicBoxSpace(B, 0.0, 1.0, (0,), (3,), np.float32, metadata={"kind": "points"}),
    }, metadata={"kind": "observation"})
    moved = space.to(TB, "cpu")
    assert moved.backend is TB
    assert moved.metadata == {"kind": "observation"}
    assert moved["image"].metadata == IMAGE_MD
    assert moved["text"].metadata == {"kind": "instruction"}
    assert moved["nested"].metadata == {"label": "t"}
    assert moved["nested"][0].metadata == {"kind": "flag"}
    assert moved["dyn"].metadata == {"kind": "points"}
    back = moved.to(B)
    assert spaces_equal_with_metadata(back, space)


def test_to_keeps_metadata_for_every_space(factory):
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    space = factory(IMAGE_MD)
    moved = space.to(TB, "cpu")
    assert moved.metadata == IMAGE_MD
    for (_, child), (_, moved_child) in zip(child_spaces(space), child_spaces(moved)):
        assert moved_child.metadata == child.metadata


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _annotate(space: Space, prefix: str = "root") -> Space:
    """Give ``space`` and every nested space a distinct metadata entry."""
    if isinstance(space, DictSpace):
        space = DictSpace(space.backend, {k: _annotate(v, f"{prefix}.{k}") for k, v in space.spaces.items()}, device=space.device)
    elif isinstance(space, TupleSpace):
        space = TupleSpace(space.backend, [_annotate(v, f"{prefix}.{i}") for i, v in enumerate(space.spaces)], device=space.device)
    elif isinstance(space, UnionSpace):
        space = UnionSpace(space.backend, [_annotate(v, f"{prefix}.{i}") for i, v in enumerate(space.spaces)], device=space.device)
    elif isinstance(space, BatchedSpace):
        space = BatchedSpace(_annotate(space.single_space, f"{prefix}.single"), space.batch_shape)
    elif isinstance(space, GraphSpace):
        space = GraphSpace(
            space.backend,
            _annotate(space.node_feature_space, f"{prefix}.nodes"),
            None if space.edge_feature_space is None else _annotate(space.edge_feature_space, f"{prefix}.edges"),
            is_edge=space.is_edge, min_nodes=space.min_nodes, max_nodes=space.max_nodes,
            min_edges=space.min_edges, max_edges=space.max_edges, batch_shape=space.batch_shape,
        )
    return space.with_metadata(label=prefix, extra={"path": prefix.split(".")})


def test_json_round_trip_keeps_nested_metadata(factory):
    space = _annotate(factory(None))
    payload = json.loads(json.dumps(space_to_json(space)))
    assert payload["metadata"] == {"label": "root", "extra": {"path": ["root"]}}
    restored = json_to_space(payload, B, None)
    assert spaces_equal_with_metadata(restored, space)


def test_json_omits_empty_metadata_and_loads_old_payloads(factory):
    space = factory(None)
    payload = space_to_json(space)
    assert "metadata" not in json.dumps(payload)
    restored = json_to_space(payload, B, None)
    assert restored == space and restored.metadata == {}


def test_json_from_before_metadata_loads():
    old_payload = {
        "type": "DictSpace",
        "spaces": {
            "image": {"type": "BoxSpace", "shape": [2, 2, 3], "low": [[[0]]], "high": [[[255]]], "dtype": "uint8"},
            "flag": {"type": "BinarySpace", "shape": [1], "dtype": "bool"},
        },
    }
    restored = json_to_space(old_payload, B, None)
    assert restored.metadata == {} and restored["image"].metadata == {}


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------

BATCHABLE = ["box", "box-array-bounds", "binary", "dynamic-box", "dict", "tuple", "graph", "batched"]


@pytest.mark.parametrize("name", BATCHABLE + ["text", "union"])
def test_batch_space_keeps_metadata(name):
    space = _annotate(SPACE_FACTORIES[name](None))
    batched = sbu.batch_space(space, 3)
    assert spaces_equal_with_metadata(batched, sbu.batch_space(space, 3))
    assert batched.metadata == space.metadata
    if isinstance(space, (TextSpace, UnionSpace)):
        assert batched.single_space is space
    else:
        for (_, child), (_, batched_child) in zip(child_spaces(space), child_spaces(batched)):
            assert batched_child.metadata == child.metadata


@pytest.mark.parametrize("name", BATCHABLE)
def test_unbatch_spaces_keeps_metadata(name):
    space = _annotate(SPACE_FACTORIES[name](None))
    batched = sbu.batch_space(space, 3)
    unbatched = list(sbu.unbatch_spaces(batched))
    assert len(unbatched) == 3
    for item in unbatched:
        assert spaces_equal_with_metadata(item, space)


@pytest.mark.parametrize("name", BATCHABLE)
def test_reshape_and_swap_batch_dims_keep_metadata(name):
    space = _annotate(SPACE_FACTORIES[name](None))
    batched = sbu.batch_space(sbu.batch_space(space, 2), 3)  # (3, 2, ...)
    reshaped = sbu.reshape_batch_size(batched, (3, 2), (6,))
    assert spaces_equal_with_metadata(reshaped, sbu.batch_space(space, 6))
    swapped = sbu.swap_batch_dims(batched, 0, 1)
    assert spaces_equal_with_metadata(swapped, sbu.batch_space(sbu.batch_space(space, 3), 2))


@pytest.mark.parametrize("name", ["box", "binary", "dynamic-box", "dict", "tuple"])
def test_batch_differing_spaces_keeps_shared_metadata_only(name):
    factory = SPACE_FACTORIES[name]
    same = sbu.batch_differing_spaces([factory(IMAGE_MD), factory(IMAGE_MD)])
    assert same.metadata == IMAGE_MD
    differing = sbu.batch_differing_spaces([factory(IMAGE_MD), factory({"kind": "depth"})])
    assert differing.metadata == {}


def test_batch_differing_spaces_recurses_per_key():
    a = DictSpace(B, {"img": box(metadata=IMAGE_MD), "q": box(metadata={"kind": "proprio"})})
    b = DictSpace(B, {"img": box(metadata=IMAGE_MD), "q": box(metadata={"kind": "proprio", "units": "rad"})})
    batched = sbu.batch_differing_spaces([a, b])
    assert batched["img"].metadata == IMAGE_MD
    assert batched["q"].metadata == {}


def test_flatten_space_drops_metadata():
    space = DictSpace(B, {"img": box((2, 2, 3), IMAGE_MD)}, metadata={"kind": "observation"})
    assert sfu.flatten_space(space).metadata == {}
    assert sfu.flatten_space(space["img"]).metadata == {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_find_spaces_by_metadata():
    space = DictSpace(B, {
        "cameras": DictSpace(B, {
            "front": box((4, 4, 3), IMAGE_MD, 0, 255, np.uint8),
            "depth": box((4, 4), {"kind": "depth", "units": "m"}),
        }, metadata={"kind": "image"}),  # containers are not leaves
        "wrist": TupleSpace(B, [box((4, 4, 3), {"kind": "image", "layout": "HWC"}, 0, 255, np.uint8), box()]),
        "state": box((7,), {"kind": "proprio"}),
    })
    images = find_spaces_by_metadata(space, kind="image")
    assert [path for path, _ in images] == [("cameras", "front"), ("wrist", 0)]
    assert images[0][1] is space["cameras"]["front"]
    assert [p for p, _ in find_spaces_by_metadata(space, kind="image", source="front")] == [("cameras", "front")]
    assert [p for p, _ in find_spaces_by_metadata(space, lambda s: s.shape == (7,))] == [("state",)]
    assert len(find_spaces_by_metadata(space)) == 5
    leaf = box(metadata={"kind": "action"})
    assert find_spaces_by_metadata(leaf, kind="action") == [((), leaf)]
    assert find_spaces_by_metadata(leaf, kind="image") == []


def test_spaces_equal_with_metadata_checks_nested_spaces():
    a = DictSpace(B, {"x": box(metadata={"kind": "proprio"})})
    b = DictSpace(B, {"x": box(metadata={"kind": "action"})})
    assert a == b
    assert not spaces_equal_with_metadata(a, b)
    assert spaces_equal_with_metadata(a, DictSpace(B, {"x": box(metadata={"kind": "proprio"})}))
    assert not spaces_equal_with_metadata(a, box())
