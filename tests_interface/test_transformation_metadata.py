"""Which transformations keep ``Space.metadata`` on their target spaces."""

from __future__ import annotations

import numpy as np
import pytest

from unienv_interface.backends import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace, TupleSpace
from unienv_interface.transformations import (
    BatchifyTransformation,
    ChainedTransformation,
    CropTransformation,
    DictExcludeKeyTransformation,
    DictIncludeKeyTransformation,
    DictTransformation,
    FlattenDictTransformation,
    IdentityTransformation,
    ImageResizeTransformation,
    IterativeTransformation,
    LeafwiseTransformation,
    RescaleTransformation,
    UnBatchifyTransformation,
    UnflattenDictTransformation,
)

IMAGE_MD = {"kind": "image", "format": "bgr", "layout": "HWC", "source": "front"}
STATE_MD = {"kind": "proprio", "units": "rad"}
OBS_MD = {"label": "observation"}


def image(md=IMAGE_MD) -> BoxSpace:
    return BoxSpace(B, 0, 255, np.uint8, shape=(8, 6, 3), metadata=md)


def state(md=STATE_MD) -> BoxSpace:
    return BoxSpace(B, -2.0, 2.0, np.float32, shape=(4,), metadata=md)


def obs() -> DictSpace:
    return DictSpace(B, {
        "cameras": DictSpace(B, {"front": image(), "wrist": image({"kind": "image", "source": "wrist"})}, metadata={"label": "cams"}),
        "state": state(),
    }, metadata=OBS_MD)


@pytest.mark.parametrize("transformation", [
    RescaleTransformation(-1.0, 1.0),
    CropTransformation(-1.0, 1.0),
    IdentityTransformation(),
], ids=["rescale", "crop", "identity"])
def test_box_transformations_keep_metadata(transformation):
    target = transformation.get_target_space_from_source(state())
    assert target.metadata == STATE_MD


def test_image_resize_keeps_metadata_with_new_shape():
    target = ImageResizeTransformation(4, 3).get_target_space_from_source(image())
    assert target.shape == (4, 3, 3)
    assert target.metadata == IMAGE_MD


@pytest.mark.parametrize("axis", [0, 1])
def test_batchify_and_unbatchify_keep_metadata(axis):
    source = BoxSpace(B, 0, 255, np.uint8, shape=(2, 8, 6, 3), metadata=IMAGE_MD)
    batched = BatchifyTransformation(axis=axis).get_target_space_from_source(source)
    assert batched.metadata == IMAGE_MD
    unbatched = UnBatchifyTransformation(axis=axis).get_target_space_from_source(batched)
    assert unbatched == source and unbatched.metadata == IMAGE_MD
    batched_dict = BatchifyTransformation().get_target_space_from_source(obs())
    assert batched_dict.metadata == OBS_MD
    assert batched_dict["cameras"]["front"].metadata == IMAGE_MD


def test_dict_transformation_keeps_container_and_leaf_metadata():
    target = DictTransformation({
        "state": RescaleTransformation(-1.0, 1.0),
        "cameras/front": ImageResizeTransformation(4, 3),
    }).get_target_space_from_source(obs())
    assert target.metadata == OBS_MD
    assert target["state"].metadata == STATE_MD
    assert target["cameras"].metadata == {"label": "cams"}
    assert target["cameras"]["front"].metadata == IMAGE_MD
    assert target["cameras"]["front"].shape == (4, 3, 3)


def test_include_and_exclude_keys_keep_metadata():
    included = DictIncludeKeyTransformation(["cameras/front", "state"]).get_target_space_from_source(obs())
    assert included.metadata == OBS_MD
    assert included["cameras"].metadata == {"label": "cams"}
    assert list(included["cameras"].keys()) == ["front"]
    assert included["cameras"]["front"].metadata == IMAGE_MD
    assert included["state"].metadata == STATE_MD

    excluded = DictExcludeKeyTransformation(["cameras/wrist"]).get_target_space_from_source(obs())
    assert excluded.metadata == OBS_MD
    assert excluded["cameras"].metadata == {"label": "cams"}
    assert list(excluded["cameras"].keys()) == ["front"]
    assert excluded["cameras"]["front"].metadata == IMAGE_MD


def test_flatten_and_unflatten_dict_keep_leaf_metadata():
    flat = FlattenDictTransformation().get_target_space_from_source(obs())
    assert flat.metadata == OBS_MD
    assert flat["cameras/front"].metadata == IMAGE_MD
    assert flat["state"].metadata == STATE_MD
    nested = UnflattenDictTransformation().get_target_space_from_source(flat)
    assert nested.metadata == OBS_MD
    assert nested["cameras"]["wrist"].metadata == {"kind": "image", "source": "wrist"}
    # The flattened-away inner dict metadata is not recoverable
    assert nested["cameras"].metadata == {}


def test_chained_transformation_keeps_metadata():
    chained = ChainedTransformation([
        DictExcludeKeyTransformation(["cameras/wrist"]),
        DictTransformation({"state": RescaleTransformation(-1.0, 1.0)}),
        FlattenDictTransformation(),
        BatchifyTransformation(),
    ])
    target = chained.get_target_space_from_source(obs())
    assert target.metadata == OBS_MD
    assert target["cameras/front"].metadata == IMAGE_MD
    assert target["state"].metadata == STATE_MD


def test_iterative_and_leafwise_transformations_keep_metadata():
    source = DictSpace(B, {
        "a": state(),
        "t": TupleSpace(B, [state({"kind": "action"})], metadata={"label": "tuple"}),
    }, metadata=OBS_MD)
    target = IterativeTransformation(RescaleTransformation(-1.0, 1.0)).get_target_space_from_source(source)
    assert target.metadata == OBS_MD
    assert target["a"].metadata == STATE_MD
    assert target["t"].metadata == {"label": "tuple"}
    assert target["t"][0].metadata == {"kind": "action"}

    leafwise = LeafwiseTransformation({
        "a": RescaleTransformation(-1.0, 1.0),
        "t": (CropTransformation(-1.0, 1.0),),
    }).get_target_space_from_source(source)
    assert leafwise.metadata == OBS_MD
    assert leafwise["a"].metadata == STATE_MD
    assert leafwise["t"].metadata == {"label": "tuple"}
    assert leafwise["t"][0].metadata == {"kind": "action"}
