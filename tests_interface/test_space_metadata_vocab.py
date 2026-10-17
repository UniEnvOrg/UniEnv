"""Metadata vocabulary helpers: the builders and ``annotate``."""

from __future__ import annotations

import numpy as np
import pytest

from unienv_interface.backends import NumpyComputeBackend as B
from unienv_interface.space.spaces import BinarySpace, BoxSpace, DictSpace, TupleSpace
from unienv_interface.space.space_utils import metadata_utils as smu
from unienv_interface.space.space_utils.metadata_utils import (
    action_metadata, annotate, depth_metadata, image_metadata, metadata, proprio_metadata,
)


def box(shape=(2,), metadata=None) -> BoxSpace:
    return BoxSpace(B, -1.0, 1.0, np.float32, shape=shape, metadata=metadata)


def image(shape=(480, 640, 3), metadata=None) -> BoxSpace:
    return BoxSpace(B, 0, 255, np.uint8, shape=shape, metadata=metadata)


def obs_space() -> DictSpace:
    return DictSpace(B, {
        "cameras": DictSpace(B, {
            "front": image(),
            "wrist": image((240, 320, 3), metadata={"source": "wrist_cam"}),
        }, metadata={"label": "cameras"}),
        "state": TupleSpace(B, [box((7,)), box((3,), metadata={"units": "m"})]),
        "flag": BinarySpace(B, shape=(1,)),
    }, metadata={"label": "observation"})


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def test_builders_are_exported_from_space_utils():
    from unienv_interface.space import space_utils
    for name in ("metadata", "image_metadata", "depth_metadata", "proprio_metadata", "action_metadata", "annotate"):
        assert getattr(space_utils, name) is getattr(smu, name)


def test_metadata_builder_drops_none_values():
    assert metadata("force", units="N", frame=None, names=["fx", "fy", "fz"]) == {
        "kind": "force", "units": "N", "names": ["fx", "fy", "fz"],
    }
    assert metadata(None, label="x") == {"label": "x"}


def test_image_metadata_defaults_and_overrides():
    assert image_metadata() == {"kind": "image", "format": "rgb", "layout": "HWC"}
    assert image_metadata(format="bgr", layout="CHW", source="front_cam", fps=30, label="front") == {
        "kind": "image", "format": "bgr", "layout": "CHW", "source": "front_cam", "fps": 30, "label": "front",
    }
    assert image_metadata(format=None, layout=None) == {"kind": "image"}


def test_depth_metadata_defaults_and_overrides():
    assert depth_metadata() == {"kind": "depth", "units": "m", "layout": "HW"}
    assert depth_metadata(units="mm", source="wrist", fps=15.0) == {
        "kind": "depth", "units": "mm", "layout": "HW", "source": "wrist", "fps": 15.0,
    }


def test_proprio_metadata():
    assert proprio_metadata() == {"kind": "proprio"}
    names = ["x", "y", "z", "qw", "qx", "qy", "qz"]
    assert proprio_metadata(units=["m"] * 3 + ["normalized"] * 4, names=names, frame="base", rotation="quat_wxyz") == {
        "kind": "proprio", "units": ["m", "m", "m", "normalized", "normalized", "normalized", "normalized"],
        "names": names, "frame": "base", "rotation": "quat_wxyz",
    }


def test_action_metadata():
    assert action_metadata() == {"kind": "action"}
    assert action_metadata(
        control="joint_position", mode="delta", units="rad", normalized=True, range=[-0.1, 0.1],
        names=["j0", "j1"], frame="base", rotation="rot6d", label="arm",
    ) == {
        "kind": "action", "control": "joint_position", "mode": "delta", "units": "rad", "normalized": True,
        "range": [-0.1, 0.1], "names": ["j0", "j1"], "frame": "base", "rotation": "rot6d", "label": "arm",
    }
    # False is a value, not "absent"
    assert action_metadata(normalized=False) == {"kind": "action", "normalized": False}


def test_builder_output_is_accepted_by_spaces():
    space = box((4, 4, 3), metadata=image_metadata(source="cam"))
    assert space.metadata == {"kind": "image", "format": "rgb", "layout": "HWC", "source": "cam"}


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------

def test_annotate_merges_metadata_at_string_paths():
    space = obs_space()
    annotated = annotate(space, {
        "cameras/front": image_metadata(source="front_cam"),
        "cameras/wrist": image_metadata(),
        "state/0": proprio_metadata(units="rad"),
    })
    assert annotated["cameras"]["front"].metadata == {"kind": "image", "format": "rgb", "layout": "HWC", "source": "front_cam"}
    # merged into the existing metadata
    assert annotated["cameras"]["wrist"].metadata == {"kind": "image", "format": "rgb", "layout": "HWC", "source": "wrist_cam"}
    assert annotated["state"].spaces[0].metadata == {"kind": "proprio", "units": "rad"}
    # the containers on the paths keep their own metadata
    assert annotated.metadata == {"label": "observation"}
    assert annotated["cameras"].metadata == {"label": "cameras"}
    # the data domain is unchanged
    assert annotated == space


def test_annotate_tuple_paths_and_root():
    space = obs_space()
    annotated = annotate(space, {
        ("state", 1): {"frame": "world"},
        ("cameras",): {"source": "rig"},
        (): {"kind": "observation"},
        "": {"fps": 10},
    })
    assert annotated["state"].spaces[1].metadata == {"units": "m", "frame": "world"}
    assert annotated["cameras"].metadata == {"label": "cameras", "source": "rig"}
    assert annotated.metadata == {"label": "observation", "kind": "observation", "fps": 10}
    # a leaf space can be annotated at its root
    assert annotate(box(), {"": {"units": "N"}}).metadata == {"units": "N"}
    assert annotate(box(), {(): {"units": "N"}}).metadata == {"units": "N"}


def test_annotate_negative_tuple_index():
    annotated = annotate(obs_space(), {("state", -1): {"frame": "world"}, "state/-2": {"frame": "base"}})
    assert annotated["state"].spaces[1].metadata == {"units": "m", "frame": "world"}
    assert annotated["state"].spaces[0].metadata == {"frame": "base"}


def test_annotate_replace():
    space = obs_space()
    annotated = annotate(space, {"cameras/wrist": {"kind": "image"}, "": {"kind": "observation"}}, replace=True)
    assert annotated["cameras"]["wrist"].metadata == {"kind": "image"}
    assert annotated.metadata == {"kind": "observation"}
    # replace only applies to the annotated spaces
    assert annotated["cameras"].metadata == {"label": "cameras"}
    assert annotate(space, {"cameras/wrist": {}}, replace=True)["cameras"]["wrist"].metadata == {}


def test_annotate_leaves_input_and_untouched_spaces_intact():
    space = obs_space()
    annotated = annotate(space, {"cameras/front": image_metadata()})
    # the input is unchanged
    assert space["cameras"]["front"].metadata == {}
    assert space.metadata == {"label": "observation"}
    # spaces off the annotated path are the same objects
    assert annotated["cameras"]["wrist"] is space["cameras"]["wrist"]
    assert annotated["state"] is space["state"]
    assert annotated["flag"] is space["flag"]
    assert list(annotated.spaces) == list(space.spaces)
    # an empty mapping returns the space itself
    assert annotate(space, {}) is space


def test_annotate_strict_reports_all_missing_paths():
    space = obs_space()
    with pytest.raises(KeyError) as info:
        annotate(space, {
            "cameras/front": image_metadata(),
            "cameras/left": image_metadata(),
            ("state", 5): {},
            "flag/0": {},          # below a leaf
            "state/x": {},         # not a tuple position
        })
    message = str(info.value)
    for path in ("'cameras/left'", "('state', 5)", "'flag/0'", "'state/x'"):
        assert path in message
    assert "cameras/front" not in message


def test_annotate_non_strict_skips_missing_paths():
    space = obs_space()
    annotated = annotate(space, {"cameras/front": image_metadata(), "missing": {"kind": "x"}}, strict=False)
    assert annotated["cameras"]["front"].metadata["kind"] == "image"
    assert "missing" not in annotated.spaces


def test_annotate_rejects_bad_input():
    with pytest.raises(TypeError):
        annotate(obs_space(), {"flag": "image"})
    with pytest.raises(TypeError):
        annotate(obs_space(), {"flag": {"bad": object()}})
    with pytest.raises(TypeError):
        annotate(obs_space(), {5: {}})
