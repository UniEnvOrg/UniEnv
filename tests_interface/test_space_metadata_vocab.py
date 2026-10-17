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


# ---------------------------------------------------------------------------
# metadata_issues / validate_metadata
# ---------------------------------------------------------------------------

def issues_of(space) -> list:
    return smu.metadata_issues(space)


def test_valid_metadata_has_no_issues():
    space = DictSpace(B, {
        "front": image(metadata=image_metadata(source="front", fps=30)),
        "front_chw": image((3, 48, 64), metadata=image_metadata(layout="CHW", format="bgr")),
        "rgba": image((48, 64, 4), metadata=image_metadata(format="bgra")),
        "gray": image((48, 64), metadata=image_metadata(format="gray", layout="HW")),
        "gray_c": image((48, 64, 1), metadata=image_metadata(format="gray")),
        "depth": box((48, 64), metadata=depth_metadata()),
        "depth_c": box((48, 64, 1), metadata=depth_metadata(layout="HWC", fps=29.97)),
        "depth_chw": box((1, 48, 64), metadata=depth_metadata(layout="CHW")),
        "pose": box((8,), metadata=proprio_metadata(
            units=["m", "m", "m", "normalized", "normalized", "normalized", "normalized", "m"],
            names=["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"],
            frame="base", rotation="quat_wxyz",
        )),
        "action": box((7,), metadata=action_metadata(
            control="eef_pose", mode="delta", normalized=True, range=[[-0.1] * 7, [0.1] * 7], rotation="euler_rxyz",
        )),
        "scalar_range": box((3,), metadata=action_metadata(range=[-1, 1.5], units="rad", rotation="euler_szxz")),
        "custom": box((3,), metadata={"kind": "my_kind", "anything": {"nested": [1, 2]}, "rotation": "axis_angle"}),
        "text_label": TupleSpace(B, [box()], metadata={"label": "x", "kind": "task"}),
    }, metadata={"label": "observation"})
    assert issues_of(space) == []
    smu.validate_metadata(space)
    assert issues_of(box()) == []


def test_batched_images_are_valid():
    batched = image((16, 4, 48, 64, 3), metadata=image_metadata())
    batched_chw = image((16, 3, 48, 64), metadata=image_metadata(layout="CHW"))
    batched_depth = box((5, 48, 64), metadata=depth_metadata())
    for space in (batched, batched_chw, batched_depth):
        assert issues_of(space) == []
    from unienv_interface.space.space_utils import batch_utils as sbu
    assert issues_of(sbu.batch_space(image(metadata=image_metadata()), 8)) == []


@pytest.mark.parametrize("md, fragment", [
    ({"kind": 3}, "'kind' must be a str"),
    ({"frame": ["world"]}, "'frame' must be a str"),
    ({"control": 1}, "'control' must be a str"),
    ({"mode": None}, "'mode' must be a str"),
    ({"source": 0}, "'source' must be a str"),
    ({"label": True}, "'label' must be a str"),
    ({"units": 1}, "'units' must be a str or a list of str"),
    ({"units": ["m", 2]}, "'units' must be a str or a list of str"),
    ({"names": "j0"}, "'names' must be a list of str"),
    ({"names": ["a", "b", None]}, "'names' must be a list of str"),
    ({"normalized": 1}, "'normalized' must be a bool"),
    ({"normalized": "yes"}, "'normalized' must be a bool"),
    ({"fps": 0}, "'fps' must be a positive number"),
    ({"fps": -30}, "'fps' must be a positive number"),
    ({"fps": "30"}, "'fps' must be a positive number"),
    ({"fps": True}, "'fps' must be a positive number"),
    ({"fps": float("inf")}, "'fps' must be a positive number"),
    ({"range": [0, 1, 2]}, "'range' must be [low, high]"),
    ({"range": 1.0}, "'range' must be [low, high]"),
    ({"range": [0, "1"]}, "'range' bounds must be numbers"),
    ({"range": [False, True]}, "'range' bounds must be numbers"),
    ({"range": [1, 0]}, "'range' low must not be greater than high"),
    ({"range": [[0, 0, 2], 1]}, "'range' low must not be greater than high"),
    ({"format": "RGB"}, "'format' must be one of"),
    ({"format": "yuv"}, "'format' must be one of"),
    ({"layout": "WHC"}, "'layout' must be one of"),
    ({"rotation": "quat"}, "'rotation' must be one of"),
    ({"rotation": "euler_xyz"}, "'rotation' must be one of"),
    ({"rotation": "euler_sxxy"}, "'rotation' must be one of"),
    ({"rotation": "euler_qxyz"}, "'rotation' must be one of"),
    ({"rotation": 6}, "'rotation' must be one of"),
])
def test_type_and_value_issues(md, fragment):
    issues = issues_of(box((3,), metadata=md))
    assert len(issues) == 1, issues
    assert issues[0].startswith("<root>: ")
    assert fragment in issues[0]
    with pytest.raises(ValueError, match="1 issue"):
        smu.validate_metadata(box((3,), metadata=md))


@pytest.mark.parametrize("axes", ["sxyz", "sxyx", "sxzy", "syzx", "szxz", "szyx", "rzyx", "rxyz", "ryxy", "rzxz"])
def test_every_transforms3d_euler_axes_is_valid(axes):
    assert issues_of(box((3,), metadata={"rotation": f"euler_{axes}"})) == []


@pytest.mark.parametrize("md, md_ok, fragment", [
    ({"units": ["m", "m"]}, {"units": ["m", "m", "rad"]}, "'units' has 2 entries but the last axis of the space has size 3"),
    ({"names": ["a", "b", "c", "d"]}, {"names": ["a", "b", "c"]}, "'names' has 4 entries but the last axis of the space has size 3"),
    ({"range": [[0, 0], [1, 1]]}, {"range": [[0, 0, 0], [1, 1, 1]]}, "'range' has 2 entries but the last axis of the space has size 3"),
    ({"range": [0, [1, 1]]}, {"range": [0, [1, 1, 1]]}, "'range' has 2 entries but the last axis of the space has size 3"),
])
def test_list_lengths_must_match_the_last_axis(md, md_ok, fragment):
    assert issues_of(box((5, 3), metadata=md)) == [f"<root>: {fragment}"]
    assert issues_of(box((5, 3), metadata=md_ok)) == []


def test_list_lengths_are_not_checked_where_the_last_axis_is_unknown():
    from unienv_interface.space.spaces import DynamicBoxSpace, TextSpace
    dynamic = DynamicBoxSpace(B, -1.0, 1.0, (2, 0), (2, 5), np.float32, metadata={"names": ["a", "b", "c"]})
    assert issues_of(dynamic) == []
    fixed_last = DynamicBoxSpace(B, -1.0, 1.0, (0, 2), (5, 2), np.float32, metadata={"names": ["a", "b", "c"]})
    assert len(issues_of(fixed_last)) == 1
    assert issues_of(TextSpace(B, 5, metadata={"names": ["a", "b", "c"]})) == []
    assert issues_of(box((), metadata={"names": ["a", "b", "c"]})) == []


@pytest.mark.parametrize("shape, md, fragment", [
    ((48, 64, 3), image_metadata(format="rgba"), "format 'rgba' needs 4 channel(s), but the channel axis of layout 'HWC'"),
    ((48, 64, 4), image_metadata(format="rgb"), "format 'rgb' needs 3 channel(s)"),
    ((48, 64, 3), image_metadata(format="gray"), "format 'gray' needs 1 channel(s)"),
    ((48, 64, 3), image_metadata(layout="CHW"), "format 'rgb' needs 3 channel(s), but the channel axis of layout 'CHW' in shape (48, 64, 3) has size 48"),
    ((48, 64), image_metadata(), "kind 'image' with layout 'HWC' needs at least 3 axes, got shape (48, 64)"),
    ((3, 64), image_metadata(layout="CHW"), "kind 'image' with layout 'CHW' needs at least 3 axes"),
    ((48, 64), image_metadata(layout="HW"), "layout 'HW' has a single channel, but format 'rgb' needs 3 channels"),
    ((64,), image_metadata(layout="HW", format="gray"), "kind 'image' with layout 'HW' needs at least 2 axes"),
    ((64,), {"kind": "image"}, "kind 'image' needs at least 2 axes, got shape (64,)"),
    ((48, 64, 3), depth_metadata(layout="HWC"), "kind 'depth' needs 1 channel(s), but the channel axis of layout 'HWC'"),
    ((3, 48, 64), depth_metadata(layout="CHW"), "kind 'depth' needs 1 channel(s), but the channel axis of layout 'CHW'"),
    ((64,), depth_metadata(), "kind 'depth' with layout 'HW' needs at least 2 axes"),
    ((64, 3), {"kind": "segmentation", "layout": "HWC"}, "layout 'HWC' needs at least 3 axes, got shape (64, 3)"),
    ((16, 48, 64, 4), image_metadata(), "format 'rgb' needs 3 channel(s)"),  # batched
])
def test_image_consistency_issues(shape, md, fragment):
    issues = issues_of(box(shape, metadata=md))
    assert len(issues) == 1, issues
    assert fragment in issues[0]


def test_image_checks_are_lenient_where_they_cannot_tell():
    # no layout: the channel axis is unknown
    assert issues_of(box((48, 64, 7), metadata={"kind": "image", "format": "rgb"})) == []
    # no format: any channel count
    assert issues_of(box((48, 64, 2), metadata={"kind": "image", "layout": "HWC"})) == []
    # dynamic channel axis
    from unienv_interface.space.spaces import DynamicBoxSpace
    dynamic = DynamicBoxSpace(B, -1.0, 1.0, (48, 64, 1), (48, 64, 4), np.float32, metadata=image_metadata())
    assert issues_of(dynamic) == []
    # non-array spaces are not shape-checked
    assert issues_of(DictSpace(B, {"a": box()}, metadata=image_metadata())) == []


def test_issues_are_reported_with_paths_for_every_nested_space():
    from unienv_interface.space.spaces import BatchedSpace, TextSpace
    space = DictSpace(B, {
        "cameras": DictSpace(B, {
            "front": image((48, 64, 4), metadata=image_metadata()),
            "ok": image(metadata=image_metadata()),
        }, metadata={"fps": 0}),
        "state": TupleSpace(B, [box((3,)), box((2,), metadata={"names": ["a"]})]),
        "texts": BatchedSpace(TextSpace(B, 5, metadata={"label": 1}), batch_shape=(2,)),
    }, metadata={"normalized": "no"})
    issues = issues_of(space)
    assert [issue.split(": ", 1)[0] for issue in issues] == [
        "<root>", "cameras", "cameras/front", "state/1", "texts/single_space",
    ]
    with pytest.raises(ValueError) as info:
        smu.validate_metadata(space)
    message = str(info.value)
    assert "5 issue(s)" in message
    for issue in issues:
        assert f"  - {issue}" in message


def test_annotate_then_validate_does_not_touch_untouched_leaves():
    space = obs_space()
    annotated = annotate(space, {"cameras/front": image_metadata(format="rgba")})
    assert issues_of(space) == []
    assert issues_of(annotated) == ["cameras/front: format 'rgba' needs 4 channel(s), but the channel axis of layout 'HWC' in shape (480, 640, 3) has size 3"]
    # validation does not change metadata
    assert annotated["cameras"]["wrist"].metadata == {"source": "wrist_cam"}
    assert annotated["cameras"]["wrist"] is space["cameras"]["wrist"]
