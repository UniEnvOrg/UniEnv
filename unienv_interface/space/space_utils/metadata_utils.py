"""Helpers for working with ``Space.metadata`` across nested spaces.

Besides lookup and comparison, this module holds the builders for the shared
metadata vocabulary (``image_metadata``, ``depth_metadata``, ...), ``annotate``
to attach metadata to the nested spaces of an existing space and
``metadata_issues`` / ``validate_metadata`` to check described spaces against
the vocabulary (see "Metadata" in the ``Space`` docstring).
"""
import math
import typing
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from ..spaces import *
from ..metadata import EMPTY_METADATA

__all__ = [
    "child_spaces",
    "spaces_equal_with_metadata",
    "find_spaces_by_metadata",
    "common_metadata",
    "metadata",
    "image_metadata",
    "depth_metadata",
    "proprio_metadata",
    "action_metadata",
    "annotate",
    "metadata_issues",
    "validate_metadata",
]

SpacePath = Tuple[Union[str, int], ...]

def child_spaces(space : Space) -> List[Tuple[Union[str, int], Space]]:
    """The nested spaces of ``space`` as ``(key, subspace)`` pairs.

    Keys are dict keys for ``DictSpace``, positions for ``TupleSpace`` /
    ``UnionSpace``, and ``"single_space"`` / ``"node_feature_space"`` /
    ``"edge_feature_space"`` for ``BatchedSpace`` / ``GraphSpace``.
    """
    if isinstance(space, DictSpace):
        return list(space.spaces.items())
    if isinstance(space, (TupleSpace, UnionSpace)):
        return list(enumerate(space.spaces))
    if isinstance(space, BatchedSpace):
        return [("single_space", space.single_space)]
    if isinstance(space, GraphSpace):
        children = [("node_feature_space", space.node_feature_space)]
        if space.edge_feature_space is not None:
            children.append(("edge_feature_space", space.edge_feature_space))
        return children
    return []

def _metadata_tree_equal(a : Space, b : Space) -> bool:
    if a.metadata != b.metadata:
        return False
    children_a = child_spaces(a)
    children_b = child_spaces(b)
    if len(children_a) != len(children_b):
        return False
    return all(
        key_a == key_b and _metadata_tree_equal(sub_a, sub_b)
        for (key_a, sub_a), (key_b, sub_b) in zip(children_a, children_b)
    )

def spaces_equal_with_metadata(a : Space, b : Space) -> bool:
    """``a == b`` and every pair of corresponding (nested) spaces has equal metadata.

    ``Space.__eq__`` ignores metadata by design; use this where the description
    of the data matters as well.
    """
    return a == b and _metadata_tree_equal(a, b)

def common_metadata(spaces : typing.Iterable[Space]) -> SpaceMetadata:
    """The metadata shared by all ``spaces``: theirs if all are equal, else (or without spaces) empty.

    Used where several spaces become one, e.g. batching differing spaces or
    merging the keys of several ``DictSpace`` s into one: a description that
    not every contributor agrees on fits none of the result.
    """
    iterator = iter(spaces)
    first = next(iterator, None)
    if first is None:
        return EMPTY_METADATA
    metadata = first.metadata
    return metadata if all(space.metadata == metadata for space in iterator) else EMPTY_METADATA

def find_spaces_by_metadata(
    space : Space,
    predicate : Optional[Callable[[Space], bool]] = None,
    **match : Any,
) -> List[Tuple[SpacePath, Space]]:
    """Find the leaves of a (nested) ``DictSpace`` / ``TupleSpace`` by their metadata.

    Returns ``(path, leaf)`` pairs, in iteration order, for every leaf (a space
    that is not a ``DictSpace`` / ``TupleSpace``) whose metadata contains all
    ``match`` items and, if given, for which ``predicate(leaf)`` holds. ``path``
    is the tuple of dict keys / tuple positions leading to the leaf (``()``
    when ``space`` itself is a leaf), i.e. the same path that indexes the data.

    Example::

        cameras = find_spaces_by_metadata(obs_space, kind="image")
        # [(("cameras", "front"), BoxSpace(...)), (("wrist",), BoxSpace(...))]
    """
    found : List[Tuple[SpacePath, Space]] = []

    def visit(current : Space, path : SpacePath) -> None:
        if isinstance(current, DictSpace):
            for key, subspace in current.spaces.items():
                visit(subspace, path + (key,))
            return
        if isinstance(current, TupleSpace):
            for i, subspace in enumerate(current.spaces):
                visit(subspace, path + (i,))
            return
        metadata = current.metadata
        if all(key in metadata and metadata[key] == value for key, value in match.items()) and (
            predicate is None or predicate(current)
        ):
            found.append((path, current))

    visit(space, ())
    return found

# ---------------------------------------------------------------------------
# Builders for the shared vocabulary
# ---------------------------------------------------------------------------

def metadata(kind : Optional[str], **fields : Any) -> Dict[str, Any]:
    """A metadata ``dict`` of the given ``kind`` with ``fields``; ``None`` values are dropped.

    Example::

        metadata("force", units="N", frame="wrist")
        # {"kind": "force", "units": "N", "frame": "wrist"}
    """
    built = {"kind": kind, **fields}
    return {key: value for key, value in built.items() if value is not None}

def image_metadata(
    *,
    format : Optional[str] = "rgb",
    layout : Optional[str] = "HWC",
    source : Optional[str] = None,
    fps : Optional[float] = None,
    **extra : Any,
) -> Dict[str, Any]:
    """Metadata of a color image (``kind="image"``); ``None`` values are dropped.

    ``format`` is ``"rgb"``, ``"bgr"``, ``"rgba"``, ``"bgra"`` or ``"gray"`` and
    ``layout`` the order of the trailing axes, ``"HWC"``, ``"CHW"`` or ``"HW"``.
    """
    return metadata("image", format=format, layout=layout, source=source, fps=fps, **extra)

def depth_metadata(
    *,
    units : Optional[str] = "m",
    layout : Optional[str] = "HW",
    source : Optional[str] = None,
    fps : Optional[float] = None,
    **extra : Any,
) -> Dict[str, Any]:
    """Metadata of a depth image (``kind="depth"``); ``None`` values are dropped.

    ``layout`` is ``"HW"`` (no channel axis), or ``"HWC"`` / ``"CHW"`` with a single channel.
    """
    return metadata("depth", units=units, layout=layout, source=source, fps=fps, **extra)

def proprio_metadata(
    *,
    units : Optional[Union[str, Sequence[str]]] = None,
    names : Optional[Sequence[str]] = None,
    frame : Optional[str] = None,
    rotation : Optional[str] = None,
    **extra : Any,
) -> Dict[str, Any]:
    """Metadata of proprioceptive data (``kind="proprio"``); ``None`` values are dropped.

    ``units`` and ``names`` may be lists with one entry per element of the last axis.
    """
    return metadata("proprio", units=units, names=names, frame=frame, rotation=rotation, **extra)

def action_metadata(
    *,
    control : Optional[str] = None,
    mode : Optional[str] = None,
    units : Optional[Union[str, Sequence[str]]] = None,
    normalized : Optional[bool] = None,
    range : Optional[Sequence[Any]] = None,
    names : Optional[Sequence[str]] = None,
    frame : Optional[str] = None,
    rotation : Optional[str] = None,
    **extra : Any,
) -> Dict[str, Any]:
    """Metadata of an action (``kind="action"``); ``None`` values are dropped.

    ``control`` is what is commanded (e.g. ``"joint_position"``, ``"eef_pose"``),
    ``mode`` how (``"absolute"``, ``"delta"`` or ``"velocity"``). With
    ``normalized=True``, ``range=[low, high]`` gives the physical range the
    normalized values map to.
    """
    return metadata(
        "action", control=control, mode=mode, units=units, normalized=normalized,
        range=range, names=names, frame=frame, rotation=rotation, **extra,
    )

# ---------------------------------------------------------------------------
# Annotating nested spaces
# ---------------------------------------------------------------------------

def _normalize_path(path : Union[str, Sequence[Union[str, int]]]) -> SpacePath:
    if isinstance(path, str):
        return tuple(path.split("/")) if path else ()
    if isinstance(path, tuple):
        return path
    raise TypeError(f"Space paths must be '/'-separated strings or tuples of keys / indices, got {type(path).__name__}: {path!r}")

def _resolve_child(space : Space, key : Union[str, int]) -> Tuple[bool, Any]:
    """``(found, key)``: the key of ``space``'s child addressed by a path element."""
    if isinstance(space, DictSpace):
        return key in space.spaces, key
    if isinstance(space, TupleSpace):
        if isinstance(key, str):
            try:
                key = int(key)
            except ValueError:
                return False, key
        if isinstance(key, int) and not isinstance(key, bool) and -len(space.spaces) <= key < len(space.spaces):
            return True, key % len(space.spaces)
    return False, key

def annotate(
    space : Space,
    mapping : Mapping[Union[str, Tuple[Union[str, int], ...]], Mapping[str, Any]],
    *,
    replace : bool = False,
    strict : bool = True,
) -> Space:
    """Return ``space`` with metadata attached to the nested spaces named in ``mapping``.

    Each key of ``mapping`` is a path through nested ``DictSpace`` and
    ``TupleSpace`` spaces: a ``"/"``-separated string (``"cameras/front"``, ``"0/joints"``; tuple
    positions as numbers) or a tuple of dict keys / tuple positions
    (``("cameras", "front")``); ``""`` or ``()`` is ``space`` itself. Its value
    is merged into the metadata of that space, or replaces it with
    ``replace=True`` (see :meth:`Space.with_metadata`).

    Spaces that are not on an annotated path are kept as they are (the same
    objects), as is the metadata of every space that is not annotated; the
    containers on the paths are rebuilt. With ``strict=True`` (the default) a
    ``KeyError`` listing every path that does not exist is raised, otherwise
    such paths are skipped.

    Example::

        obs_space = from_gym_space(env.observation_space, backend)
        obs_space = annotate(obs_space, {
            "pixels": image_metadata(source="front_cam"),
            "agent/qpos": proprio_metadata(units="rad"),
        })
    """
    tree : Dict[str, Any] = {"updates": [], "children": {}}
    missing : List[Any] = []
    for raw_path, updates in mapping.items():
        if not isinstance(updates, Mapping):
            raise TypeError(f"The metadata for path {raw_path!r} must be a mapping, got {type(updates).__name__}: {updates!r}")
        current_space, node, found = space, tree, True
        for element in _normalize_path(raw_path):
            found, key = _resolve_child(current_space, element)
            if not found:
                break
            current_space = current_space.spaces[key]
            node = node["children"].setdefault(key, {"updates": [], "children": {}})
        if found:
            node["updates"].append(updates)
        else:
            missing.append(raw_path)
    if missing and strict:
        raise KeyError(f"annotate: no nested space at path(s) {missing}")

    def rebuild(current : Space, node : Dict[str, Any]) -> Space:
        if node["children"]:
            if isinstance(current, DictSpace):
                children = dict(current.spaces)
                for key, child_node in node["children"].items():
                    children[key] = rebuild(children[key], child_node)
                current = DictSpace(current.backend, children, metadata=current.metadata)
            else:
                children = list(current.spaces)
                for key, child_node in node["children"].items():
                    children[key] = rebuild(children[key], child_node)
                current = TupleSpace(current.backend, children, metadata=current.metadata)
        for updates in node["updates"]:
            current = current.with_metadata(updates, replace=replace)
        return current

    return rebuild(space, tree)

# ---------------------------------------------------------------------------
# Validation against the vocabulary
# ---------------------------------------------------------------------------

_IMAGE_FORMAT_CHANNELS : Dict[str, int] = {"rgb": 3, "bgr": 3, "rgba": 4, "bgra": 4, "gray": 1}
_IMAGE_LAYOUTS : Tuple[str, ...] = ("HWC", "CHW", "HW")
_ROTATIONS : Tuple[str, ...] = ("quat_wxyz", "quat_xyzw", "rot6d", "matrix", "axis_angle")
_STR_KEYS : Tuple[str, ...] = ("kind", "frame", "control", "mode", "source", "label")

def _is_valid_rotation(value : str) -> bool:
    if value in _ROTATIONS:
        return True
    if not value.startswith("euler_"):
        return False
    # transforms3d axes: "s" (static / extrinsic) or "r" (rotating / intrinsic) and three
    # axes in which consecutive axes differ, e.g. "sxyz", "rzyx", "szxz"
    axes = value[len("euler_"):]
    return (
        len(axes) == 4 and axes[0] in "sr" and all(axis in "xyz" for axis in axes[1:])
        and axes[1] != axes[2] and axes[2] != axes[3]
    )

def _is_number(value : Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def _is_str_list(value : Any) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value)

def _static_shape(space : Space) -> Optional[Tuple[Optional[int], ...]]:
    """The shape of an array space, ``None`` for the size of a dynamic axis; ``None`` if not an array space."""
    if isinstance(space, (BoxSpace, BinarySpace)):
        return tuple(space.shape)
    if isinstance(space, DynamicBoxSpace):
        return tuple(low if low == high else None for low, high in zip(space.shape_low, space.shape_high))
    return None

def _space_metadata_issues(space : Space) -> List[str]:
    """Issues with the metadata of ``space`` itself (nested spaces are not visited)."""
    md = space.metadata
    if not md:
        return []
    issues : List[str] = []
    shape = _static_shape(space)
    last_axis = shape[-1] if shape else None  # None: unknown, or not an array space

    def check_length(key : str, length : int) -> None:
        if last_axis is not None and length != last_axis:
            issues.append(f"{key!r} has {length} entries but the last axis of the space has size {last_axis}")

    for key in _STR_KEYS:
        if key in md and not isinstance(md[key], str):
            issues.append(f"{key!r} must be a str, got {md[key]!r}")

    if "format" in md and md["format"] not in _IMAGE_FORMAT_CHANNELS:
        issues.append(f"'format' must be one of {sorted(_IMAGE_FORMAT_CHANNELS)}, got {md['format']!r}")
    if "layout" in md and md["layout"] not in _IMAGE_LAYOUTS:
        issues.append(f"'layout' must be one of {list(_IMAGE_LAYOUTS)}, got {md['layout']!r}")
    if "rotation" in md and not (isinstance(md["rotation"], str) and _is_valid_rotation(md["rotation"])):
        issues.append(
            f"'rotation' must be one of {list(_ROTATIONS)} or 'euler_<axes>' with transforms3d axes "
            f"(e.g. 'euler_sxyz', 'euler_rzyx'), got {md['rotation']!r}"
        )

    if "units" in md:
        units = md["units"]
        if _is_str_list(units):
            check_length("units", len(units))
        elif not isinstance(units, str):
            issues.append(f"'units' must be a str or a list of str, got {units!r}")
    if "names" in md:
        names = md["names"]
        if _is_str_list(names):
            check_length("names", len(names))
        else:
            issues.append(f"'names' must be a list of str, got {names!r}")
    if "normalized" in md and not isinstance(md["normalized"], bool):
        issues.append(f"'normalized' must be a bool, got {md['normalized']!r}")
    if "fps" in md:
        fps = md["fps"]
        if not (_is_number(fps) and math.isfinite(fps) and fps > 0):
            issues.append(f"'fps' must be a positive number, got {fps!r}")
    if "range" in md:
        issues.extend(_range_issues(md["range"], check_length))

    kind = md.get("kind")
    layout = md.get("layout") if md.get("layout") in _IMAGE_LAYOUTS else None
    if shape is not None and (layout is not None or kind in ("image", "depth")):
        issues.extend(_image_issues(kind, layout, md.get("format"), shape))
    return issues

def _range_issues(value : Any, check_length : Callable[[str, int], None]) -> List[str]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        return [f"'range' must be [low, high], got {value!r}"]
    bounds = []
    for bound in value:
        if _is_number(bound):
            bounds.append([bound])
        elif isinstance(bound, (list, tuple)) and len(bound) > 0 and all(_is_number(item) for item in bound):
            bounds.append(list(bound))
        else:
            return [f"'range' bounds must be numbers or lists of numbers, got {value!r}"]
    for length in sorted({len(bound) for bound in value if isinstance(bound, (list, tuple))}):
        check_length("range", length)
    low, high = bounds
    if len(low) == 1:
        low = low * len(high)
    if len(high) == 1:
        high = high * len(low)
    if len(low) != len(high):
        return [f"'range' bounds have different lengths, got {value!r}"]
    if any(lo > hi for lo, hi in zip(low, high)):
        return [f"'range' low must not be greater than high, got {value!r}"]
    return []

def _image_issues(
    kind : Optional[str],
    layout : Optional[str],
    image_format : Any,
    shape : Tuple[Optional[int], ...],
) -> List[str]:
    """Check the trailing axes of an array space against ``layout`` / ``format`` (leading batch axes are allowed)."""
    min_ndim = 2 if layout in (None, "HW") else 3
    if len(shape) < min_ndim:
        if kind not in ("image", "depth"):
            what = f"layout {layout!r}"
        else:
            what = f"kind {kind!r}" + ("" if layout is None else f" with layout {layout!r}")
        return [f"{what} needs at least {min_ndim} axes, got shape {shape}"]
    if layout is None:
        return []  # cannot tell whether there is a channel axis
    channels = 1 if layout == "HW" else shape[-1] if layout == "HWC" else shape[-3]
    expected = None
    if kind == "depth":
        expected, reason = 1, "kind 'depth'"
    elif image_format in _IMAGE_FORMAT_CHANNELS:
        expected, reason = _IMAGE_FORMAT_CHANNELS[image_format], f"format {image_format!r}"
    if expected is None or channels is None or channels == expected:
        return []
    if layout == "HW":
        return [f"layout 'HW' has a single channel, but {reason} needs {expected} channels"]
    return [f"{reason} needs {expected} channel(s), but the channel axis of layout {layout!r} in shape {shape} has size {channels}"]

def _format_path(path : SpacePath) -> str:
    return "/".join(str(key) for key in path) if path else "<root>"

def metadata_issues(space : Space) -> List[str]:
    """Problems with the metadata of ``space`` and all its nested spaces, as messages.

    Checks the keys of the shared vocabulary (see "Metadata" in the ``Space``
    docstring), leniently: unknown keys and kinds are fine and only what can be
    checked is checked.

    * types: ``kind``, ``frame``, ``control``, ``mode``, ``source`` and ``label``
      are strings, ``units`` a string or a list of strings, ``names`` a list of
      strings, ``normalized`` a bool, ``fps`` a positive number and ``range``
      ``[low, high]`` of numbers or lists of numbers (``low <= high``);
    * values: ``format`` (``"rgb"``, ``"bgr"``, ``"rgba"``, ``"bgra"``,
      ``"gray"``), ``layout`` (``"HWC"``, ``"CHW"``, ``"HW"``) and ``rotation``
      (``"quat_wxyz"``, ``"quat_xyzw"``, ``"rot6d"``, ``"matrix"``,
      ``"axis_angle"`` or ``"euler_<axes>"`` with transforms3d axes such as
      ``"sxyz"`` / ``"rzyx"``);
    * array spaces (``BoxSpace``, ``BinarySpace``, ``DynamicBoxSpace``): lists in
      ``units``, ``names`` and ``range`` have one entry per element of the last
      axis; ``layout`` matches the trailing axes (``"HWC"`` / ``"CHW"``: at
      least 3, ``"HW"``: at least 2; leading batch axes are allowed), the
      channel count matches ``format`` (``rgb`` / ``bgr``: 3, ``rgba`` /
      ``bgra``: 4, ``gray``: 1) and ``kind="depth"`` has a single channel;
      ``kind="image"`` / ``"depth"`` spaces have at least 2 axes.

    Each message starts with the ``"/"``-separated path of the space
    (``"<root>"`` for ``space`` itself), e.g.
    ``"cameras/front: format 'rgb' needs 3 channel(s), ..."``.
    """
    issues : List[str] = []

    def visit(current : Space, path : SpacePath) -> None:
        issues.extend(f"{_format_path(path)}: {issue}" for issue in _space_metadata_issues(current))
        for key, subspace in child_spaces(current):
            visit(subspace, path + (key,))

    visit(space, ())
    return issues

def validate_metadata(space : Space) -> None:
    """Raise ``ValueError`` listing every :func:`metadata_issues` of ``space`` (if there is any).

    Meant for tests and construction-time checks, e.g. in an adaptor's test suite::

        def test_observation_space_metadata():
            validate_metadata(make_env().observation_space)
    """
    issues = metadata_issues(space)
    if issues:
        raise ValueError(
            f"Space metadata has {len(issues)} issue(s):\n" + "\n".join(f"  - {issue}" for issue in issues)
        )
