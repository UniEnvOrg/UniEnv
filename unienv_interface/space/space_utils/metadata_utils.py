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
    "metadata",
    "image_metadata",
    "depth_metadata",
    "proprio_metadata",
    "action_metadata",
    "annotate",
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
