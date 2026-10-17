"""Helpers for working with ``Space.metadata`` across nested spaces."""
import typing
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

from ..spaces import *

__all__ = [
    "child_spaces",
    "spaces_equal_with_metadata",
    "find_spaces_by_metadata",
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
