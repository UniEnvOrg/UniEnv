from ..space import Space
from .binary import BinarySpace
from .box import BoxSpace
from .dict import DictSpace
from .graph import GraphSpace, GraphInstance
from .text import TextSpace
from .tuple import TupleSpace
from .union import UnionSpace

__all__ = [
    "Space",
    "BinarySpace",
    "BoxSpace",
    "DictSpace",
    "GraphSpace",
    "GraphInstance",
    "TextSpace",
    "TupleSpace",
    "UnionSpace"
]
