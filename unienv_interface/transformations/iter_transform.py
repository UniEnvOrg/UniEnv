from typing import Union, Any, Optional, Mapping, List, Callable, Dict, Tuple

from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.space import Space, DictSpace, TupleSpace
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.transformations import serialization_utils as tsu
from unienv_interface.utils.symbol_util import serialize_function, deserialize_function

import copy
from .transformation import DataTransformation, TargetDataT


def default_is_leaf_fn(space: Space[Any, BDeviceType, BDtypeType, BRNGType]):
    return not isinstance(space, (DictSpace, TupleSpace))


def find_leaf_source_space(
    source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]],
    is_leaf_node_fn: Callable[[Space[Any, BDeviceType, BDtypeType, BRNGType]], bool],
) -> Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]]:
    if source_space is None:
        return None
    if is_leaf_node_fn(source_space):
        return source_space
    if isinstance(source_space, DictSpace):
        for subspace in source_space.spaces.values():
            leaf_space = find_leaf_source_space(subspace, is_leaf_node_fn)
            if leaf_space is not None:
                return leaf_space
        return None
    if isinstance(source_space, TupleSpace):
        for subspace in source_space.spaces:
            leaf_space = find_leaf_source_space(subspace, is_leaf_node_fn)
            if leaf_space is not None:
                return leaf_space
        return None
    raise ValueError(f"Unsupported space type: {type(source_space)}")


def _map_leaf_spaces(
    source_space: Space[Any, BDeviceType, BDtypeType, BRNGType],
    is_leaf_node_fn: Callable[[Space[Any, BDeviceType, BDtypeType, BRNGType]], bool],
    fn: Callable[[Space[Any, BDeviceType, BDtypeType, BRNGType]], Any],
) -> Any:
    """Apply ``fn`` to every leaf space, returning the results in the space's dict/tuple structure."""
    if is_leaf_node_fn(source_space):
        return fn(source_space)
    elif isinstance(source_space, DictSpace):
        return {
            key: _map_leaf_spaces(subspace, is_leaf_node_fn, fn)
            for key, subspace in source_space.spaces.items()
        }
    elif isinstance(source_space, TupleSpace):
        return tuple(
            _map_leaf_spaces(subspace, is_leaf_node_fn, fn)
            for subspace in source_space.spaces
        )
    else:
        raise ValueError(f"Unsupported space type: {type(source_space)}")


def _iter_leaves(structure: Any):
    if isinstance(structure, Mapping):
        for value in structure.values():
            yield from _iter_leaves(value)
    elif isinstance(structure, tuple):
        for value in structure:
            yield from _iter_leaves(value)
    else:
        yield structure


class LeafwiseTransformation(DataTransformation):
    """Apply a separate transformation to each leaf of a dict/tuple space.

    ``transformations`` mirrors the space structure: a mapping for a
    ``DictSpace``, a tuple for a ``TupleSpace`` and a ``DataTransformation`` for
    a leaf. This is what :meth:`IterativeTransformation.direction_inverse`
    returns when the leaves need different inverses (e.g. rescaling leaves with
    different source bounds).
    """
    def __init__(
        self,
        transformations: Union[DataTransformation, Mapping[str, Any], Tuple[Any, ...]],
    ):
        self.transformations = transformations
        self.has_inverse = all(t.has_inverse for t in _iter_leaves(transformations))

    @staticmethod
    def _map(
        structure: Any,
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]],
        fn: Callable[[DataTransformation, Optional[Space], Any], Any],
        data: Any = None,
    ) -> Any:
        if isinstance(structure, DataTransformation):
            return fn(structure, source_space, data)
        if isinstance(structure, Mapping):
            if source_space is not None and not isinstance(source_space, DictSpace):
                raise ValueError(f"Expected a DictSpace, got {type(source_space)}")
            return {
                key: LeafwiseTransformation._map(
                    sub_structure,
                    None if source_space is None else source_space[key],
                    fn,
                    None if data is None else data[key],
                )
                for key, sub_structure in structure.items()
            }
        if isinstance(structure, tuple):
            if source_space is not None and not isinstance(source_space, TupleSpace):
                raise ValueError(f"Expected a TupleSpace, got {type(source_space)}")
            return tuple(
                LeafwiseTransformation._map(
                    sub_structure,
                    None if source_space is None else source_space[i],
                    fn,
                    None if data is None else data[i],
                )
                for i, sub_structure in enumerate(structure)
            )
        raise ValueError(f"Unsupported transformation structure: {type(structure)}")

    def get_target_space_from_source(self, source_space):
        def build(structure: Any, space: Space) -> Space:
            if isinstance(structure, DataTransformation):
                return structure.get_target_space_from_source(space)
            if isinstance(structure, Mapping):
                assert isinstance(space, DictSpace), f"Expected a DictSpace, got {type(space)}"
                rsts = {key: build(sub_structure, space[key]) for key, sub_structure in structure.items()}
                first = next(iter(rsts.values()), space)
                return DictSpace(first.backend, rsts, device=first.device)
            assert isinstance(space, TupleSpace), f"Expected a TupleSpace, got {type(space)}"
            rsts = tuple(build(sub_structure, space[i]) for i, sub_structure in enumerate(structure))
            first = rsts[0] if len(rsts) > 0 else space
            return TupleSpace(first.backend, rsts, device=first.device)
        return build(self.transformations, source_space)

    def transform(self, source_space, data):
        return self._map(
            self.transformations,
            source_space,
            lambda t, space, leaf_data: t.transform(space, leaf_data),
            data,
        )

    def direction_inverse(self, source_space = None) -> Optional["LeafwiseTransformation"]:
        if not self.has_inverse:
            return None
        return LeafwiseTransformation(self._map(
            self.transformations,
            source_space,
            lambda t, space, _: t.direction_inverse(space),
        ))

    def close(self):
        for transformation in _iter_leaves(self.transformations):
            transformation.close()

    def serialize(
        self,
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> Dict[str, Any]:
        def encode(structure: Any, space: Optional[Space]) -> Dict[str, Any]:
            if isinstance(structure, DataTransformation):
                return {"leaf": tsu.transformation_to_json(structure, source_space=space)}
            if isinstance(structure, Mapping):
                return {"dict": {
                    key: encode(sub_structure, None if space is None else space[key])
                    for key, sub_structure in structure.items()
                }}
            return {"tuple": [
                encode(sub_structure, None if space is None else space[i])
                for i, sub_structure in enumerate(structure)
            ]}
        return {"transformations": encode(self.transformations, source_space)}

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> "LeafwiseTransformation":
        def decode(node: Dict[str, Any], space: Optional[Space]) -> Any:
            if "leaf" in node:
                return tsu.json_to_transformation(node["leaf"], source_space=space)
            if "dict" in node:
                return {
                    key: decode(sub_node, None if space is None else space[key])
                    for key, sub_node in node["dict"].items()
                }
            return tuple(
                decode(sub_node, None if space is None else space[i])
                for i, sub_node in enumerate(node["tuple"])
            )
        return cls(decode(json_data["transformations"], source_space))


class IterativeTransformation(DataTransformation):
    def __init__(
        self,
        transformation: DataTransformation,
        is_leaf_node_fn: Callable[[Space[Any, BDeviceType, BDtypeType, BRNGType]], bool] = default_is_leaf_fn,
        inv_is_leaf_node_fn: Callable[[Space[Any, BDeviceType, BDtypeType, BRNGType]], bool] = default_is_leaf_fn
    ):
        self.transformation = transformation
        self.is_leaf_node_fn = is_leaf_node_fn
        self.inv_is_leaf_node_fn = inv_is_leaf_node_fn
        self.has_inverse = transformation.has_inverse

    def get_target_space_from_source(
        self, 
        source_space : Space[Any, BDeviceType, BDtypeType, BRNGType]
    ):
        if self.is_leaf_node_fn(source_space):
            return self.transformation.get_target_space_from_source(source_space)
        elif isinstance(source_space, DictSpace):
            rsts = {
                key: self.get_target_space_from_source(subspace)
                for key, subspace in source_space.spaces.items()
            }
            backend = source_space.backend if len(rsts) == 0 else next(iter(rsts.values())).backend
            device = source_space.device if len(rsts) == 0 else next(iter(rsts.values())).device
            return DictSpace(
                backend,
                rsts,
                device=device
            )
        elif isinstance(source_space, TupleSpace):
            rsts = tuple(
                self.get_target_space_from_source(subspace)
                for subspace in source_space.spaces
            )
            backend = source_space.backend if len(rsts) == 0 else next(iter(rsts)).backend
            device = source_space.device if len(rsts) == 0 else next(iter(rsts)).device
            return TupleSpace(
                backend,
                rsts,
                device=device
            )
        else:
            raise ValueError(f"Unsupported space type: {type(source_space)}")

    def transform(
        self, 
        source_space: Space,
        data: Union[Mapping[str, Any], BArrayType]
    ) -> Union[Mapping[str, Any], BArrayType]:
        if self.is_leaf_node_fn(source_space):
            return self.transformation.transform(source_space, data)
        elif isinstance(source_space, DictSpace):
            return {
                key: self.transform(subspace, data[key])
                for key, subspace in source_space.spaces.items()
            }
        elif isinstance(source_space, TupleSpace):
            return tuple(
                self.transform(subspace, data[i])
                for i, subspace in enumerate(source_space.spaces)
            )
        else:
            raise ValueError(f"Unsupported space type: {type(source_space)}")

    def direction_inverse(
        self,
        source_space = None,
    ) -> Optional[Union["IterativeTransformation", LeafwiseTransformation]]:
        """Return the inverse transformation.

        Inverses that depend on the source space (e.g. ``RescaleTransformation``)
        are computed per leaf of ``source_space``. When all leaves are equal a
        single ``IterativeTransformation`` is returned, otherwise a
        ``LeafwiseTransformation`` holding one inverse per leaf.
        """
        if not self.has_inverse:
            return None

        if source_space is not None:
            leaf_spaces = list(_iter_leaves(_map_leaf_spaces(source_space, self.is_leaf_node_fn, lambda s: s)))
            if len(leaf_spaces) > 0 and any(leaf != leaf_spaces[0] for leaf in leaf_spaces[1:]):
                return LeafwiseTransformation(_map_leaf_spaces(
                    source_space,
                    self.is_leaf_node_fn,
                    self.transformation.direction_inverse,
                ))
            leaf_source_space = leaf_spaces[0] if len(leaf_spaces) > 0 else None
        else:
            leaf_source_space = None

        return IterativeTransformation(
            self.transformation.direction_inverse(leaf_source_space),
            is_leaf_node_fn=self.inv_is_leaf_node_fn,
            inv_is_leaf_node_fn=self.is_leaf_node_fn
        )

    def close(self):
        self.transformation.close()

    def serialize(
        self,
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> Dict[str, Any]:
        leaf_source_space = find_leaf_source_space(source_space, self.is_leaf_node_fn)
        return {
            "transformation": tsu.transformation_to_json(
                self.transformation,
                source_space=leaf_source_space,
            ),
            "is_leaf_node_fn": serialize_function(self.is_leaf_node_fn),
            "inv_is_leaf_node_fn": serialize_function(self.inv_is_leaf_node_fn),
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> "IterativeTransformation":
        is_leaf_node_fn = deserialize_function(json_data.get("is_leaf_node_fn", {"mode": "name", "value": f"{default_is_leaf_fn.__module__}.{default_is_leaf_fn.__qualname__}"}))
        inv_is_leaf_node_fn = deserialize_function(json_data.get("inv_is_leaf_node_fn", {"mode": "name", "value": f"{default_is_leaf_fn.__module__}.{default_is_leaf_fn.__qualname__}"}))
        leaf_source_space = find_leaf_source_space(source_space, is_leaf_node_fn)
        return cls(
            transformation=tsu.json_to_transformation(
                json_data["transformation"],
                source_space=leaf_source_space,
            ),
            is_leaf_node_fn=is_leaf_node_fn,
            inv_is_leaf_node_fn=inv_is_leaf_node_fn,
        )
