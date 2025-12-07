from typing import Union, Any, Optional, Mapping, List, Callable, Dict

from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.space import Space, DictSpace, TupleSpace
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

import copy
from .transformation import DataTransformation, TargetDataT

def default_is_leaf_fn(space : Space[Any, BDeviceType, BDtypeType, BRNGType]):
    return not isinstance(space, (DictSpace, TupleSpace))

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
    ) -> Optional["IterativeTransformation"]:
        if not self.has_inverse:
            return None

        return IterativeTransformation(
            self.transformation.direction_inverse(),
            is_leaf_node_fn=self.inv_is_leaf_node_fn,
            inv_is_leaf_node_fn=self.is_leaf_node_fn
        )

    def close(self):
        self.transformation.close()