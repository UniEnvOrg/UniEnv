from typing import Union, Any, Optional, Iterable, List, Callable, Dict

from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.space import Space, DictSpace
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

import copy
from .transformation import DataTransformation, TargetDataT

class ChainedTransformation(DataTransformation):
    def __init__(
        self,
        transformations : Iterable[DataTransformation],
    ):
        self.transformations = list(transformations)
        self.has_inverse = all(
            transformation.has_inverse for transformation in self.transformations
        )

    def get_target_space_from_source(
        self, 
        source_space : DictSpace[BDeviceType, BDtypeType, BRNGType]
    ):
        space = source_space
        for transformation in self.transformations:
            space = transformation.get_target_space_from_source(space)
        return space

    def transform(
        self, 
        source_space,
        data
    ):
        new_space = source_space
        new_data = data
        for transformation in self.transformations:
            next_space = transformation.get_target_space_from_source(new_space)
            new_data = transformation.transform(new_space, new_data)
            new_space = next_space
        return new_data

    def direction_inverse(
        self,
        source_space: Optional[Space] = None,
    ) -> Optional["ChainedTransformation"]:
        if not self.has_inverse:
            return None
        
        inverse_mapping = {
            key: transformation.direction_inverse(source_space)
            for key, transformation in self.mapping.items()
        }

        return ChainedTransformation(
            transformations=[
                transformation.direction_inverse(source_space)
                for transformation in reversed(self.transformations)
            ]
        )

    def close(self):
        for transformation in self.transformations:
            transformation.close()