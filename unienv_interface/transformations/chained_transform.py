from typing import Union, Any, Optional, Iterable, List, Callable, Dict

from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.space import Space, DictSpace
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.transformations import serialization_utils as tsu

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

        source_spaces = [source_space]
        for transformation in self.transformations:
            next_space = transformation.get_target_space_from_source(source_spaces[-1])
            source_spaces.append(next_space)
        
        inverted_transformations = []
        for i in reversed(range(len(self.transformations))):
            inverted_transformations.append(self.transformations[i].direction_inverse(source_spaces[i]))

        return ChainedTransformation(
            inverted_transformations
        )

    def close(self):
        for transformation in self.transformations:
            transformation.close()

    def serialize(
        self,
        source_space: Optional[Space] = None,
    ) -> Dict[str, Any]:
        current_source_space = source_space
        serialized_transformations = []
        for transformation in self.transformations:
            serialized_transformations.append(
                tsu.transformation_to_json(transformation, source_space=current_source_space)
            )
            if current_source_space is not None:
                current_source_space = transformation.get_target_space_from_source(current_source_space)

        return {
            "transformations": serialized_transformations,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space] = None,
    ) -> "ChainedTransformation":
        current_source_space = source_space
        transformations = []
        for t_data in json_data["transformations"]:
            transformation = tsu.json_to_transformation(t_data, source_space=current_source_space)
            transformations.append(transformation)
            if current_source_space is not None:
                current_source_space = transformation.get_target_space_from_source(current_source_space)

        return cls(
            transformations=transformations
        )
