from typing import Union, Any, Optional, Iterable, List, Callable, Dict

from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.space import Space, DictSpace
from unienv_interface.space.space import _holds_jax_tracer
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
        new_data = data
        for transformation, space in zip(self.transformations, self._intermediate_source_spaces(source_space)):
            new_data = transformation.transform(space, new_data)
        return new_data

    def _intermediate_source_spaces(self, source_space : Space) -> List[Space]:
        """The source space each transformation of the chain sees, cached on the instance.

        ``transform`` is called with the same source space every step, and
        rebuilding the intermediate spaces (space construction and validation)
        dominated its cost. The cache is keyed on the identity of the source
        space and of the transformations, plus the global space-mutation and
        transformation-config counters, so in-place changes to a space or a
        transformation (see ``Space._notify_mutation`` and
        ``DataTransformation.__setattr__``) rebuild it. Spaces built inside a jax
        trace (holding tracers) are never cached.
        """
        transformations = self.transformations
        cache = self.__dict__.get("_spaces_cache")
        if (
            cache is not None
            and cache[0] is source_space
            and cache[1] == Space._mutation_epoch
            and cache[2] == DataTransformation._config_epoch
            and len(cache[3]) == len(transformations)
            and all(a is b for a, b in zip(cache[3], transformations))
        ):
            return cache[4]
        spaces = [source_space]
        for transformation in transformations[:-1]:  # the final target space is not needed
            spaces.append(transformation.get_target_space_from_source(spaces[-1]))
        if not any(_holds_jax_tracer(space) for space in spaces):
            # Counters read after building: transformations may build new spaces via
            # in-place assignment (e.g. ``DictSpace.__setitem__`` on a copy)
            self._spaces_cache = (
                source_space, Space._mutation_epoch, DataTransformation._config_epoch,
                tuple(transformations), spaces,
            )
        return spaces

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_spaces_cache", None)
        return state

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
