from typing import Union, Any, Optional, Mapping, List, Callable, Dict

from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.space import Space, DictSpace
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

import copy
from .transformation import DataTransformation, TargetDataT

def get_chained_value(
    data : Mapping[str, Any],
    chained_key : List[str],
    ignore_missing_keys : bool = False,
) -> Any:
    assert len(chained_key) >= 1, "Chained key must have at least one key"
    if chained_key[0] not in data.keys():
        if ignore_missing_keys:
            return None
        else:
            raise KeyError(f"Key '{chained_key[0]}' not found in data")
    
    if len(chained_key) == 1:
        return data[chained_key[0]]
    else:
        return get_chained_value(
            data[chained_key[0]],
            chained_key[1:],
            ignore_missing_keys=ignore_missing_keys
        )

def call_function_on_chained_dict(
    data : Mapping[str, Any],
    chained_key : List[str],
    function : Callable[[Any], Any],
    ignore_missing_keys : bool = False,
) -> Mapping[str, Any]:
    assert len(chained_key) >= 1, "Chained key must have at least one key"
    if chained_key[0] not in data.keys():
        if ignore_missing_keys:
            return data
        else:
            raise KeyError(f"Key '{chained_key[0]}' not found in data")
    
    new_data = copy.copy(data)
    if len(chained_key) == 1:    
        new_data[chained_key[0]] = function(data[chained_key[0]])
    else:
        new_data[chained_key[0]] = call_function_on_chained_dict(
            data[chained_key[0]],
            chained_key[1:],
            function,
            ignore_missing_keys=ignore_missing_keys
        )
    return new_data

def call_conditioned_function_on_chained_dict(
    space : DictSpace[BDeviceType, BDtypeType, BRNGType],
    data : Mapping[str, Any],
    chained_key : List[str],
    function : Callable[[Space, Any], Any],
    ignore_missing_keys : bool = False,
) -> Mapping[str, Any]:
    assert len(chained_key) >= 1, "Chained key must have at least one key"
    if chained_key[0] not in data.keys() or chained_key[0] not in space.keys():
        if ignore_missing_keys:
            return data
        else:
            raise KeyError(f"Key '{chained_key[0]}' not found in data")
    
    new_data = copy.copy(data)
    if len(chained_key) == 1:
        new_data[chained_key[0]] = function(space[chained_key[0]], data[chained_key[0]])
    else:
        assert isinstance(space[chained_key[0]], DictSpace), \
            f"Expected DictSpace for key '{chained_key[0]}', but got {type(space[chained_key[0]])}"
        new_data[chained_key[0]] = call_conditioned_function_on_chained_dict(
            space[chained_key[0]],
            data[chained_key[0]],
            chained_key[1:],
            function,
            ignore_missing_keys=ignore_missing_keys
        )
    return new_data

class DictTransformation(DataTransformation):
    def __init__(
        self,
        mapping: Dict[str, DataTransformation],
        ignore_missing_keys : bool = False
    ):
        self.mapping = mapping
        self.ignore_missing_keys = ignore_missing_keys
        self.has_inverse = all(
            transformation.has_inverse for transformation in mapping.values()
        )

    def get_target_space_from_source(
        self, 
        source_space : DictSpace[BDeviceType, BDtypeType, BRNGType]
    ):
        if not isinstance(source_space, DictSpace):
            raise ValueError("Source space must be a DictSpace")
        new_space = source_space
        for key, transformation in self.mapping.items():
            new_space = call_function_on_chained_dict(
                new_space,
                key.split('/'),
                transformation.get_target_space_from_source,
                ignore_missing_keys=self.ignore_missing_keys
            )
        return new_space

    def transform(
        self, 
        source_space: Space,
        data: Union[Mapping[str, Any], BArrayType]
    ) -> Union[Mapping[str, Any], BArrayType]:
        new_data = data
        for key, transformation in self.mapping.items():
            new_data = call_conditioned_function_on_chained_dict(
                source_space,
                new_data,
                key.split('/'),
                transformation.transform,
                ignore_missing_keys=self.ignore_missing_keys
            )
        return new_data

    def direction_inverse(
        self,
        source_space = None,
    ) -> Optional["DictTransformation"]:
        if not self.has_inverse:
            return None

        inverse_mapping = {}
        for key, transformation in self.mapping.items():
            inverse_mapping[key] = transformation.direction_inverse(
                None if source_space is None else get_chained_value(
                    source_space,
                    key.split('/'),
                    ignore_missing_keys=self.ignore_missing_keys
                )
            )
        
        return DictTransformation(
            mapping=inverse_mapping,
            ignore_missing_keys=self.ignore_missing_keys
        )

    def close(self):
        for transformation in self.mapping.values():
            transformation.close()