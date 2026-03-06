from typing import Union, Any, Optional, Mapping, List, Callable, Dict

from unienv_interface.space import Space, DictSpace
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.transformations import serialization_utils as tsu

import copy
from .transformation import DataTransformation

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


def get_mapping_source_space(
    source_space: Optional[Space],
    mapping_key: str,
    nested_separator: str,
    ignore_missing_keys: bool,
) -> Optional[Space]:
    if source_space is None:
        return None
    return get_chained_value(
        source_space,
        mapping_key.split(nested_separator),
        ignore_missing_keys=ignore_missing_keys,
    )

class DictTransformation(DataTransformation):
    def __init__(
        self,
        mapping: Dict[str, DataTransformation],
        ignore_missing_keys : bool = False,
        nested_separator : str = '/'
    ):
        self.mapping = mapping
        self.ignore_missing_keys = ignore_missing_keys
        self.nested_separator = nested_separator
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
                key.split(self.nested_separator),
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
                key.split(self.nested_separator),
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
            if source_space is not None:
                current_source = get_chained_value(
                    source_space,
                    key.split(self.nested_separator),
                    ignore_missing_keys=self.ignore_missing_keys
                )
                if current_source is None:
                    continue
            else:
                current_source = None
            inverse_mapping[key] = transformation.direction_inverse(
                current_source
            )
        
        return DictTransformation(
            mapping=inverse_mapping,
            ignore_missing_keys=self.ignore_missing_keys,
            nested_separator=self.nested_separator
        )

    def close(self):
        for transformation in self.mapping.values():
            transformation.close()

    def serialize(
        self,
        source_space: Optional[Space] = None,
    ) -> Dict[str, Any]:
        return {
            "mapping": {
                key: tsu.transformation_to_json(
                    transformation,
                    source_space=get_mapping_source_space(
                        source_space,
                        key,
                        self.nested_separator,
                        self.ignore_missing_keys,
                    ),
                )
                for key, transformation in self.mapping.items()
            },
            "ignore_missing_keys": self.ignore_missing_keys,
            "nested_separator": self.nested_separator,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space] = None,
    ) -> "DictTransformation":
        ignore_missing_keys = json_data.get("ignore_missing_keys", False)
        nested_separator = json_data.get("nested_separator", "/")
        return cls(
            mapping={
                key: tsu.json_to_transformation(
                    transformation_data,
                    source_space=get_mapping_source_space(
                        source_space,
                        key,
                        nested_separator,
                        ignore_missing_keys,
                    ),
                )
                for key, transformation_data in json_data["mapping"].items()
            },
            ignore_missing_keys=ignore_missing_keys,
            nested_separator=nested_separator,
        )
