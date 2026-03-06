from typing import Union, Any, Optional, Mapping, List, Dict

from unienv_interface.space import Space, DictSpace
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType

from .transformation import DataTransformation


def flatten_nested_mapping(
    data: Mapping[str, Any],
    nested_separator: str,
) -> Dict[str, Any]:
    flattened_data: Dict[str, Any] = {}

    def _flatten(current_data: Mapping[str, Any], parent_key: Optional[str] = None) -> None:
        for key, value in current_data.items():
            if nested_separator in key:
                raise ValueError(
                    f"Cannot flatten key '{key}' because it already contains separator '{nested_separator}'."
                )
            new_key = key if parent_key is None else f"{parent_key}{nested_separator}{key}"
            if isinstance(value, DictSpace):
                _flatten(value.spaces, new_key)
            elif isinstance(value, Mapping):
                _flatten(value, new_key)
            else:
                flattened_data[new_key] = value

    _flatten(data)
    return flattened_data


def _insert_unflattened_value(
    target_data: Dict[str, Any],
    chained_key: List[str],
    value: Any,
    original_key: str,
) -> None:
    current_data = target_data
    for key in chained_key[:-1]:
        if key not in current_data:
            current_data[key] = {}
        elif not isinstance(current_data[key], dict):
            raise ValueError(
                f"Cannot unflatten key '{original_key}' because '{key}' is already mapped to a non-dict value."
            )
        current_data = current_data[key]

    leaf_key = chained_key[-1]
    if leaf_key in current_data:
        raise ValueError(
            f"Cannot unflatten key '{original_key}' because '{leaf_key}' already exists."
        )
    current_data[leaf_key] = value


def unflatten_mapping(
    data: Mapping[str, Any],
    nested_separator: str,
) -> Dict[str, Any]:
    unflattened_data: Dict[str, Any] = {}
    for key, value in data.items():
        chained_key = key.split(nested_separator)
        _insert_unflattened_value(
            unflattened_data,
            chained_key,
            value,
            key,
        )
    return unflattened_data


class FlattenDictTransformation(DataTransformation):
    has_inverse = True

    def __init__(
        self,
        nested_separator: str = "/",
    ):
        self.nested_separator = nested_separator

    def get_target_space_from_source(
        self,
        source_space: DictSpace[BDeviceType, BDtypeType, BRNGType],
    ) -> DictSpace[BDeviceType, BDtypeType, BRNGType]:
        if not isinstance(source_space, DictSpace):
            raise ValueError("Source space must be a DictSpace")
        return DictSpace(
            source_space.backend,
            flatten_nested_mapping(source_space.spaces, self.nested_separator),
            device=source_space.device,
        )

    def transform(
        self,
        source_space: Space,
        data: Union[Mapping[str, Any], BArrayType],
    ) -> Union[Mapping[str, Any], BArrayType]:
        if not isinstance(source_space, DictSpace):
            raise ValueError("Source space must be a DictSpace")
        if not isinstance(data, Mapping):
            raise ValueError("Data must be a mapping for FlattenDictTransformation")
        return flatten_nested_mapping(data, self.nested_separator)

    def direction_inverse(
        self,
        source_space: Optional[Space] = None,
    ) -> "UnflattenDictTransformation":
        return UnflattenDictTransformation(nested_separator=self.nested_separator)

    def serialize(
        self,
        source_space: Optional[Space] = None,
    ) -> Dict[str, Any]:
        return {
            "nested_separator": self.nested_separator,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space] = None,
    ) -> "FlattenDictTransformation":
        return cls(
            nested_separator=json_data.get("nested_separator", "/"),
        )


class UnflattenDictTransformation(DataTransformation):
    has_inverse = True

    def __init__(
        self,
        nested_separator: str = "/",
    ):
        self.nested_separator = nested_separator

    def get_target_space_from_source(
        self,
        source_space: DictSpace[BDeviceType, BDtypeType, BRNGType],
    ) -> DictSpace[BDeviceType, BDtypeType, BRNGType]:
        if not isinstance(source_space, DictSpace):
            raise ValueError("Source space must be a DictSpace")
        return DictSpace(
            source_space.backend,
            unflatten_mapping(source_space.spaces, self.nested_separator),
            device=source_space.device,
        )

    def transform(
        self,
        source_space: Space,
        data: Union[Mapping[str, Any], BArrayType],
    ) -> Union[Mapping[str, Any], BArrayType]:
        if not isinstance(source_space, DictSpace):
            raise ValueError("Source space must be a DictSpace")
        if not isinstance(data, Mapping):
            raise ValueError("Data must be a mapping for UnflattenDictTransformation")
        return unflatten_mapping(data, self.nested_separator)

    def direction_inverse(
        self,
        source_space: Optional[Space] = None,
    ) -> "FlattenDictTransformation":
        return FlattenDictTransformation(nested_separator=self.nested_separator)

    def serialize(
        self,
        source_space: Optional[Space] = None,
    ) -> Dict[str, Any]:
        return {
            "nested_separator": self.nested_separator,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space] = None,
    ) -> "UnflattenDictTransformation":
        return cls(
            nested_separator=json_data.get("nested_separator", "/"),
        )
