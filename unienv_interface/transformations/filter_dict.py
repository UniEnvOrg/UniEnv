from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import Space, DictSpace
from typing import Union, Any, Optional, Dict, Set, Iterable, List, Mapping
from unienv_interface.space import DictSpace
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.transformations import serialization_utils as tsu
import copy


def exclude_chained_key_in_dict(
    d : Union[Dict[str, Any], DictSpace[BDeviceType, BDtypeType, BRNGType]],
    chained_key : List[str],
    /,
    ignore_missing_keys : bool = False,
) -> Union[Dict[str, Any], DictSpace[BDeviceType, BDtypeType, BRNGType]]:
    if len(chained_key) == 0:
        return d

    new_dict = d
    first_key = chained_key[0]
    if len(chained_key) == 1:
        if first_key in new_dict.keys():
            new_dict = copy.copy(d)
            del new_dict[first_key]
        elif not ignore_missing_keys:
            raise KeyError(f"Key '{first_key}' not found in the source dictionary.")
    else:
        if first_key in new_dict.keys():
            new_dict = copy.copy(d)
            new_dict[first_key] = exclude_chained_key_in_dict(
                new_dict[first_key],
                chained_key[1:],
                ignore_missing_keys=ignore_missing_keys
            )
        elif not ignore_missing_keys:
            raise KeyError(f"Key '{first_key}' not found in the source dictionary.")
    return new_dict

def include_chained_key_in_dict(
    d : Union[Dict[str, Any], DictSpace[BDeviceType, BDtypeType, BRNGType]],
    chained_key : List[str],
    *,
    target_d : Optional[Union[Dict[str, Any], DictSpace[BDeviceType, BDtypeType, BRNGType]]] = None,
    ignore_missing_keys : bool = False,
    ) -> Union[Dict[str, Any], DictSpace[BDeviceType, BDtypeType, BRNGType]]:
    if len(chained_key) == 0:
        return d

    if target_d is None:
        target_d = DictSpace(d.backend, {}, device=d.device) if isinstance(d, DictSpace) else dict()

    first_key = chained_key[0]

    if first_key not in d.keys():
        if ignore_missing_keys:
            return target_d
        else:
            raise KeyError(f"Key '{first_key}' not found in the source dictionary.")
    
    first_value = d[first_key]
    target_d[first_key] = include_chained_key_in_dict(
        first_value,
        chained_key[1:],
        target_d=None,
        ignore_missing_keys=ignore_missing_keys
    )
    
    return target_d

class DictIncludeKeyTransformation(DataTransformation):
    has_inverse = False
    def __init__(
        self,
        enabled_keys : Iterable[str],
        *,
        ignore_missing_keys: bool = False,
        nested_separator : str = '/'
    ):
        self.nested_separator = nested_separator
        self.enabled_keys = enabled_keys
        self.ignore_missing_keys = ignore_missing_keys

    @property
    def enabled_keys(self) -> Set[str]:
        return self._enabled_keys
    
    @enabled_keys.setter
    def enabled_keys(self, value: Iterable[str]):
        enabled_keys = set(value)
        self._enabled_keys = enabled_keys
        # Compute the list of chained keys
        self._chained_keys : List[List[str]] = []
        for key in enabled_keys:
            self._chained_keys.append(key.split(self.nested_separator))

    def get_target_space_from_source(self, source_space):
        new_space = None
        for chained_key in self._chained_keys:
            try:
                new_space = include_chained_key_in_dict(
                    source_space,
                    chained_key,
                    target_d=new_space,
                    ignore_missing_keys=self.ignore_missing_keys
                )
            except KeyError as e:
                raise ValueError(*e.args)
        return new_space

    def transform(self, source_space, data):
        new_data = None
        for chained_key in self._chained_keys:
            try:
                new_data = include_chained_key_in_dict(
                    data,
                    chained_key,
                    target_d=new_data,
                    ignore_missing_keys=self.ignore_missing_keys
                )
            except KeyError as e:
                raise ValueError(*e.args)
        return new_data

    def serialize(
        self,
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> Dict[str, Any]:
        return {
            "enabled_keys": list(self._enabled_keys),
            "nested_separator": self.nested_separator,
            "ignore_missing_keys": self.ignore_missing_keys,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> "DictIncludeKeyTransformation":
        return cls(
            enabled_keys=json_data["enabled_keys"],
            nested_separator=json_data.get("nested_separator", "/"),
            ignore_missing_keys=json_data.get("ignore_missing_keys", False),
        )


class DictExcludeKeyTransformation(DataTransformation):
    has_inverse = False
    def __init__(
        self,
        excluded_keys : Iterable[str],
        *,
        ignore_missing_keys: bool = False,
        nested_separator : str = '/'
    ):
        self.nested_separator = nested_separator
        self.excluded_keys = excluded_keys
        self.ignore_missing_keys = ignore_missing_keys
    
    @property
    def excluded_keys(self) -> Set[str]:
        return self._excluded_keys
    
    @excluded_keys.setter
    def excluded_keys(self, value: Iterable[str]):
        excluded_keys = set(value)
        self._excluded_keys = excluded_keys
        # Compute the list of chained keys
        self._chained_keys : List[List[str]] = []
        for key in excluded_keys:
            self._chained_keys.append(key.split(self.nested_separator))
        
    def get_target_space_from_source(self, source_space):
        new_space = source_space
        for chained_key in self._chained_keys:
            try:
                new_space = exclude_chained_key_in_dict(
                    new_space,
                    chained_key,
                    ignore_missing_keys=self.ignore_missing_keys
                )
            except KeyError as e:
                raise ValueError(*e.args)
        return new_space
    
    def transform(self, source_space, data):
        new_data = data
        for chained_key in self._chained_keys:
            try:
                new_data = exclude_chained_key_in_dict(
                    new_data,
                    chained_key,
                    ignore_missing_keys=self.ignore_missing_keys,
                )
            except KeyError as e:
                raise ValueError(*e.args)
        return new_data

    def serialize(
        self,
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> Dict[str, Any]:
        return {
            "excluded_keys": list(self._excluded_keys),
            "nested_separator": self.nested_separator,
            "ignore_missing_keys": self.ignore_missing_keys,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> "DictExcludeKeyTransformation":
        return cls(
            excluded_keys=json_data["excluded_keys"],
            nested_separator=json_data.get("nested_separator", "/"),
            ignore_missing_keys=json_data.get("ignore_missing_keys", False),
        )
