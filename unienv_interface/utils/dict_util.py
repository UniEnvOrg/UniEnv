from typing import Any, Mapping, List, Sequence, Union, Dict
from unienv_interface.space import DictSpace

__all__ = [
    "flatten_keys_in_mapping",
    "unflatten_keys_in_mapping",
    "nested_get",
]

def _items(d : Union[Mapping[str, Any], DictSpace]):
    # DictSpace is not a Mapping and has no items(); iterate its sub-spaces.
    return d.spaces.items() if isinstance(d, DictSpace) else d.items()

def flatten_keys_in_mapping(
    d : Union[Mapping[str, Any], DictSpace],
    nested_separator : str = '/'
) -> Union[Mapping[str, Any], DictSpace]:
    """
    Flattens the keys of a nested dictionary or DictSpace. For example, {'a': {'b': 1}} will be flattened to {'a.b': 1}.
    Input:
        d : Union[Mapping[str, Any], DictSpace]
    Output:
        Union[Mapping[str, Any], DictSpace]
    """
    result : Dict[str, Any] = {}
    for key, value in _items(d):
        if isinstance(value, (DictSpace, Mapping)):
            for sub_key, sub_value in _items(flatten_keys_in_mapping(value, nested_separator=nested_separator)):
                result[f'{key}{nested_separator}{sub_key}'] = sub_value
        else:
            result[key] = value
    if isinstance(d, DictSpace):
        result = DictSpace(
            d.backend,
            result,
            d.device,
            metadata=d.metadata,
        )
    return result

def unflatten_keys_in_mapping(
    d : Union[Mapping[str, Any], DictSpace],
    nested_separator : str = '/'
) -> Union[Mapping[str, Any], DictSpace]:
    """
    Unflattens the keys of a dictionary or DictSpace that were flattened using the flatten_keys function. For example, {'a.b': 1} will be unflattened to {'a': {'b': 1}}.
    Input:
        d : Union[Mapping[str, Any], DictSpace]
    Output:
        Union[Mapping[str, Any], DictSpace]
    """
    result : Dict[str, Any] = {}
    all_firstlevel_keys = set()
    for key in d.keys():
        firstlevel_key = key.split(nested_separator)[0]
        all_firstlevel_keys.add(firstlevel_key)
    for firstlevel_key in all_firstlevel_keys:
        sub_dict : Dict[str, Any] = {}
        if firstlevel_key in d.keys():
            result[firstlevel_key] = d[firstlevel_key]
        else:
            for key, value in _items(d):
                if key.startswith(f'{firstlevel_key}{nested_separator}'):
                    sub_key = key[len(firstlevel_key) + len(nested_separator):]
                    sub_dict[sub_key] = value
            result[firstlevel_key] = unflatten_keys_in_mapping(sub_dict, nested_separator=nested_separator)
    if isinstance(d, DictSpace):
        result = DictSpace(
            d.backend,
            result,
            d.device,
            metadata=d.metadata,
        )
    return result

def nested_get(
    d : Union[Mapping[str, Any], DictSpace],
    chained_key : Sequence[str]
) -> Any:
    if len(chained_key) == 0:
        return d
    else:
        return nested_get(d[chained_key[0]], chained_key[1:])