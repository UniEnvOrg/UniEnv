from importlib import metadata
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable, Type, Callable, Mapping

from unienv_interface.space import Space, DictSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils.symbol_util import *

from unienv_data.base import SpaceStorage, BatchT

import numpy as np
import os
import json


def _merge_nested_mappings(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Merge secondary into primary without clobbering explicitly matched keys."""
    merged: Dict[str, Any] = dict(primary)
    for merge_key, merge_value in secondary.items():
        if (
            merge_key in merged
            and isinstance(merged[merge_key], Mapping)
            and isinstance(merge_value, Mapping)
        ):
            merged[merge_key] = _merge_nested_mappings(merged[merge_key], merge_value)
        elif merge_key not in merged:
            merged[merge_key] = merge_value
    return merged

def map_transform(
    data : Dict[str, Any],
    value_map : Dict[str, Any],
    fn : Callable[[str, Any, Any], Any], # (str, data, value_map) -> transformed data
    prefix : str = "",
    nested_separator : str = '/',
) -> Tuple[
    Dict[str, Any], # Transformed data
    Dict[str, Any], # Residual data
]:
    transformed_data = {}
    residual_data = {}
    for key, value in data.items() if isinstance(data, Mapping) else data.spaces.items():
        full_key = prefix + key
        if full_key in value_map:
            transformed_data[key] = fn(full_key, value, value_map[full_key])
        elif isinstance(value, Mapping) or isinstance(value, DictSpace):
            sub_transformed, sub_residual = map_transform(
                value,
                value_map,
                fn,
                prefix=full_key + nested_separator,
            )
            if len(sub_transformed) > 0:
                transformed_data[key] = sub_transformed
            if len(sub_residual) > 0:
                residual_data[key] = sub_residual
        else:
            residual_data[key] = value
    if len(residual_data) > 0 and (prefix + "*") in value_map:
        residual_transformed = fn(prefix + "*", residual_data, value_map[prefix + "*"])
        if isinstance(residual_transformed, Mapping) or isinstance(residual_transformed, DictSpace):
            for key, value in residual_transformed.items():
                if key in transformed_data and isinstance(transformed_data[key], Mapping) and isinstance(value, Mapping):
                    transformed_data[key] = _merge_nested_mappings(transformed_data[key], value)
                elif key not in transformed_data:
                    transformed_data[key] = value
        residual_data = {}
    return transformed_data, residual_data

def get_chained_residual_space(
    space : DictSpace[BDeviceType, BDtypeType, BRNGType],
    all_keys : List[str],
    prefix : str = "",
    nested_separator : str = '/',
) -> Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]:
    residual_spaces = {}

    if len(residual_spaces) > 0 and (prefix + "*") in all_keys:
        return DictSpace(
            space.backend,
            {},
            device=space.device,
        )

    for key, subspace in space.spaces.items():
        full_key = prefix + key
        if full_key in all_keys:
            continue
        elif isinstance(subspace, DictSpace):
            sub_residual = get_chained_residual_space(
                subspace,
                all_keys,
                prefix=full_key + nested_separator,
            )
            if sub_residual is not None and len(sub_residual.spaces) > 0:
                residual_spaces[key] = sub_residual
        else:
            residual_spaces[key] = subspace
    
    if len(residual_spaces) == 0:
        return None

    return DictSpace(
        space.backend,
        residual_spaces,
        device=space.device,
    )

def get_chained_space(
    space : DictSpace[BDeviceType, BDtypeType, BRNGType],
    key_chain : str,
    all_keys : List[str],
    nested_separator : str = '/',
) -> Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]]:
    if key_chain.endswith("*"):
        prefix = key_chain[:-1]
        subspace = get_chained_residual_space(
            get_chained_space(
                space,
                prefix,
                all_keys,
            ) if len(prefix) > 0 else space,
            [key for key in all_keys if key != key_chain],
            prefix=prefix,
        )
        return subspace
    key_chain = key_chain.split(nested_separator)
    current_space : Space[Any, BDeviceType, BDtypeType, BRNGType]
    current_space = space
    for key in key_chain:
        if len(key) == 0:
            continue
        if not isinstance(current_space, DictSpace) or key not in current_space.spaces:
            return None
        current_space = current_space.spaces[key]
    return current_space

class DictStorage(SpaceStorage[
    Dict[str, Any],
    BArrayType,
    BDeviceType,
    BDtypeType,
    BRNGType,
]):
    # ========== Class Attributes ==========
    @classmethod
    def create(
        cls,
        single_instance_space: Space[Any, BDeviceType, BDtypeType, BRNGType],
        storage_cls_map : Dict[
            str,
            Type[SpaceStorage],
        ],
        *args,
        capacity : Optional[int] = None,
        cache_path : Optional[str] = None,
        multiprocessing : bool = False,
        key_kwargs : Dict[str, Any] = {},
        type_kwargs : Dict[Type[SpaceStorage[Any, BArrayType, BDeviceType, BDtypeType, BRNGType]], Dict[str, Any]] = {},
        nested_separator : str = '/',
        **kwargs
    ) -> "DictStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)

        storage_map = {}
        all_keys = list(storage_cls_map.keys())
        for key, sub_storage_cls in storage_cls_map.items():
            sub_storage_path = key.replace(nested_separator, ".").replace("*", "_default") + (sub_storage_cls.single_file_ext or "")
            subspace = get_chained_space(single_instance_space, key, all_keys, nested_separator=nested_separator)
            if subspace is None:
                continue
            sub_kwargs = kwargs.copy()
            if sub_storage_cls in type_kwargs:
                sub_kwargs.update(type_kwargs[sub_storage_cls])
            if key in key_kwargs:
                sub_kwargs.update(key_kwargs[key])
            storage_map[key] = sub_storage_cls.create(
                subspace,
                *args,
                cache_path=None if cache_path is None else os.path.join(cache_path, sub_storage_path),
                capacity=capacity,
                multiprocessing=multiprocessing,
                **sub_kwargs
            )

        return DictStorage(
            single_instance_space,
            storage_map,
            cache_filename=cache_path,
            nested_separator=nested_separator,
        )

    @classmethod
    def load_from(
        cls,
        path : Union[str, os.PathLike],
        single_instance_space : Space[Any, BDeviceType, BDtypeType, BRNGType],
        *,
        capacity : Optional[int] = None,
        read_only : bool = True,
        multiprocessing : bool = False,
        key_kwargs : Dict[str, Any] = {},
        type_kwargs : Dict[Type[SpaceStorage[Any, BArrayType, BDeviceType, BDtypeType, BRNGType]], Dict[str, Any]] = {},
        **kwargs
    ) -> "DictStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        metadata_path = os.path.join(path, "dict_storage_metadata.json")
        assert os.path.exists(metadata_path), f"Metadata file {metadata_path} does not exist"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        assert metadata["storage_type"] == cls.__name__, \
            f"Expected storage type {cls.__name__}, but found {metadata['storage_type']}"
        
        nested_separator = metadata.get("nested_separator", '/')
        storage_map_metadata = metadata["storage_map"]
        storage_map = {}

        all_keys = list(storage_map_metadata.keys())
        for key, storage_meta in storage_map_metadata.items():
            storage_cls : Type[SpaceStorage] = get_class_from_full_name(storage_meta["type"])
            storage_path = storage_meta["path"]
            
            subspace = get_chained_space(single_instance_space, key, all_keys, nested_separator=nested_separator)
            if subspace is None:
                continue

            sub_kwargs = kwargs.copy()
            if storage_cls in type_kwargs:
                sub_kwargs.update(type_kwargs[storage_cls])
            if key in key_kwargs:
                sub_kwargs.update(key_kwargs[key])
            storage_map[key] = storage_cls.load_from(
                os.path.join(path, storage_path),
                subspace,
                capacity=capacity,
                read_only=read_only,
                multiprocessing=multiprocessing,
                **sub_kwargs
            )

        return DictStorage(
            single_instance_space,
            storage_map,
            cache_filename=path,
        )

    # ========== Instance Implementations ==========
    single_file_ext = None

    def __init__(
        self,
        single_instance_space: DictSpace[BDeviceType, BDtypeType, BRNGType],
        storage_map : Dict[
            str,
            SpaceStorage[
                BArrayType,
                BArrayType,
                BDeviceType,
                BDtypeType,
                BRNGType,
            ],
        ],
        cache_filename: Optional[Union[str, os.PathLike]] = None,
        nested_separator : str = '/',
    ):
        assert len(storage_map) > 0, "Storage map cannot be empty"
        first_storage = next(iter(storage_map.values()))
        init_capacity = first_storage.capacity
        init_len = len(first_storage)
        for key, storage in storage_map.items():
            assert storage.capacity == init_capacity, \
                f"All storages must have the same capacity, but storage {key} has capacity {storage.capacity} while first storage has capacity {init_capacity}"
            assert len(storage) == init_len, \
                f"All storages must have the same length, but storage {key} has length {len(storage)} while first storage has length {init_len}"

        super().__init__(single_instance_space)
        self._batched_instance_space = sbu.batch_space(single_instance_space, 1)
        self.storage_map = storage_map
        self.nested_separator = nested_separator
        self._cache_filename = cache_filename if all(
            storage.cache_filename is not None for storage in storage_map.values()
        ) else None

    @property
    def cache_filename(self) -> Optional[Union[str, os.PathLike]]:
        return self._cache_filename

    @property
    def is_mutable(self) -> bool:
        return all(storage.is_mutable for storage in self.storage_map.values())

    @property
    def is_multiprocessing_safe(self) -> bool:
        return all(storage.is_multiprocessing_safe for storage in self.storage_map.values())

    @property
    def capacity(self) -> Optional[int]:
        return next(iter(self.storage_map.values())).capacity
    
    def extend_length(self, length):
        for storage in self.storage_map.values():
            storage.extend_length(length)

    def shrink_length(self, length):
        for storage in self.storage_map.values():
            storage.shrink_length(length)

    def __len__(self):
        return len(next(iter(self.storage_map.values())))

    def get_flattened(self, index):
        unflat_data = self.get(index)
        if isinstance(index, int):
            flat_data = sfu.flatten_data(self.single_instance_space, unflat_data)
        else:
            flat_data = sfu.flatten_data(self._batched_instance_space, unflat_data, start_dim=1)
        return flat_data

    def get(self, index):
        result, residual = map_transform(
            self.single_instance_space,
            self.storage_map,
            lambda key, space, storage: storage.get(index)
        )
        assert len(residual) == 0, f"Some spaces do not have corresponding storage: {residual}"
        return result
    
    def set_flattened(self, index, value):
        if isinstance(index, int):
            unflat_data = sfu.unflatten_data(self.single_instance_space, value)
        else:
            unflat_data = sfu.unflatten_data(self._batched_instance_space, value, start_dim=1)
        self.set(index, unflat_data)

    def set(self, index, value):
        _, residual = map_transform(
            value,
            self.storage_map,
            lambda key, data, storage: storage.set(index, data),
            nested_separator=self.nested_separator,
        )
        assert len(residual) == 0, f"Some spaces do not have corresponding storage: {residual}"

    def get_subspace_by_key(
        self,
        key: str,
    ) -> Space[Any, BDeviceType, BDtypeType, BRNGType]:
        return get_chained_space(
            self.single_instance_space,
            key,
            list(self.storage_map.keys()),
            nested_separator=self.nested_separator,
        )

    def clear(self):
        for storage in self.storage_map.values():
            storage.clear()

    def dumps(self, path):
        os.makedirs(path, exist_ok=True)

        storage_map_metadata = {}
        for key, storage in self.storage_map.items():
            sub_storage_path = key.replace(self.nested_separator, ".").replace("*", "_default") + (storage.single_file_ext or "")
            storage_map_metadata[key] = {
                "type": get_full_class_name(type(storage)),
                "path": sub_storage_path,
            }
            storage.dumps(os.path.join(path, sub_storage_path))
        
        metadata = {
            "storage_type": __class__.__name__,
            "storage_map": storage_map_metadata,
            "nested_separator": self.nested_separator,
        }
        with open(os.path.join(path, "dict_storage_metadata.json"), "w") as f:
            json.dump(metadata, f)

    def close(self):
        for storage in self.storage_map.values():
            storage.close()
