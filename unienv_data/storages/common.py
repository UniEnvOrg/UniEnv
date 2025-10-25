from importlib import metadata
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable, Type

from unienv_interface.space import Space, BoxSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.utils.symbol_util import *

from unienv_data.base import SpaceStorage, BatchT

import numpy as np
import os
import json

class FlattenedStorage(SpaceStorage[
    BatchT,
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
        inner_storage_cls : Type[SpaceStorage[BArrayType, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        *args,
        capacity : Optional[int] = None,
        cache_path : Optional[str] = None,
        **kwargs
    ) -> "FlattenedStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        flattened_space = sfu.flatten_space(single_instance_space)
        inner_storage_path = "inner_storage" + (inner_storage_cls.single_file_ext or "")

        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)

        inner_storage = inner_storage_cls.create(
            flattened_space,
            *args,
            cache_path=None if cache_path is None else os.path.join(cache_path, inner_storage_path),
            capacity=capacity,
            **kwargs
        )
        return FlattenedStorage(
            single_instance_space,
            inner_storage,
            inner_storage_path,
        )

    @classmethod
    def load_from(
        cls,
        path : Union[str, os.PathLike],
        single_instance_space : Space[Any, BDeviceType, BDtypeType, BRNGType],
        *,
        capacity : Optional[int] = None,
        read_only : bool = True,
        **kwargs
    ) -> "FlattenedStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        metadata_path = os.path.join(path, "flattened_metadata.json")
        assert os.path.exists(metadata_path), f"Metadata file {metadata_path} does not exist"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        assert metadata["storage_type"] == cls.__name__, \
            f"Expected storage type {cls.__name__}, but found {metadata['storage_type']}"
        inner_storage_cls : Type[SpaceStorage] = get_class_from_full_name(metadata["inner_storage_type"])
        inner_storage_path = metadata["inner_storage_path"]
        flattened_space = sfu.flatten_space(single_instance_space)
        inner_storage = inner_storage_cls.load_from(
            os.path.join(path, inner_storage_path),
            flattened_space,
            capacity=capacity,
            read_only=read_only,
            **kwargs
        )
        return FlattenedStorage(
            single_instance_space,
            inner_storage,
            inner_storage_path,
        )

    # ========== Instance Implementations ==========
    single_file_ext = None

    def __init__(
        self,
        single_instance_space: Space[Any, BDeviceType, BDtypeType, BRNGType],
        inner_storage : SpaceStorage[
            BArrayType,
            BArrayType,
            BDeviceType,
            BDtypeType,
            BRNGType,
        ],
        inner_storage_path : Union[str, os.PathLike],
    ):
        super().__init__(single_instance_space)
        assert inner_storage.backend == single_instance_space.backend, \
            f"Inner storage backend {inner_storage.backend} does not match single instance space backend {single_instance_space.backend}"
        assert inner_storage.device == single_instance_space.device, \
            f"Inner storage device {inner_storage.device} does not match single instance space device {single_instance_space.device}"
        assert sfu.flatten_space(single_instance_space) == inner_storage.single_instance_space, \
            f"Inner storage single instance space {inner_storage.single_instance_space} does not match flattened space of single instance space {sfu.flatten_space(single_instance_space)}"

        self._batched_instance_space = sbu.batch_space(single_instance_space, 1)
        self.inner_storage = inner_storage
        self.inner_storage_path = inner_storage_path
        
    @property
    def capacity(self) -> Optional[int]:
        return self.inner_storage.capacity
    
    def extend_length(self, length):
        self.inner_storage.extend_length(length)

    def shrink_length(self, length):
        self.inner_storage.shrink_length(length)
    
    def __len__(self):
        return len(self.inner_storage)
    
    def get_flattened(self, index):
        return self.inner_storage.get(index)

    def get(self, index):
        result = self.inner_storage.get(index)
        if isinstance(index, int):
            result = sfu.unflatten_data(self.single_instance_space, result)
        else:
            result = sfu.unflatten_data(self._batched_instance_space, result, start_dim=1)
        return result
    
    def set_flattened(self, index, value):
        self.inner_storage.set(index, value)

    def set(self, index, value):
        if isinstance(index, int):
            value = sfu.flatten_data(self.single_instance_space, value)
        else:
            value = sfu.flatten_data(self._batched_instance_space, value, start_dim=1)
        self.inner_storage.set(index, value)

    def clear(self):
        self.inner_storage.clear()

    def dumps(self, path):
        metadata = {
            "storage_type": __class__.__name__,
            "inner_storage_type": get_full_class_name(type(self.inner_storage)),
            "inner_storage_path": self.inner_storage_path,
        }
        self.inner_storage.dumps(os.path.join(path, self.inner_storage_path))
        with open(os.path.join(path, "flattened_metadata.json"), "w") as f:
            json.dump(metadata, f)

    def close(self):
        self.inner_storage.close()
