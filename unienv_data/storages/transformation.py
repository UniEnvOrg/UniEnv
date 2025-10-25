from importlib import metadata
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable, Type

from unienv_interface.space import Space, BoxSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.utils.symbol_util import *
from unienv_interface.transformations import DataTransformation

from unienv_data.base import SpaceStorage, BatchT

import numpy as np
import os
import json
import pickle

class TransformedStorage(SpaceStorage[
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
        data_transformation : DataTransformation,
        capacity : Optional[int] = None,
        cache_path : Optional[str] = None,
        **kwargs
    ) -> "TransformedStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        assert data_transformation.has_inverse, "To transform storages (potentially to save space), you need to use inversible data transformations"
        transformed_space = data_transformation.get_target_space_from_source(single_instance_space)
        inner_storage_path = "transformed_inner_storage" + (inner_storage_cls.single_file_ext or "")

        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)

        inner_storage = inner_storage_cls.create(
            transformed_space,
            *args,
            cache_path=None if cache_path is None else os.path.join(cache_path, inner_storage_path),
            capacity=capacity,
            **kwargs
        )
        return TransformedStorage(
            single_instance_space,
            data_transformation,
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
    ) -> "TransformedStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        metadata_path = os.path.join(path, "transformed_metadata.json")
        assert os.path.exists(metadata_path), f"Metadata file {metadata_path} does not exist"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        assert metadata["storage_type"] == cls.__name__, \
            f"Expected storage type {cls.__name__}, but found {metadata['storage_type']}"
        
        data_transform_path = os.path.join(path, "data_transformation.pkl")
        with open(data_transform_path, "rb") as f:
            data_transform = pickle.load(f)
        
        assert isinstance(data_transform, DataTransformation)
        transformed_space = data_transform.get_target_space_from_source(single_instance_space)

        inner_storage_cls : Type[SpaceStorage] = get_class_from_full_name(metadata["inner_storage_type"])
        inner_storage_path = metadata["inner_storage_path"]
        inner_storage = inner_storage_cls.load_from(
            os.path.join(path, inner_storage_path),
            transformed_space,
            capacity=capacity,
            read_only=read_only,
            **kwargs
        )
        return TransformedStorage(
            single_instance_space,
            data_transform,
            inner_storage,
            inner_storage_path,
        )

    # ========== Instance Implementations ==========
    single_file_ext = None

    def __init__(
        self,
        single_instance_space: Space[Any, BDeviceType, BDtypeType, BRNGType],
        data_transformation : DataTransformation,
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
        transformed_space = data_transformation.get_target_space_from_source(single_instance_space)

        assert inner_storage.backend == transformed_space.backend, \
            f"Inner storage backend {inner_storage.backend} does not match single instance space backend {single_instance_space.backend}"
        assert inner_storage.device == transformed_space.device, \
            f"Inner storage device {inner_storage.device} does not match single instance space device {single_instance_space.device}"
        assert inner_storage.single_instance_space == transformed_space

        self._transformed_space = transformed_space
        self._batched_transformed_space = sbu.batch_space(transformed_space, 1)
        self._batched_instance_space = sbu.batch_space(single_instance_space, 1)
        self.inner_storage = inner_storage
        self.inner_storage_path = inner_storage_path
        self.data_transformation = data_transformation
        self.inv_data_transformation = data_transformation.direction_inverse(single_instance_space)
        
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
        dat = self.get(index)
        if isinstance(index, int):
            return sfu.flatten_data(self.single_instance_space, dat)
        else:
            return sfu.flatten_data(self.single_instance_space, dat, start_dim=1)

    def get(self, index):
        result = self.inner_storage.get(index)
        result = self.inv_data_transformation.transform(
            self._transformed_space if isinstance(index, int) else self._batched_transformed_space,
            result
        )
        return result
    
    def set_flattened(self, index, value):
        if isinstance(index, int):
            set_value = sfu.unflatten_data(self.single_instance_space, value)
        else:
            set_value = sfu.unflatten_data(self._batched_instance_space, value)
        self.set(index, set_value)

    def set(self, index, value):
        transformed_value = self.data_transformation.transform(
            self.single_instance_space if isinstance(index, int) else self._batched_instance_space,
            value
        )
        self.inner_storage.set(index, transformed_value)

    def clear(self):
        self.inner_storage.clear()

    def dumps(self, path):
        metadata = {
            "storage_type": __class__.__name__,
            "inner_storage_type": get_full_class_name(type(self.inner_storage)),
            "inner_storage_path": self.inner_storage_path,
            "transformation": get_full_class_name(type(self.data_transformation))
        }
        self.inner_storage.dumps(os.path.join(path, self.inner_storage_path))
        with open(os.path.join(path, "transformed_metadata.json"), "w") as f:
            json.dump(metadata, f)
        with open(os.path.join(path, "data_transformation.pkl"), "wb") as f:
            pickle.dump(self.data_transformation, f)

    def close(self):
        self.inner_storage.close()
