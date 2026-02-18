from typing import Dict, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence, Mapping, TypeVar
import numpy as np
import copy

from unienv_interface.space import Space, DictSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.serialization import serialize_backend, deserialize_backend
from unienv_interface.utils.symbol_util import *

from unienv_data.base import SpaceStorage, BatchT

import os
import json

WrapperBatchT = TypeVar("WrapperBatchT")
WrapperBArrayT = TypeVar("WrapperBArrayT")
WrapperBDeviceT = TypeVar("WrapperBDeviceT")
WrapperBDtypeT = TypeVar("WrapperBDtypeT")
WrapperBRngT = TypeVar("WrapperBRngT")

def data_to(
    data : Any,
    source_backend : Optional[ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]] = None,
    target_backend : Optional[ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]] = None,
    target_device : Optional[WrapperBDeviceT] = None,
):
    if source_backend.is_backendarray(data):
        if source_backend is not None and target_backend is not None and target_backend != source_backend:
            data = target_backend.from_other_backend(
                source_backend,
                data
            )
        if target_device is not None:
            data = (source_backend or target_backend).to_device(
                data,
                target_device
            )
    elif isinstance(data, Mapping):
        data = {
            key: data_to(value, source_backend, target_backend, target_device)
            for key, value in data.items()
        }
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        data = [
            data_to(value, source_backend, target_backend, target_device)
            for value in data
        ]
        try:
            data = type(data)(data)  # Preserve the type of the original sequence
        except:
            pass
    return data

class ToBackendOrDeviceStorage(
    SpaceStorage[
        WrapperBatchT, WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT
    ],
    Generic[
        WrapperBatchT, WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ]
):
    # ========== Class Implementations ==========
    @classmethod
    def create(
        cls,
        single_instance_space: Space[WrapperBatchT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT],
        inner_storage_cls : Type[SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        *args,
        capacity : Optional[int] = None,
        cache_path : Optional[str] = None,
        multiprocessing : bool = False,
        backend : Optional[ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]] = None,
        device : Optional[BDeviceType] = None,
        inner_storage_kwargs : Dict[str, Any] = {},
        **kwargs
    ) -> "ToBackendOrDeviceStorage[WrapperBatchT, WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT, BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        inner_storage_path = "inner_storage" + (inner_storage_cls.single_file_ext or "")

        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)

        _inner_storage_kwargs = kwargs.copy()
        _inner_storage_kwargs.update(inner_storage_kwargs)
        inner_storage = inner_storage_cls.create(
            single_instance_space.to(
                backend=backend,
                device=device
            ),
            *args,
            cache_path=None if cache_path is None else os.path.join(cache_path, inner_storage_path),
            capacity=capacity,
            multiprocessing=multiprocessing,
            **_inner_storage_kwargs
        )
        if (backend is None or backend == single_instance_space.backend) and (device is None or device == single_instance_space.device):
            return inner_storage

        return ToBackendOrDeviceStorage(
            single_instance_space,
            inner_storage,
            inner_storage_path,
            cache_filename=cache_path,
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
        **kwargs
    ) -> Union[
        "ToBackendOrDeviceStorage[WrapperBatchT, WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT, BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]",
        SpaceStorage[WrapperBatchT, WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]
    ]:
        metadata_path = os.path.join(path, "backend_metadata.json")
        assert os.path.exists(metadata_path), f"Metadata file {metadata_path} does not exist"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        assert metadata["storage_type"] == cls.__name__, \
            f"Expected storage type {cls.__name__}, but found {metadata['storage_type']}"
        inner_storage_cls : Type[SpaceStorage] = get_class_from_full_name(metadata["inner_storage_type"])
        inner_storage_path = metadata["inner_storage_path"]
        inner_backend = deserialize_backend(metadata["inner_backend"])
        inner_device = inner_backend.deserialize_device(metadata["inner_device"])

        if inner_backend != single_instance_space.backend or inner_device != single_instance_space.device:
            inner_space = single_instance_space.to(
                backend=inner_backend if inner_backend != single_instance_space.backend else None,
                device=inner_device if inner_device != single_instance_space.device else None
            )
        else:
            inner_space = single_instance_space
        
        inner_storage = inner_storage_cls.load_from(
            os.path.join(path, inner_storage_path),
            inner_space,
            capacity=capacity,
            read_only=read_only,
            multiprocessing=multiprocessing,
            **kwargs
        )

        if inner_backend == single_instance_space.backend and inner_device == single_instance_space.device:
            return inner_storage
        else:
            return ToBackendOrDeviceStorage(
                single_instance_space,
                inner_storage,
                inner_storage_path,
                cache_filename=path,
            )

    # ======== Instance Implementations ==========
    single_file_ext = None

    def __init__(
        self,
        single_instance_space : Space[WrapperBatchT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT],
        inner_storage : SpaceStorage[
            BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
        ],
        inner_storage_path : Union[str, os.PathLike],
        cache_filename : Optional[Union[str, os.PathLike]] = None,
    ):
        super().__init__(single_instance_space)
        inner_backend = None if inner_storage.backend == single_instance_space.backend else inner_storage.backend
        inner_device = None if inner_storage.device == single_instance_space.device else inner_storage.device
        current_backend = None if inner_storage.backend == single_instance_space.backend else single_instance_space.backend
        current_device = None if inner_storage.device == single_instance_space.device else single_instance_space.device

        self._batched_instance_space = sbu.batch_space(
            single_instance_space,
            1
        )
        self._batched_inner_space = self._batched_instance_space.to(
            backend=inner_backend,
            device=inner_device
        )

        self.inner_storage = inner_storage
        self.inner_storage_path = inner_storage_path
        self._cache_filename = cache_filename

        self.inner_backend = inner_backend
        self.inner_device = inner_device
        self.current_backend = current_backend
        self.current_device = current_device
    
    @property
    def cache_filename(self) -> Optional[Union[str, os.PathLike]]:
        return self._cache_filename
    
    @property
    def is_mutable(self) -> bool:
        return self.inner_storage.is_mutable
    
    @property
    def is_multiprocessing_safe(self) -> bool:
        return self.inner_storage.is_multiprocessing_safe
    
    @property
    def capacity(self) -> Optional[int]:
        return self.inner_storage.capacity
    
    def extend_length(self, length):
        self.inner_storage.extend_length(length)

    def shrink_length(self, length):
        self.inner_storage.shrink_length(length)
    
    def __len__(self):
        return len(self.inner_storage)
    
    def get(self, index):
        if self.backend.is_backendarray(index):
            index = data_to(
                index,
                source_backend=self.backend,
                target_backend=self.inner_backend,
                target_device=self.inner_device
            )
        target_data = self.inner_storage.get(index)
        return data_to(
            target_data,
            source_backend=self.inner_storage.backend,
            target_backend=self.current_backend,
            target_device=self.current_device
        )
    
    def set(self, index, value):
        target_value = data_to(
            value,
            source_backend=self.backend,
            target_backend=self.inner_backend,
            target_device=self.inner_device
        )
        if self.backend.is_backendarray(index):
            index = data_to(
                index,
                source_backend=self.backend,
                target_backend=self.inner_backend,
                target_device=self.inner_device
            )
        self.inner_storage.set(index, target_value)

    def clear(self):
        self.inner_storage.clear()

    def dumps(self, path):
        metadata = {
            "storage_type": __class__.__name__,
            "inner_storage_type": get_full_class_name(type(self.inner_storage)),
            "inner_storage_path": self.inner_storage_path,
            "inner_backend": serialize_backend(self.inner_storage.backend),
            "inner_device": self.inner_storage.backend.serialize_device(self.inner_storage.device),
        }
        self.inner_storage.dumps(os.path.join(path, self.inner_storage_path))
        with open(os.path.join(path, "backend_metadata.json"), "w") as f:
            json.dump(metadata, f)
    
    def close(self):
        self.inner_storage = None