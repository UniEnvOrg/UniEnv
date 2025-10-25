from typing import Dict, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence, Mapping, TypeVar
import numpy as np
import copy

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils import seed_util
from unienv_interface.space import Space, GraphInstance

from unienv_data.base import BatchBase, BatchT

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
    elif isinstance(data, Sequence):
        data = [
            data_to(value, source_backend, target_backend, target_device)
            for value in data
        ]
        try:
            data = type(data)(data)  # Preserve the type of the original sequence
        except:
            pass
    return data

class ToBackendOrDeviceBatch(
    Generic[
        WrapperBatchT, WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ],
    BatchBase[WrapperBatchT, WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]
):
    def __init__(
        self,
        batch : BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType],
        backend : Optional[ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]] = None,
        device : Optional[WrapperBDeviceT] = None,
    ):
        super().__init__(
            batch.single_space.to(backend=backend, device=device),
            single_metadata_space=None if batch.single_metadata_space is None else batch.single_metadata_space.to(backend=backend, device=device),
        )
        self.batch = batch
        self.target_backend = backend
        self.target_device = device
    
    def __len__(self) -> int:
        return len(self.batch)
    
    @property
    def is_mutable(self) -> bool:
        return self.batch.is_mutable
    
    @property
    def backend(self) -> ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]:
        return self.target_backend if self.target_backend is not None else self.batch.backend

    @property
    def device(self) -> Optional[WrapperBDeviceT]:
        return self.target_device if self.target_device is not None else self.batch.device

    def get_flattened_at(self, idx):
        if self.target_backend.is_backendarray(idx):
            idx = data_to(
                idx,
                source_backend=self.target_backend,
                target_backend=self.batch.backend,
                target_device=self.batch.device
            )
        o_data = self.batch.get_flattened_at(idx)
        return data_to(
            o_data,
            source_backend=self.batch.backend,
            target_backend=self.target_backend,
            target_device=self.device
        )

    def get_flattened_at_with_metadata(self, idx):
        if self.target_backend.is_backendarray(idx):
            idx = data_to(
                idx,
                source_backend=self.target_backend,
                target_backend=self.batch.backend,
                target_device=self.batch.device
            )
        o_data, o_metadata = self.batch.get_flattened_at_with_metadata(idx)
        return (
            data_to(
                o_data,
                source_backend=self.batch.backend,
                target_backend=self.target_backend,
                target_device=self.device
            ),
            data_to(
                o_metadata,
                source_backend=self.batch.backend,
                target_backend=self.target_backend,
                target_device=self.device
            ) if o_metadata is not None else None
        )

    def set_flattened_at(self, idx, value):
        assert self.is_mutable, "Batch is not mutable"
        if self.target_backend.is_backendarray(idx):
            idx = data_to(
                idx,
                source_backend=self.target_backend,
                target_backend=self.batch.backend,
                target_device=self.batch.device
            )
        value = data_to(
            value,
            source_backend=self.target_backend,
            target_backend=self.batch.backend,
            target_device=self.batch.device
        )
        self.batch.set_flattened_at(idx, value)
    
    def extend_flattened(self, value):
        assert self.is_mutable, "Batch is not mutable"
        value = data_to(
            value,
            source_backend=self.target_backend,
            target_backend=self.batch.backend,
            target_device=self.batch.device
        )
        self.batch.extend_flattened(value)
    
    def get_at(self, idx):
        if self.target_backend.is_backendarray(idx):
            idx = data_to(
                idx,
                source_backend=self.target_backend,
                target_backend=self.batch.backend,
                target_device=self.batch.device
            )
        o_data = self.batch.get_at(idx)
        return (
            data_to(
                o_data,
                source_backend=self.batch.backend,
                target_backend=self.target_backend,
                target_device=self.device
            )
        )
    
    def get_at_with_metadata(self, idx):
        if self.target_backend.is_backendarray(idx):
            idx = data_to(
                idx,
                source_backend=self.target_backend,
                target_backend=self.batch.backend,
                target_device=self.batch.device
            )
        o_data, o_metadata = self.batch.get_at_with_metadata(idx)
        return (
            data_to(
                o_data,
                source_backend=self.batch.backend,
                target_backend=self.target_backend,
                target_device=self.device
            ),
            data_to(
                o_metadata,
                source_backend=self.batch.backend,
                target_backend=self.target_backend,
                target_device=self.device
            ) if o_metadata is not None else None
        )
    
    def set_at(self, idx, value):
        assert self.is_mutable, "Batch is not mutable"
        if self.target_backend.is_backendarray(idx):
            idx = data_to(
                idx,
                source_backend=self.target_backend,
                target_backend=self.batch.backend,
                target_device=self.batch.device
            )
        o_value = data_to(
            value,
            source_backend=self.target_backend,
            target_backend=self.batch.backend,
            target_device=self.batch.device
        )
        self.batch.set_at(idx, o_value)

    def extend(self, value):
        assert self.is_mutable, "Batch is not mutable"
        o_value = data_to(
            value,
            source_backend=self.target_backend,
            target_backend=self.batch.backend,
            target_device=self.batch.device
        )
        self.batch.extend(o_value)

    def remove_at(self, idx):
        assert self.is_mutable, "Batch is not mutable"
        self.batch.remove_at(idx)
    
    def close(self):
        self.batch.close()
