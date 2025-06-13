from typing import Any, Tuple, Union, Optional, List, Dict, Type, TypeVar, Generic
from unienv_data.base import BatchBase, BatchT, SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType, BatchSampler
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu

class StepSampler(
    BatchSampler[
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        data : BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType],
        batch_size : int,
        seed : Optional[int] = None,
        device : Optional[BDeviceType] = None
    ):
        assert batch_size > 0, "Batch size must be a positive integer"
        self.data = data
        self.batch_size = batch_size
        self._device = device
        self.sampled_space = sbu.batch_space(
            self.data.single_space,
            batch_size
        )
        self.sampled_space_flat = sfu.flatten_space(self.sampled_space, start_dim=1)
        self.data_rng = self.backend.random.random_number_generator(
            seed,
            device=data.device
        )
        if device is not None:
            self.sampled_space = self.sampled_space.to(device=device)

    @property
    def sampled_metadata_space(self) -> Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]:
        return sbu.batch_space(
            self.data.single_metadata_space,
            self.batch_size
        ) if self.data.single_metadata_space is not None else None

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.data.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self._device or self.data.device

    def get_flat_at(self, idx):
        dat = self.data.get_flattened_at(idx)
        if self._device is not None:
            dat = self.backend.to_device(dat, self._device)
        return dat

    def get_flat_at_with_metadata(self, idx):
        dat, metadata = self.data.get_flattened_at_with_metadata(idx)
        if self._device is not None:
            dat = self.backend.to_device(dat, self._device)
        if metadata is not None and self._device is not None:
            metadata = self.backend.map_fn_over_arrays(metadata, lambda x: self.backend.to_device(x, self._device))
        return dat, metadata

    def get_at(self, idx):
        dat = self.data.get_at(idx)
        if self._device is not None:
            dat = self.sampled_space.from_same_backend(dat)
        return dat

    def get_at_with_metadata(self, idx):
        dat, metadata = self.data.get_at_with_metadata(idx)
        if self._device is not None:
            dat = self.sampled_space.from_same_backend(dat)
        if metadata is not None and self._device is not None:
            metadata = self.backend.map_fn_over_arrays(metadata, lambda x: self.backend.to_device(x, self._device))
        return dat, metadata