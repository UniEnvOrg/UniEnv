from typing import Optional, Any, Union
from .common import *
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.transformations.transformation import DataTransformation, TargetDataT, SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT
from unienv_interface.space import Space, flatten_utils as sfu, batch_utils as sbu

class TransformedBatch(
    BatchBase[
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        batch : BatchBase[
            SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT
        ],
        transformation : DataTransformation[
            BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType,
            SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT
        ]
    ):
        self.batch = batch
        self.transformation = transformation
        super().__init__(
            self.transformation.target_space,
        )

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.transformation.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.transformation.device
    
    @property
    def is_mutable(self) -> bool:
        return self.transformation.has_inverse and self.batch.is_mutable

    def __len__(self) -> int:
        return len(self.batch)
    
    def get_flattened_at(self, idx : Optional[Union[int, slice, BArrayType]] = None) -> BArrayType:
        dat = self.get_at(idx)
        if isinstance(idx, int):
            return sfu.flatten_data(
                self.single_space,
                dat
            )
        else:
            return sfu.flatten_data(
                self._batched_space,
                dat,
                start_dim=1
            )
    
    def set_flattened_at(self, idx : Optional[Union[int, slice, BArrayType]], value : BArrayType) -> None:
        if isinstance(idx, int):
            value = sfu.unflatten_data(
                self.single_space,
                value
            )
        else:
            value = sfu.unflatten_data(
                self._batched_space,
                value,
                start_dim=1
            )
        self.set_at(idx, value)
    
    def extend_flattened(self, value : BArrayType) -> None:
        value = sfu.unflatten_data(
            self._batched_space,
            value,
            start_dim=1
        )
        self.extend(value)

    def get_at(self, idx : Optional[Union[int, slice, BArrayType]] = None) -> BatchT:
        source_dat = self.batch.get_at(idx)
        
        if not isinstance(idx, int):
            target_dat = self.transformation.transform(
                source_dat
            )
        else:
            target_dat = self.transformation.transform_batched(
                source_dat
            )
        return target_dat

    def set_at(self, idx : Optional[Union[int, slice, BArrayType]], value : BatchT) -> None:
        assert self.transformation.has_inverse, "Cannot set values on a transformed batch without an inverse transformation"
        assert self.batch.is_mutable, "Cannot set values on an immutable batch"
        if not isinstance(idx, int):
            source_dat = self.transformation.inverse_transform(
                value
            )
        else:
            source_dat = self.transformation.inverse_transform_batched(
                value
            )
        self.batch.set_at(idx, source_dat)

    def extend(self, value : BatchT) -> None:
        assert self.transformation.has_inverse, "Cannot extend values on a transformed batch without an inverse transformation"
        assert self.batch.is_mutable, "Cannot extend values on an immutable batch"
        source_dat = self.transformation.inverse_transform_batched(
            value
        )
        self.batch.extend(source_dat)

    def close(self):
        self.batch.close()
        self.transformation.close()

class TransformedSampler(BatchSampler[
    SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType,
    BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    rng : Optional[SamplerRNGType] = None

    def __init__(
        self,
        sampler : BatchSampler[
            SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT,
            BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
        ],
        transformation : DataTransformation[
            SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType,
            SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT
        ]
    ):
        self.sampler = sampler
        self.transformation = transformation

        self.sampled_space = sbu.batch_space(
            self.transformation.target_space,
            self.sampler.batch_size
        )
        self.sampled_space_flat = sfu.flatten_space(
            self.sampled_space, start_dim=1
        )

    @property
    def batch_size(self) -> int:
        return self.sampler.batch_size
    
    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.transformation.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.transformation.device
    
    @property
    def data(self) -> BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.sampler.data

    @property
    def data_rng(self) -> Optional[BRNGType]:
        return self.sampler.data_rng
    
    @data_rng.setter
    def data_rng(self, value : Optional[BRNGType]) -> None:
        self.sampler.data_rng = value
    
    def sample_flat(self) -> SamplerArrayType:
        return space_flatten_utils.flatten_data(self.sampled_space, self.sample(), start_dim=1)

    def sample(self) -> SamplerBatchT:
        source_sample = self.sampler.sample()
        return self.transformation.transform_batched(source_sample)

    def close(self):
        self.transformation.close()
        self.sampler.close()