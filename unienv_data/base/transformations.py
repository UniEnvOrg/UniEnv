from typing import Optional, Any, Union
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from .common import *
from unienv_interface.transformations.transformation import DataTransformation, TargetDataT, SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT
from unienv_interface.space import Space

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
        ],
        metadata_transformation : Optional[DataTransformation[
            Dict[str, Any], BArrayType, BDeviceType, BDtypeType, BRNGType,
            Dict[str, Any], SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT
        ]] = None
    ):
        self.batch = batch
        self.transformation = transformation
        self.metadata_transformation = metadata_transformation if self.batch.single_metadata_space is not None else None
        super().__init__(
            self.transformation.target_space,
            self.metadata_transformation.target_space if self.metadata_transformation is not None else self.batch.single_metadata_space,
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
    
    def get_flattened_at(self, idx : Union[IndexableType, BArrayType]) -> BArrayType:
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
    
    def get_flattened_at_with_metadata(self, idx : Union[IndexableType, BArrayType]) -> Tuple[BArrayType, Optional[Dict[str, Any]]]:
        dat, metadata = self.get_at_with_metadata(idx)
        if isinstance(idx, int):
            dat = sfu.flatten_data(
                self.single_space,
                dat
            )
        else:
            dat = sfu.flatten_data(
                self._batched_space,
                dat,
                start_dim=1
            )
        
        return dat, metadata

    def set_flattened_at(self, idx : Union[IndexableType, BArrayType], value : BArrayType) -> None:
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

    def get_at(self, idx : Union[IndexableType, BArrayType] = None) -> BatchT:
        source_dat = self.batch.get_at(idx)
        
        if isinstance(idx, int):
            target_dat = self.transformation.transform(
                source_dat
            )
        else:
            target_dat = self.transformation.transform_batched(
                source_dat
            )
        return target_dat

    def get_at_with_metadata(self, idx : Union[IndexableType, BArrayType]) -> Tuple[BatchT, Optional[Dict[str, Any]]]:
        source_dat, metadata = self.batch.get_at_with_metadata(idx)

        if isinstance(idx, int):
            source_dat = self.transformation.transform(
                source_dat
            )
            if self.metadata_transformation is not None and metadata is not None:
                metadata = self.metadata_transformation.transform(
                    metadata
                )
        else:
            source_dat = self.transformation.transform_batched(
                source_dat
            )
            if self.metadata_transformation is not None and metadata is not None:
                metadata = self.metadata_transformation.transform_batched(
                    metadata
                )
        return source_dat, metadata

    def set_at(self, idx : Union[IndexableType, BArrayType], value : BatchT) -> None:
        assert self.transformation.has_inverse, "Cannot set values on a transformed batch without an inverse transformation"
        assert self.batch.is_mutable, "Cannot set values on an immutable batch"
        if isinstance(idx, int):
            source_dat = self.transformation.inverse_transform(
                value
            )
        else:
            source_dat = self.transformation.inverse_transform_batched(
                value
            )
        self.batch.set_at(idx, source_dat)

    def remove_at(self, idx : Union[IndexableType, BArrayType]) -> None:
        return self.batch.remove_at(idx)

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
        ],
        metadata_transformation : Optional[DataTransformation[
            Dict[str, Any], BArrayType, BDeviceType, BDtypeType, BRNGType,
            Dict[str, Any], SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT
        ]] = None
    ):
        self.sampler = sampler
        self.transformation = transformation
        self.metadata_transformation = metadata_transformation if self.sampler.sampled_metadata_space is not None else None

        self.sampled_space = sbu.batch_space(
            self.transformation.target_space,
            self.sampler.batch_size
        )
        self.sampled_space_flat = sfu.flatten_space(
            self.sampled_space, start_dim=1
        )
        self.sampled_metadata_space = sbu.batch_space(
            self.metadata_transformation.target_space,
            self.sampler.batch_size
        ) if self.metadata_transformation is not None else self.sampler.sampled_metadata_space

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
    
    def get_flat_at(self, idx):
        dat = self.get_at(idx)
        if isinstance(idx, int):
            return sfu.flatten_data(
                self.sampled_space,
                dat
            )
        else:
            return sfu.flatten_data(
                self.sampled_space_flat,
                dat,
                start_dim=1
            )
    
    def get_flat_at_with_metadata(self, idx):
        dat, metadata = self.get_at_with_metadata(idx)
        if isinstance(idx, int):
            dat = sfu.flatten_data(
                self.sampled_space,
                dat
            )
        else:
            dat = sfu.flatten_data(
                self.sampled_space_flat,
                dat,
                start_dim=1
            )
        return dat, metadata
    
    def get_at(self, idx):
        dat = self.sampler.get_at(idx)
        if isinstance(idx, int):
            dat = self.transformation.transform(
                dat
            )
        else:
            dat = self.transformation.transform_batched(
                dat
            )
        return dat
    
    def get_at_with_metadata(self, idx):
        dat, metadata = self.sampler.get_at_with_metadata(idx)
        if isinstance(idx, int):
            dat = self.transformation.transform(
                dat
            )
            if self.metadata_transformation is not None and metadata is not None:
                metadata = self.metadata_transformation.transform(
                    metadata
                )
        else:
            dat = self.transformation.transform_batched(
                dat
            )
            if self.metadata_transformation is not None and metadata is not None:
                metadata = self.metadata_transformation.transform_batched(
                    metadata
                )
        return dat, metadata

    def close(self):
        self.transformation.close()
        if self.metadata_transformation is not None:
            self.metadata_transformation.close()
        self.sampler.close()