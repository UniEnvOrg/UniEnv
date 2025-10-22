from typing import Optional, Any, Union, Tuple, Dict
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.transformations.transformation import DataTransformation, TargetDataT, SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT
from unienv_interface.space import Space

from ..base.common import BatchBase, BatchT, IndexableType

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
        transformation : DataTransformation,
        metadata_transformation : Optional[DataTransformation] = None
    ):
        self.batch = batch
        self.transformation = transformation
        self._transform_inv = None if not transformation.has_inverse else transformation.direction_inverse(batch.single_space)
        self.metadata_transformation = metadata_transformation if self.batch.single_metadata_space is not None else None
        
        super().__init__(
            transformation.get_target_space_from_source(batch.single_space),
            self.metadata_transformation.get_target_space_from_source(batch.single_metadata_space) if self.metadata_transformation is not None else self.batch.single_metadata_space,
        )

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.single_space.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.single_space.device
    
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
        
        target_dat = self.transformation.transform(
            self.batch.single_space if isinstance(idx, int) else self.batch._batched_space,
            source_dat
        )
        return target_dat

    def get_at_with_metadata(self, idx : Union[IndexableType, BArrayType]) -> Tuple[BatchT, Optional[Dict[str, Any]]]:
        source_dat, metadata = self.batch.get_at_with_metadata(idx)

        target_dat = self.transformation.transform(
            self.batch.single_space if isinstance(idx, int) else self.batch._batched_space,
            source_dat
        )
        target_metadata = metadata if self.metadata_transformation is None else self.metadata_transformation.transform(
            self.batch.single_metadata_space if isinstance(idx, int) else self.batch._batched_metadata_space,
            metadata
        )
        return target_dat, target_metadata

    def set_at(self, idx : Union[IndexableType, BArrayType], value : BatchT) -> None:
        assert self.transformation.has_inverse, "Cannot set values on a transformed batch without an inverse transformation"
        assert self.batch.is_mutable, "Cannot set values on an immutable batch"
        source_dat = self._transform_inv.transform(
            self.single_space if isinstance(idx, int) else self._batched_space,
            value
        )
        self.batch.set_at(idx, source_dat)

    def remove_at(self, idx : Union[IndexableType, BArrayType]) -> None:
        return self.batch.remove_at(idx)

    def extend(self, value : BatchT) -> None:
        assert self.transformation.has_inverse, "Cannot extend values on a transformed batch without an inverse transformation"
        assert self.batch.is_mutable, "Cannot extend values on an immutable batch"
        source_dat = self._transform_inv.transform(
            self.batch._batched_space,
            value
        )
        self.batch.extend(source_dat)

    def close(self):
        self.batch.close()
        self.transformation.close()
