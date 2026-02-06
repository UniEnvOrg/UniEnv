from typing import Optional, Any, Union, Tuple, Dict
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from unienv_data.base import BatchBase, BatchT, IndexableType

class SubIndexedBatch(BatchBase[
    BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    """
    A batch that offers a view of the original batch with a fixed index range.
    This does not support batch extension. 
    """

    def __init__(
        self,
        batch: BatchBase[
            BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
        ],
        sub_indexes: Union[IndexableType, BArrayType],
    ):
        self.batch = batch

        if isinstance(sub_indexes, int):
            sub_indexes = slice(sub_indexes, sub_indexes + 1)
        if isinstance(sub_indexes, slice):
            sub_indexes = self.backend.arange(*sub_indexes.indices(len(batch)), device=batch.device, dtype=self.backend.default_index_dtype)
        if self.backend.is_backendarray(sub_indexes) and self.backend.dtype_is_boolean(sub_indexes.dtype):
            sub_indexes = self.backend.nonzero(sub_indexes)[0]
        assert self.backend.is_backendarray(sub_indexes) or sub_indexes is Ellipsis, "Sub indexes must be a backend array"
        if self.backend.is_backendarray(sub_indexes):
            assert len(sub_indexes.shape) == 1, "Sub indexes must be a 1D array"
            assert self.backend.dtype_is_real_integer(sub_indexes.dtype), "Sub indexes must be an array of integers"

        self.sub_indexes = sub_indexes
        super().__init__(
            batch.single_space,
            batch.single_metadata_space,
        )
        
    @property
    def is_mutable(self) -> bool:
        return self.batch.is_mutable

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.batch.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.batch.device

    def __len__(self) -> int:
        return len(self.sub_indexes) if self.sub_indexes is not Ellipsis else len(self.batch)

    def get_flattened_at(self, idx):
        if self.sub_indexes is Ellipsis:
            return self.batch.get_flattened_at(idx)
        else:
            return self.batch.get_flattened_at(self.sub_indexes[idx])

    def get_flattened_at_with_metadata(self, idx):
        if self.sub_indexes is Ellipsis:
            return self.batch.get_flattened_at_with_metadata(idx)
        else:
            return self.batch.get_flattened_at_with_metadata(self.sub_indexes[idx])

    def get_at(self, idx):
        if self.sub_indexes is Ellipsis:
            return self.batch.get_at(idx)
        else:
            return self.batch.get_at(self.sub_indexes[idx])

    def get_at_with_metadata(self, idx) -> Tuple[BatchT, Dict[str, Any]]:
        if self.sub_indexes is Ellipsis:
            return self.batch.get_at_with_metadata(idx)
        else:
            return self.batch.get_at_with_metadata(self.sub_indexes[idx])
    
    def set_at(self, idx, value):
        if self.sub_indexes is Ellipsis:
            self.batch.set_at(idx, value)
        else:
            self.batch.set_at(self.sub_indexes[idx], value)
    
    def set_flattened_at(self, idx, value):
        if self.sub_indexes is Ellipsis:
            self.batch.set_flattened_at(idx, value)
        else:
            self.batch.set_flattened_at(self.sub_indexes[idx], value)

    def extend(self, value):
        if self.sub_indexes is Ellipsis:
            self.batch.extend(value)
        else:
            raise NotImplementedError("SubIndexedBatch does not support extension")

    def extend_flattened(self, value):
        if self.sub_indexes is Ellipsis:
            self.batch.extend_flattened(value)
        else:
            raise NotImplementedError("SubIndexedBatch does not support extension")

    def close(self) -> None:
        pass