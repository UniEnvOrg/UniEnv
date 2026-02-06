from typing import Optional, Any, Union, Tuple, Dict, Sequence
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from unienv_data.base import BatchBase, BatchT

def recursive_index(
    space_or_data : Union[BatchT, Space],
    index : Sequence[Any],
) -> Union[BatchT, Space]:
    if len(index) == 0:
        return space_or_data
    else:
        return recursive_index(
            space_or_data[index[0]],
            index[1:]
        )

class SubItemBatch(BatchBase[
    BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    """
    A batch that offers a view of the original batch with a fixed index.
    This is a read-only batch, since it is a view of the original batch. If you want to change the data, you should mutate the containing batch instead.
    """

    is_mutable = False

    def __init__(
        self,
        batch: BatchBase[
            BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
        ],
        sub_indexes: Sequence[Any],
    ):
        self.batch = batch
        self.sub_indexes = sub_indexes
        super().__init__(
            recursive_index(batch.single_space, sub_indexes),
            recursive_index(batch.single_metadata_space, sub_indexes) if batch.single_metadata_space is not None else None,
        )

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.batch.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.batch.device

    def __len__(self) -> int:
        return len(self.batch)

    def get_flattened_at(self, idx):
        return recursive_index(
            self.batch.get_flattened_at(idx),
            self.sub_indexes
        )

    def get_flattened_at_with_metadata(self, idx):
        return recursive_index(
            self.batch.get_flattened_at_with_metadata(idx),
            self.sub_indexes
        )

    def get_at(self, idx):
        return recursive_index(
            self.batch.get_at(idx),
            self.sub_indexes
        )

    def get_at_with_metadata(self, idx) -> Tuple[BatchT, Dict[str, Any]]:
        return recursive_index(
            self.batch.get_at_with_metadata(idx),
            self.sub_indexes
        )

    def close(self) -> None:
        pass