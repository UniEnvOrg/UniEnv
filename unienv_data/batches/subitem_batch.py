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
        # Metadata describes whole samples rather than fields, so it is forwarded unchanged.
        super().__init__(
            recursive_index(batch.single_space, sub_indexes),
            batch.single_metadata_space,
        )

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.batch.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.batch.device

    def __len__(self) -> int:
        return len(self.batch)

    def get_at(self, idx):
        return recursive_index(
            self.batch.get_at(idx),
            self.sub_indexes
        )

    def get_at_with_metadata(self, idx) -> Tuple[BatchT, Dict[str, Any]]:
        data, metadata = self.batch.get_at_with_metadata(idx)
        return recursive_index(data, self.sub_indexes), metadata

    # get_flattened_at / get_flattened_at_with_metadata: the BatchBase defaults flatten the
    # selected item with this batch's space (the parent's flat layout cannot be indexed by key).

    def close(self) -> None:
        pass