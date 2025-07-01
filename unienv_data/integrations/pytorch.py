from typing import Optional, Union, Tuple, Dict, Any

from torch.utils.data import Dataset

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_data.base import BatchBase, BatchT

class UniEnvPyTorchDataset(Dataset):
    def __init__(
        self, 
        batch : BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType],
        include_metadata : bool = False,
    ):
        """
        A PyTorch Dataset wrapper for UniEnvPy batches.
        Note that UniEnv's `BatchBase` will automatically collate data when indexed with batches, and therefore in the dataloader you can set `collate_fn=None`.
        
        Args:
            batch (BatchBase): The UniEnvPy batch to wrap.
        """
        self.batch = batch
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.batch)

    def __getitem__(self, index) -> Union[BatchT, Tuple[BatchT, Dict[str, Any]]]:
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Union[BatchT, Tuple[BatchT, Dict[str, Any]]]: The batch data at the specified index,
            optionally including metadata if `include_metadata` is True.
        """
        assert isinstance(index, int), "Index must be an integer."
        if self.include_metadata:
            return self.batch.get_at_with_metadata(index)
        return self.batch.get_at(index)

    def __getitems__(self, indices: list[int]) -> list[Union[BatchT, Tuple[BatchT, Dict[str, Any]]]]:
        """
        Get multiple items from the dataset.

        Args:
            indices (list[int]): The indices of the items to retrieve.

        Returns:
            Union[BatchT, Tuple[BatchT, Dict[str, Any]]]: Batch data at the specified indices, 
            optionally including metadata if `include_metadata` is True.
        """
        indices = self.batch.backend.asarray(indices, dtype=self.batch.backend.default_integer_dtype, device=self.batch.device)
        if self.include_metadata:
            return self.batch.get_at_with_metadata(indices)
        return self.batch.get_at(indices)