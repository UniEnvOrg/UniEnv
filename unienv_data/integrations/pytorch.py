from typing import Optional, Union, Tuple, Dict, Any

from torch.utils.data import Dataset

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType
from unienv_interface.space.space_utils import construct_utils as scu, batch_utils as sbu
from unienv_data.base import BatchBase, BatchT

__all__ = [
    "UniEnvAsPyTorchDataset",
    "PyTorchAsUniEnvDataset",
]

class UniEnvAsPyTorchDataset(Dataset):
    def __init__(
        self, 
        batch : BatchBase[BatchT, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType],
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

class PyTorchAsUniEnvDataset(BatchBase[BatchT, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]):
    is_mutable = False
    def __init__(
        self,
        dataset: Dataset,
    ):
        """
        A UniEnvPy BatchBase wrapper for PyTorch Datasets.

        Args:
            dataset (Dataset): The PyTorch Dataset to wrap.
        """

        assert len(dataset) >= 0, "The provided PyTorch Dataset must have a defined length."
        self.dataset = dataset
        tmp_data = dataset[0]
        single_space = scu.construct_space_from_data(tmp_data, PyTorchComputeBackend)
        super().__init__(
            single_space,
            None
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def get_at_with_metadata(self, idx):
        if isinstance(idx, int):
            data = self.dataset[idx]
            return data, {}
        elif idx is Ellipsis:
            idx = self.backend.arange(0, len(self))
        elif isinstance(idx, slice):
            idx = self.backend.arange(*idx.indices(len(self)))
        elif self.backend.is_backendarray(idx):
            if self.backend.dtype_is_boolean(idx.dtype):
                idx = self.backend.nonzero(idx)[0]
            else:
                assert self.backend.dtype_is_real_integer(idx.dtype), "Index array must be of integer or boolean type."
                idx = (idx + len(self)) % len(self)
        
        idx_list = self.backend.to_numpy(idx).tolist()
        if hasattr(self.dataset, "__getitems__"):
            data_list = self.dataset.__getitems__(idx_list)
        else:
            data_list = [self.dataset[i] for i in idx_list]
        aggregated_data = sbu.concatenate(self._batched_space, data_list)
        return aggregated_data, {}

    def get_at(self, idx):
        data, _ = self.get_at_with_metadata(idx)
        return data