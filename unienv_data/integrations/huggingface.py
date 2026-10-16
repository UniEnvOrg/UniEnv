from typing import Optional, Any
import numpy as np
from datasets import Dataset as HFDataset
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace
from unienv_interface.space.space_utils import construct_utils as scu, batch_utils as sbu
from unienv_interface.utils.dict_util import nested_get
from unienv_data.base import BatchBase, BatchT
from unienv_data.batches.subitem_batch import SubItemBatch

__all__ = [
    'HFAsUniEnvDataset'
]

class HFAsUniEnvDataset(BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    BACKEND_TO_FORMAT_MAP = {
        "numpy": "numpy",
        "pytorch": "torch",
        "jax": "jax",
    }

    is_mutable = False

    @staticmethod
    def create(
        hf_dataset: HFDataset,
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device : Optional[BDeviceType] = None,
    ) -> "HFAsUniEnvDataset[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        kwargs = {}
        if backend.simplified_name != 'numpy' and device is not None:
            # HF only accepts string identifiers for jax devices (they are not picklable).
            kwargs['device'] = str(device) if backend.simplified_name == 'jax' else device
        
        hf_dataset = hf_dataset.with_format(
            __class__.BACKEND_TO_FORMAT_MAP[backend.simplified_name],
            **kwargs
        )
        first_data = _scalars_to_arrays(hf_dataset[0])
        return __class__(
            hf_dataset,
            scu.construct_space_from_data(first_data, backend)
        )

    def __init__(
        self,
        hf_dataset: HFDataset,
        space : Space[Any, BDeviceType, BDtypeType, BRNGType],
    ) -> None:
        self.hf_dataset = hf_dataset
        super().__init__(
            space
        )
    
    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _normalize_index(self, idx):
        """Turn a UniEnv index into one HF datasets accepts (int, slice or list of ints)."""
        if idx is Ellipsis:
            return slice(None)
        if isinstance(idx, (int, slice)):
            return idx
        if self.backend.is_backendarray(idx):
            if self.backend.dtype_is_boolean(idx.dtype):
                assert idx.shape == (len(self),), f"Boolean index must have shape ({len(self)},), got {idx.shape}"
                idx = self.backend.nonzero(idx)[0]
            idx = self.backend.to_numpy(idx)
        idx = np.asarray(idx)
        assert idx.ndim == 1, f"Index arrays must be 1-D, got shape {idx.shape}"
        assert np.all((idx >= -len(self)) & (idx < len(self))), f"Index {idx} out of bounds for dataset of length {len(self)}"
        return (idx.astype(np.int64) % max(len(self), 1)).tolist()

    def get_at_with_metadata(self, idx):
        return _scalars_to_arrays(self.hf_dataset[self._normalize_index(idx)]), {}
    
    def get_at(self, idx):
        return self.get_at_with_metadata(idx)[0]
    
    def get_column(self, nested_keys):
        # Columns may be named with dots (e.g. LeRobot's "observation.state").
        single_key = ".".join(str(k) for k in nested_keys)
        if isinstance(self.single_space, DictSpace) and single_key in self.single_space.spaces:
            column_dataset = __class__(
                self.hf_dataset.select_columns(single_key),
                DictSpace(self.backend, {single_key: self.single_space[single_key]}, device=self.device),
            )
            return SubItemBatch(column_dataset, [single_key])
        return super().get_column(nested_keys)


def _scalars_to_arrays(data : Any) -> Any:
    """HF's numpy format returns numpy scalars for scalar columns; UniEnv expects (0-d) arrays."""
    if isinstance(data, np.generic):
        return np.asarray(data)
    if isinstance(data, dict):
        return {key: _scalars_to_arrays(value) for key, value in data.items()}
    return data
