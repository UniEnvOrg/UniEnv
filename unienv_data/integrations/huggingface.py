from typing import Optional, Any, Mapping
import numpy as np
from datasets import Dataset as HFDataset
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace, BoxSpace, BinarySpace, TextSpace
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
        hf_index = self._normalize_index(idx)
        data = _scalars_to_arrays(self.hf_dataset[hf_index])
        if not isinstance(hf_index, int):
            num_rows = len(range(len(self))[hf_index]) if isinstance(hf_index, slice) else len(hf_index)
            data = _columns_to_batch(self.single_space, data, num_rows)
        return data, {}
    
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
    """HF's numpy format returns numpy scalars for scalar columns; UniEnv expects (0-d) arrays.
    Text values (``np.str_`` / ``np.bytes_``) stay Python strings, as for ``TextSpace``."""
    if isinstance(data, (np.str_, np.bytes_)):
        return data.item()
    if isinstance(data, np.generic):
        return np.asarray(data)
    if isinstance(data, dict):
        return {key: _scalars_to_arrays(value) for key, value in data.items()}
    return data


def _columns_to_batch(space : Space, data : Any, num_rows : int) -> Any:
    """Bring a batch HF returned for a list / slice index into the layout of the batched ``space``.

    HF returns struct (nested dict) columns row by row (a list or object array of dicts) and
    text columns as lists / unicode arrays; UniEnv batches nested dicts as dicts of stacked
    arrays and text as object arrays. Empty batches get arrays of shape ``(0, *shape)``.
    """
    if num_rows == 0:
        return _empty_batch(space)
    if isinstance(space, DictSpace):
        if isinstance(data, Mapping):
            return {key: _columns_to_batch(subspace, data[key], num_rows) for key, subspace in space.spaces.items()}
        # Row-oriented: one dict per row
        return sbu.concatenate(space, [_scalars_to_arrays(row) for row in data])
    if isinstance(space, TextSpace):
        return sbu.concatenate(space, [_scalars_to_arrays(item) for item in data])
    return data


def _empty_batch(space : Space) -> Any:
    if isinstance(space, DictSpace):
        return {key: _empty_batch(subspace) for key, subspace in space.spaces.items()}
    if isinstance(space, (BoxSpace, BinarySpace)):
        return space.backend.zeros((0, *space.shape), dtype=space.dtype, device=space.device)
    return np.empty((0,), dtype=object)
