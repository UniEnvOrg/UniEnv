from typing import Optional, Any
from datasets import Dataset as HFDataset
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from unienv_interface.space.space_utils import construct_utils as scu, batch_utils as sbu
from unienv_interface.utils.dict_util import nested_get
from unienv_data.base import BatchBase, BatchT

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
            kwargs['device'] = device
        
        hf_dataset = hf_dataset.with_format(
            __class__.BACKEND_TO_FORMAT_MAP[backend.simplified_name],
            **kwargs
        )
        first_data = hf_dataset[0]
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

    def get_at_with_metadata(self, idx):
        if self.backend.is_backendarray(idx) and self.backend.dtype_is_boolean(idx):
            idx = self.backend.nonzero(idx)[0]
        return self.hf_dataset[idx], {}
    
    def get_at(self, idx):
        return self.get_at_with_metadata(idx)[0]
    
    def get_column(self, nested_keys):
        single_key = ".".join(nested_keys)
        return __class__(
            self.hf_dataset.select_columns(single_key),
            self.single_space[single_key]
        )
