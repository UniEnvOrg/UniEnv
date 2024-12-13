from typing import Any, Tuple, Union, Optional, List, Dict, Type, TypeVar, Generic
from unienv_data.base import BatchBase, BatchT, SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType, BatchSampler
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, flatten_utils as sfu, batch_utils as sbu

class StepSampler(
    BatchSampler[
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        data : BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType],
        batch_size : int,
        seed : Optional[int] = None,
        device : Optional[BDeviceType] = None
    ):
        assert batch_size > 0, "Batch size must be a positive integer"
        self.data = data
        self.batch_size = batch_size
        self._device = device
        self.sampled_space = sbu.batch_space(
            self.data.single_space,
            batch_size
        )
        self.sampled_space_flat = sfu.flatten_space(self.sampled_space, start_dim=1)
        self.data_rng = self.backend.random_number_generator(
            seed,
            device=data.device
        )
        if device is not None:
            self.sampled_space = self.sampled_space.to_device(device)


    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.data.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self._device or self.data.device
    
    def sample_indices(self) -> BArrayType:
        self.data_rng, indices = self.backend.random_discrete_uniform(
            self.data_rng,
            (self.batch_size,),
            0,
            len(self.data),
            device=self.data.device,
        )
        return indices

    def sample_flat(self) -> BArrayType:
        indices = self.sample_indices()
        dat = self.data.get_flattened_at(indices)
        if self._device is not None:
            dat = self.backend.to_device(dat, self._device)
        return dat
    
    def sample(self) -> BatchT:
        indices = self.sample_indices()
        dat = self.data.get_at(indices)
        if self._device is not None:
            dat = self.sampled_space.from_same_backend(dat)
        return dat