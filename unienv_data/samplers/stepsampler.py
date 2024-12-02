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
        data : BatchBase[BArrayType, BDeviceType, BDtypeType, BRNGType],
        batch_size : int,
        seed : Optional[int] = None
    ):
        assert batch_size > 0, "Batch size must be a positive integer"
        self.data = data
        self.batch_size = batch_size
        self.sampled_space = sbu.batch_space(
            self.data.single_space,
            batch_size
        )
        self.rng = self.backend.random_number_generator(
            seed,
            device=self.device
        )

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.data.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.data.device
    
    def sample(self) -> BatchT:
        self.rng, indices = self.backend.random_discrete_uniform(
            self.rng,
            (self.batch_size,),
            0,
            len(self.data),
            device=self.device,
        )
        return self.data.get_at(indices)

