from typing import Any, Tuple, Union, Optional, List, Dict, Type, TypeVar, Generic
from unienv_data.base import BatchBase, BatchT, SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType, BatchSampler
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu

class StepSampler(
    BatchSampler[
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ]
):
    """
    Samples mini-batches of individual steps (uniformly, with replacement) from ``data``.

    Performance notes:

    * For data held in RAM (e.g. an in-memory ``ReplayBuffer`` and views over it), sampling
      in-process is usually faster than ``MultiprocessingSampler``: worker processes only pay
      off when producing a batch is expensive (decoding, transformations, storage reads),
      since every batch is copied to the main process.
    * ``sample_flat()`` / ``get_flattened_at()`` flatten a sample into one array of a common
      dtype, so ``uint8`` images next to ``float32`` fields are cast to ``float32`` (4x the
      bytes and a conversion per batch). Prefer ``sample()`` for image data.
    * Read single fields through ``data.get_column(keys)`` rather than sampling whole samples.
    """
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
        self.data_rng = self.backend.random.random_number_generator(
            seed,
            device=data.device
        )
        super().__init__(
            data.single_space if device is None else data.single_space.to(device=device),
            data.single_metadata_space if device is None or data.single_metadata_space is None else data.single_metadata_space.to(device=device),
            batch_size
        )

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.data.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self._device or self.data.device

    def get_flattened_at(self, idx):
        dat = self.data.get_flattened_at(idx)
        if self._device is not None:
            dat = self.backend.to_device(dat, self._device)
        return dat

    def get_flattened_at_with_metadata(self, idx):
        dat, metadata = self.data.get_flattened_at_with_metadata(idx)
        if self._device is not None:
            dat = self.backend.to_device(dat, self._device)
        if metadata is not None and self._device is not None:
            metadata = self.backend.map_fn_over_arrays(metadata, lambda x: self.backend.to_device(x, self._device))
        return dat, metadata

    def get_at(self, idx):
        dat = self.data.get_at(idx)
        if self._device is not None:
            dat = self.backend.map_fn_over_arrays(dat, lambda x: self.backend.to_device(x, self._device))
        return dat

    def get_at_with_metadata(self, idx):
        dat, metadata = self.data.get_at_with_metadata(idx)
        if self._device is not None:
            dat = self.backend.map_fn_over_arrays(dat, lambda x: self.backend.to_device(x, self._device))
        if metadata is not None and self._device is not None:
            metadata = self.backend.map_fn_over_arrays(metadata, lambda x: self.backend.to_device(x, self._device))
        return dat, metadata