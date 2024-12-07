from typing import Any, Tuple, Union, Optional, List, Dict, Type, TypeVar, Generic, Callable
from unienv_data.base import BatchBase, BatchT, SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType, BatchSampler
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, flatten_utils as sfu, batch_utils as sbu

class SliceSampler(
    BatchSampler[
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        data : BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType],
        batch_size : int,        
        prefetch_horizon : int = 0,
        postfetch_horizon : int = 0,
        get_episode_id_fn: Optional[Callable[[BatchT], BArrayType]] = None,
        seed : Optional[int] = None,
    ):
        assert batch_size > 0, "Batch size must be a positive integer"
        assert prefetch_horizon >= 0, "Prefetch horizon must be a non-negative integer"
        assert postfetch_horizon >= 0, "Postfetch horizon must be a non-negative integer"
        assert prefetch_horizon > 0 or postfetch_horizon > 0, "At least one of prefetch_horizon and postfetch_horizon must be greater than 0, otherwise you can use `StepSampler`"
        self.data = data
        self.batch_size = batch_size
        self.prefetch_horizon = prefetch_horizon
        self.postfetch_horizon = postfetch_horizon
        self.get_episode_id_fn = get_episode_id_fn

        self.single_slice_space = sbu.batch_space(
            self.data.single_space,
            self.prefetch_horizon + self.postfetch_horizon + 1
        )
        self.sampled_space = sbu.batch_space(
            self.single_slice_space,
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
    
    def sample_indices(self) -> BArrayType:
        self.rng, indices = self.backend.random_discrete_uniform( # (B, )
            self.rng,
            (self.batch_size,),
            0,
            len(self.data),
            device=self.device,
        )
        indices_shifts = self.backend.array_api_namespace.arange( # (T, )
            -self.prefetch_horizon, self.postfetch_horizon + 1, dtype=indices.dtype, device=self.device
        )
        indices = self.backend.array_api_namespace.expand_dims(indices, axis=1) + indices_shifts # (B, T)
        indices = self.backend.array_api_namespace.clip(indices, 0, len(self.data) - 1)
        return indices

    def sample_unfiltered_flat(self) -> BArrayType:
        indices = self.sample_indices()
        flat_idx = self.backend.array_api_namespace.reshape(indices, (-1,)) # (B * T, )
        dat_flat = self.data.get_flattened_at(flat_idx) # (B * T, D)
        assert dat_flat.shape[0] == (self.prefetch_horizon + self.postfetch_horizon + 1) * self.batch_size
        dat = self.backend.array_api_namespace.reshape(dat_flat, (*indices.shape, -1)) # (B, T, D)
        return dat

    def sample_unfiltered(self) -> BatchT:
        flat_dat = self.sample_unfiltered_flat() # (B, T, D)
        dat = sfu.unflatten_data(self.sampled_space, flat_dat, start_dim=2) # (B, T, D)
        return dat
    
    def sample_flat(self) -> BArrayType:
        flat_dat = self.sample_unfiltered_flat()
        
        if self.get_episode_id_fn is not None:
            dat = sfu.unflatten_data(self.sampled_space, flat_dat, start_dim=2) # (B, T, D)

            episode_ids = self.get_episode_id_fn(dat)
            assert self.backend.is_backendarray(episode_ids)
            assert episode_ids.shape == (self.batch_size, self.prefetch_horizon + self.postfetch_horizon + 1)
            episode_id_at_step = episode_ids[:, self.prefetch_horizon]
            episode_id_eq = episode_ids == episode_id_at_step[:, None]
            if self.prefetch_horizon > 0:
                num_eq_prefetch = self.backend.array_api_namespace.sum(episode_id_eq[:, :self.prefetch_horizon], axis=1)
                fill_idx_prefetch = self.prefetch_horizon - num_eq_prefetch
                fill_value_prefetch = flat_dat[self.backend.array_api_namespace.arange(self.batch_size), fill_idx_prefetch] # (B, D)
                fill_value_prefetch = self.backend.array_api_namespace.broadcast_to(
                    self.backend.array_api_namespace.expand_dims(fill_value_prefetch, axis=1), # (B, 1, D)
                    (self.batch_size, self.prefetch_horizon, flat_dat.shape[-1]) # (B, T, D)
                )
                flat_dat_prefetch = self.backend.array_api_namespace.where(
                    episode_id_eq[:, :self.prefetch_horizon, None],
                    flat_dat[:, :self.prefetch_horizon],
                    fill_value_prefetch
                )
            else:
                flat_dat_prefetch = flat_dat[:, :self.prefetch_horizon]
            
            if self.postfetch_horizon > 0:
                num_eq_postfetch = self.backend.array_api_namespace.sum(episode_id_eq[:, -self.postfetch_horizon:], axis=1)            
                fill_idx_postfetch = self.prefetch_horizon + num_eq_postfetch
                fill_value_postfetch = flat_dat[self.backend.array_api_namespace.arange(self.batch_size), fill_idx_postfetch]
                fill_value_postfetch = self.backend.array_api_namespace.broadcast_to(
                    self.backend.array_api_namespace.expand_dims(fill_value_postfetch, axis=1),
                    (self.batch_size, self.postfetch_horizon + 1, flat_dat.shape[-1])
                )
                flat_dat_postfetch = self.backend.array_api_namespace.where(
                    episode_id_eq[:, self.prefetch_horizon:, None],
                    flat_dat[:, self.prefetch_horizon:],
                    fill_value_postfetch
                )
            else:
                flat_dat_postfetch = flat_dat[:, self.prefetch_horizon:]
            
            flat_dat = self.backend.array_api_namespace.concatenate((flat_dat_prefetch, flat_dat_postfetch), axis=1)
        return flat_dat

    def sample(self):
        flat_dat = self.sample_flat()
        dat = sfu.unflatten_data(self.sampled_space, flat_dat, start_dim=2)
        return dat