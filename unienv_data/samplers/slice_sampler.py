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
    """
    It is recommended to use SliceSampler as the final layer sampler
    Because it has to reshape the data, and we add an additional dimension T apart from the Batch dimension
    Which makes a lot of the wrappers incompatible with it
    """
    def __init__(
        self,
        data : BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType],
        batch_size : int,        
        prefetch_horizon : int = 0,
        postfetch_horizon : int = 0,
        get_episode_id_fn: Optional[Callable[[BatchT], BArrayType]] = None,
        seed : Optional[int] = None,
        device : Optional[BDeviceType] = None,
    ):
        assert batch_size > 0, "Batch size must be a positive integer"
        assert prefetch_horizon >= 0, "Prefetch horizon must be a non-negative integer"
        assert postfetch_horizon >= 0, "Postfetch horizon must be a non-negative integer"
        assert prefetch_horizon > 0 or postfetch_horizon > 0, "At least one of prefetch_horizon and postfetch_horizon must be greater than 0, otherwise you can use `StepSampler`"
        self.data = data
        self.batch_size = batch_size
        self.prefetch_horizon = prefetch_horizon
        self.postfetch_horizon = postfetch_horizon
        self._device = device

        self.single_slice_space = sbu.batch_space(
            self.data.single_space,
            self.prefetch_horizon + self.postfetch_horizon + 1
        )
        self.sampled_space = sbu.batch_space(
            self.single_slice_space,
            batch_size
        )
        self.sampled_space_flat = sfu.flatten_space(self.sampled_space, start_dim=2)
        
        if device is not None:
            self.single_slice_space = self.single_slice_space.to_device(device)
            self.sampled_space = self.sampled_space.to_device(device)

        self.data_rng = self.backend.random_number_generator(
            seed,
            device=data.device
        )
        
        self.get_episode_id_fn = get_episode_id_fn
        self._build_epid_cache()

    def _build_epid_cache(self):
        """
        Build a cache that helps speed up the filtering process
        """
        if self.get_episode_id_fn is None:
            self._epid_flatidx = None
        
        # First make a fake batch to get the episode ids
        flat_data = self.backend.array_api_namespace.zeros(
            self.sampled_space_flat.shape,
            dtype=self.sampled_space_flat.dtype,
            device=self.sampled_space_flat.device
        )
        flat_data[:] = self.backend.array_api_namespace.arange(
            flat_data.shape[-1], device=self.sampled_space_flat.device
        )[None, None, :] # (1, 1, D)

        dat = sfu.unflatten_data(self.sampled_space, flat_data, start_dim=2)
        episode_ids = self.get_episode_id_fn(dat)
        del dat

        epid_flatidx = int(episode_ids[0, 0])
        if self.backend.array_api_namespace.all(episode_ids == epid_flatidx):
            self._epid_flatidx = epid_flatidx
        else:
            self._epid_flatidx = None

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.data.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self._device or self.data.device
    
    def sample_indices(self) -> BArrayType:
        """
        Sample indexes to slice the data, returns a tensor of shape (B, T) that resides on the same device as the data
        """
        self.data_rng, indices = self.backend.random_discrete_uniform( # (B, )
            self.data_rng,
            (self.batch_size,),
            0,
            len(self.data),
            device=self.data.device,
        )
        indices_shifts = self.backend.array_api_namespace.arange( # (T, )
            -self.prefetch_horizon, self.postfetch_horizon + 1, dtype=indices.dtype, device=self.data.device
        )
        indices = self.backend.array_api_namespace.expand_dims(indices, axis=1) + indices_shifts # (B, T)
        indices = self.backend.array_api_namespace.clip(indices, 0, len(self.data) - 1)
        return indices

    def sample_unfiltered_flat(self) -> BArrayType:
        indices = self.sample_indices()
        flat_idx = self.backend.array_api_namespace.reshape(indices, (-1,)) # (B * T, )
        dat_flat = self.data.get_flattened_at(flat_idx) # (B * T, D)
        assert dat_flat.shape[0] == (self.prefetch_horizon + self.postfetch_horizon + 1) * self.batch_size
        if self._device is not None:
            dat_flat = self.backend.to_device(dat_flat, self._device)
        dat = self.backend.array_api_namespace.reshape(dat_flat, (*indices.shape, -1)) # (B, T, D)
        return dat

    def sample_unfiltered(self) -> BatchT:
        flat_dat = self.sample_unfiltered_flat() # (B, T, D)
        dat = sfu.unflatten_data(self.sampled_space, flat_dat, start_dim=2) # (B, T, D)
        return dat

    def unfiltered_to_filtered_flat(self, flat_dat: BArrayType) -> Tuple[
        BArrayType,
        BArrayType # validity mask
    ]:
        
        if self.get_episode_id_fn is not None:
            # fetch episode ids
            if self._epid_flatidx is None:
                dat = sfu.unflatten_data(self.sampled_space, flat_dat, start_dim=2) # (B, T, D)
                episode_ids = self.get_episode_id_fn(dat)
                del dat
            else:
                episode_ids = flat_dat[:, :, self._epid_flatidx]

            assert self.backend.is_backendarray(episode_ids)
            assert episode_ids.shape == (self.batch_size, self.prefetch_horizon + self.postfetch_horizon + 1)
            episode_id_at_step = episode_ids[:, self.prefetch_horizon]
            episode_id_eq = episode_ids == episode_id_at_step[:, None]
            if self.prefetch_horizon > 0:
                num_eq_prefetch = self.backend.array_api_namespace.sum(episode_id_eq[:, :self.prefetch_horizon], axis=1)
                fill_idx_prefetch = self.prefetch_horizon - num_eq_prefetch
                fill_value_prefetch = flat_dat[
                    self.backend.array_api_namespace.arange(
                        self.batch_size,
                        device=fill_idx_prefetch.device
                    ), 
                    fill_idx_prefetch
                ] # (B, D)
                fill_value_prefetch = fill_value_prefetch[:, None, :] # (B, 1, D)
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
                fill_value_postfetch = fill_value_postfetch[:, None, :] # (B, 1, D)
                flat_dat_postfetch = self.backend.array_api_namespace.where(
                    episode_id_eq[:, self.prefetch_horizon:, None],
                    flat_dat[:, self.prefetch_horizon:],
                    fill_value_postfetch
                )
            else:
                flat_dat_postfetch = flat_dat[:, self.prefetch_horizon:]
            
            flat_dat = self.backend.array_api_namespace.concatenate((flat_dat_prefetch, flat_dat_postfetch), axis=1)
        else:
            episode_id_eq = self.backend.array_api_namespace.ones(
                (flat_dat.shape[:2]),
                dtype=self.backend.default_boolean_dtype,
                device=self.backend.get_device(flat_dat)
            )
        return flat_dat, episode_id_eq

    def sample_flat(self) -> BArrayType:
        flat_dat = self.sample_unfiltered_flat()
        return self.unfiltered_to_filtered_flat(flat_dat)[0]

    def sample_flat_with_validity_mask(self) -> Tuple[
        BArrayType,
        BArrayType # validity mask
    ]:
        flat_dat = self.sample_unfiltered_flat()
        return self.unfiltered_to_filtered_flat(flat_dat)

    def sample(self):
        flat_dat = self.sample_flat()
        dat = sfu.unflatten_data(self.sampled_space, flat_dat, start_dim=2)
        return dat
    
    def sample_with_validity_mask(self) -> Tuple[
        BatchT,
        BArrayType # validity mask
    ]:
        flat_dat, validity_mask = self.sample_flat_with_validity_mask()
        dat = sfu.unflatten_data(self.sampled_space, flat_dat, start_dim=2)
        return dat, validity_mask