from typing import Any, Tuple, Union, Optional, List, Dict, Type, TypeVar, Generic, Callable, Iterator
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
        # flat_data = self.backend.array_api_namespace.zeros(
        #     self.sampled_space_flat.shape,
        #     dtype=self.sampled_space_flat.dtype,
        #     device=self.sampled_space_flat.device
        # )
        # flat_data[:] = self.backend.array_api_namespace.arange(
        #     flat_data.shape[-1], device=self.sampled_space_flat.device
        # )[None, None, :] # (1, 1, D)
        flat_data = self.backend.array_api_namespace.broadcast_to(
            self.backend.array_api_namespace.arange(
                self.sampled_space_flat.shape[-1], device=self.sampled_space_flat.device
            )[None, None, :], # (1, 1, D)
            self.sampled_space_flat.shape
        )

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
    
    def expand_index(self, index : BArrayType) -> BArrayType:
        """
        Sample indexes to slice the data, returns a tensor of shape (B, T) that resides on the same device as the data
        """
        index_shifts = self.backend.array_api_namespace.arange( # (T, )
            -self.prefetch_horizon, self.postfetch_horizon + 1, dtype=index.dtype, device=self.data.device
        )
        index = index[:, None] + index_shifts[None, :] # (B, T)
        index = self.backend.array_api_namespace.clip(index, 0, len(self.data) - 1)
        return index

    def get_unfiltered_flat(self, idx : BArrayType) -> BArrayType:
        B = idx.shape[0]
        indices = self.expand_index(idx) # (B, T)
        flat_idx = self.backend.array_api_namespace.reshape(indices, (-1,)) # (B * T, )
        dat_flat = self.data.get_flattened_at(flat_idx) # (B * T, D)
        assert dat_flat.shape[0] == (self.prefetch_horizon + self.postfetch_horizon + 1) * B
        if self._device is not None:
            dat_flat = self.backend.to_device(dat_flat, self._device)
        dat = self.backend.array_api_namespace.reshape(dat_flat, (*indices.shape, -1)) # (B, T, D)
        return dat

    def unfiltered_to_filtered_flat(self, flat_dat: BArrayType) -> Tuple[
        BArrayType, # Data (B, T, D)
        BArrayType, # validity mask (B, T)
        Optional[BArrayType] # episode id (B)
    ]:
        B = flat_dat.shape[0]
        device = self.backend.get_device(flat_dat)
        if self.get_episode_id_fn is not None:
            # fetch episode ids
            if self._epid_flatidx is None:
                dat = sfu.unflatten_data(self.sampled_space, flat_dat, start_dim=2) # (B, T, D)
                episode_ids = self.get_episode_id_fn(dat)
                del dat
            else:
                episode_ids = flat_dat[:, :, self._epid_flatidx]

            assert self.backend.is_backendarray(episode_ids)
            assert episode_ids.shape == (B, self.prefetch_horizon + self.postfetch_horizon + 1)
            episode_id_at_step = episode_ids[:, self.prefetch_horizon]
            episode_id_eq = episode_ids == episode_id_at_step[:, None]
            
            zero_to_B = self.backend.array_api_namespace.arange(
                B,
                device=device
            )
            if self.prefetch_horizon > 0:
                num_eq_prefetch = self.backend.array_api_namespace.sum(episode_id_eq[:, :self.prefetch_horizon], axis=1)
                fill_idx_prefetch = self.prefetch_horizon - num_eq_prefetch
                fill_value_prefetch = flat_dat[
                    zero_to_B, 
                    fill_idx_prefetch
                ] # (B, D)
                fill_value_prefetch = fill_value_prefetch[:, None, :] # (B, 1, D)
                flat_dat_prefetch = self.backend.array_api_namespace.where(
                    episode_id_eq[:, :self.prefetch_horizon, None],
                    flat_dat[:, :self.prefetch_horizon],
                    fill_value_prefetch
                )
            else:
                flat_dat_prefetch = None
            
            if self.postfetch_horizon > 0:
                num_eq_postfetch = self.backend.array_api_namespace.sum(episode_id_eq[:, -self.postfetch_horizon:], axis=1)            
                fill_idx_postfetch = self.prefetch_horizon + num_eq_postfetch
                fill_value_postfetch = flat_dat[
                    zero_to_B, 
                    fill_idx_postfetch
                ]
                fill_value_postfetch = fill_value_postfetch[:, None, :] # (B, 1, D)
                flat_dat_postfetch = self.backend.array_api_namespace.where(
                    episode_id_eq[:, self.prefetch_horizon:, None],
                    flat_dat[:, self.prefetch_horizon:],
                    fill_value_postfetch
                )
            else:
                flat_dat_postfetch = flat_dat[:, self.prefetch_horizon:]
            
            if flat_dat_prefetch is None:
                flat_dat = flat_dat_postfetch
            else:
                flat_dat = self.backend.array_api_namespace.concatenate([
                    flat_dat_prefetch, 
                    flat_dat_postfetch
                ], axis=1) # (B, T, D)
        else:
            episode_id_eq = self.backend.array_api_namespace.ones(
                (flat_dat.shape[:2]),
                dtype=self.backend.default_boolean_dtype,
                device=device
            )
            episode_id_at_step = None
        return flat_dat, episode_id_eq, episode_id_at_step

    def get_flat_at(self, idx : BArrayType):
        return self.get_flat_with_metadata(idx)[0]

    def get_flat_with_metadata(self, idx : BArrayType) -> Tuple[
        BArrayType,
        BArrayType, # validity mask
        Optional[BArrayType]
    ]:
        unfilt_flat_dat = self.get_unfiltered_flat(idx)
        return self.unfiltered_to_filtered_flat(unfilt_flat_dat)
    
    def get_at(self, idx : BArrayType) -> BatchT:
        return self.get_at_with_metadata(idx)[0]

    def get_at_with_metadata(self, idx : BArrayType) -> Tuple[
        BatchT,
        BArrayType,
        Optional[BArrayType]
    ]:
        flat_dat, validity_mask, episode_id = self.get_flat_with_metadata(idx)
        dat = sfu.unflatten_data(self.sampled_space, flat_dat, start_dim=2)
        return dat, validity_mask, episode_id
    
    def sample_flat_with_metadata(self) -> Tuple[
        BArrayType,
        BArrayType,
        Optional[BArrayType]
    ]:
        idx = self.sample_index()
        return self.get_flat_with_metadata(idx)

    def sample_with_metadata(self) -> Tuple[
        BatchT,
        BArrayType,
        Optional[BArrayType]
    ]:
        idx = self.sample_index()
        return self.get_at_with_metadata(idx)
    
    def epoch_iter_with_metadata(self) -> Iterator[Tuple[SamplerBatchT, BArrayType, Optional[BArrayType]]]:
        if self.data_rng is not None:
            self.data_rng, idx = self.backend.random_permutation(self.data_rng, len(self.data), device=self.data.device)
        else:
            self.rng, idx = self.backend.random_permutation(self.rng, len(self.data), device=self.data.device)
        n_batches = len(self.data) // self.batch_size
        num_left = len(self.data) % self.batch_size
        for i in range(n_batches):
            yield self.get_at_with_metadata(idx[i*self.batch_size:(i+1)*self.batch_size])
        if num_left > 0:
            yield self.get_at_with_metadata(idx[-num_left:])

    def epoch_flat_iter_with_metadata(self) -> Iterator[Tuple[BArrayType, BArrayType, Optional[BArrayType]]]:
        if self.data_rng is not None:
            self.data_rng, idx = self.backend.random_permutation(self.data_rng, len(self.data), device=self.data.device)
        else:
            self.rng, idx = self.backend.random_permutation(self.rng, len(self.data), device=self.data.device)
        n_batches = len(self.data) // self.batch_size
        num_left = len(self.data) % self.batch_size
        for i in range(n_batches):
            yield self.get_flat_with_metadata(idx[i*self.batch_size:(i+1)*self.batch_size])
        if num_left > 0:
            yield self.get_flat_with_metadata(idx[-num_left:])