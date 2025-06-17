from typing import Any, Tuple, Union, Optional, List, Dict, Type, TypeVar, Generic, Callable, Iterator
from unienv_data.base import BatchBase, BatchT, SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType, BatchSampler
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, BoxSpace, BinarySpace, DictSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu

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

        if self.data.single_metadata_space is not None:
            self.sampled_metadata_space = sbu.batch_space(
                self.data.single_metadata_space,
                self.prefetch_horizon + self.postfetch_horizon + 1
            )
            self.sampled_metadata_space = sbu.batch_space(
                self.sampled_metadata_space,
                batch_size
            )
        else:
            self.sampled_metadata_space = None
        
        if get_episode_id_fn is not None:
            if self.sampled_metadata_space is None:
                self.sampled_metadata_space = DictSpace(
                    self.backend,
                    {},
                    device=self.device
                )
            
            self.sampled_metadata_space['slice_valid_mask'] = BinarySpace(
                self.backend,
                shape=(self.batch_size, self.prefetch_horizon + self.postfetch_horizon + 1),
                dtype=self.backend.default_boolean_dtype,
                device=self.device
            )
            self.sampled_metadata_space['episode_id'] = BoxSpace(
                self.backend,
                low=-2_147_483_647,
                high=2_147_483_647,
                shape=(self.batch_size, ),
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )
        
        if device is not None:
            self.single_slice_space = self.single_slice_space.to(device=device)
            self.sampled_space = self.sampled_space.to(device=device)

        self.data_rng = self.backend.random.random_number_generator(
            seed,
            device=data.device
        )
        
        self.get_episode_id_fn = get_episode_id_fn
        self._build_epid_cache()

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.data.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self._device or self.data.device

    def _build_epid_cache(self):
        """
        Build a cache that helps speed up the filtering process
        """
        if self.get_episode_id_fn is None:
            self._epid_flatidx = None
        
        # First make a fake batch to get the episode ids
        # flat_data = self.backend.zeros(
        #     self.sampled_space_flat.shape,
        #     dtype=self.sampled_space_flat.dtype,
        #     device=self.sampled_space_flat.device
        # )
        # flat_data[:] = self.backend.arange(
        #     flat_data.shape[-1], device=self.sampled_space_flat.device
        # )[None, None, :] # (1, 1, D)
        flat_data = self.backend.broadcast_to(
            self.backend.arange(
                self.sampled_space_flat.shape[-1], device=self.sampled_space_flat.device
            )[None, None, :], # (1, 1, D)
            self.sampled_space_flat.shape
        )

        dat = sfu.unflatten_data(self.sampled_space, flat_data, start_dim=2)
        episode_ids = self.get_episode_id_fn(dat)
        del dat

        epid_flatidx = int(episode_ids[0, 0])
        if self.backend.all(episode_ids == epid_flatidx):
            self._epid_flatidx = epid_flatidx
        else:
            self._epid_flatidx = None
    
    def expand_index(self, index : BArrayType) -> BArrayType:
        """
        Sample indexes to slice the data, returns a tensor of shape (B, T) that resides on the same device as the data
        """
        index_shifts = self.backend.arange( # (T, )
            -self.prefetch_horizon, self.postfetch_horizon + 1, dtype=index.dtype, device=self.data.device
        )
        index = index[:, None] + index_shifts[None, :] # (B, T)
        index = self.backend.clip(index, 0, len(self.data) - 1)
        return index

    def _get_unfiltered_flat_with_metadata(self, idx : BArrayType) -> Tuple[BArrayType, Optional[Dict[str, Any]]]:
        B = idx.shape[0]
        indices = self.expand_index(idx) # (B, T)
        flat_idx = self.backend.reshape(indices, (-1,)) # (B * T, )
        dat_flat, metadata = self.data.get_flattened_at_with_metadata(flat_idx) # (B * T, D)
        metadata_reshaped = self.backend.map_fn_over_arrays(
            metadata,
            lambda x: self.backend.reshape(x, (*indices.shape, *x.shape[1:]))
        ) if metadata is not None else None
        assert dat_flat.shape[0] == (self.prefetch_horizon + self.postfetch_horizon + 1) * B

        dat = self.backend.reshape(dat_flat, (*indices.shape, -1)) # (B, T, D)
        return dat, metadata_reshaped

    def unfiltered_to_filtered_flat(self, flat_dat: BArrayType) -> Tuple[
        BArrayType, # Data (B, T, D)
        BArrayType, # validity mask (B, T)
        Optional[BArrayType] # episode id (B)
    ]:
        B = flat_dat.shape[0]
        device = self._device or self.backend.device(flat_dat)
        if self.get_episode_id_fn is not None:
            # fetch episode ids
            if self._epid_flatidx is None:
                if self._device is not None:
                    new_flat_dat = self.backend.to_device(flat_dat, device)
                dat = sfu.unflatten_data(self.sampled_space, flat_dat, start_dim=2) # (B, T, D)
                episode_ids = self.get_episode_id_fn(dat)
                if self._device is not None:
                    episode_ids = self.backend.to_device(episode_ids, device)
                    flat_dat = new_flat_dat
                del dat
            else:
                episode_ids = flat_dat[:, :, self._epid_flatidx]
                if self._device is not None:
                    flat_dat = self.backend.to_device(flat_dat, device)
                    episode_ids = self.backend.to_device(episode_ids, device)

            assert self.backend.is_backendarray(episode_ids)
            assert episode_ids.shape == (B, self.prefetch_horizon + self.postfetch_horizon + 1)
            episode_id_at_step = episode_ids[:, self.prefetch_horizon]
            episode_id_eq = episode_ids == episode_id_at_step[:, None]
            
            zero_to_B = self.backend.arange(
                B,
                device=device
            )
            if self.prefetch_horizon > 0:
                num_eq_prefetch = self.backend.sum(episode_id_eq[:, :self.prefetch_horizon], axis=1)
                fill_idx_prefetch = self.prefetch_horizon - num_eq_prefetch
                fill_value_prefetch = flat_dat[
                    zero_to_B, 
                    fill_idx_prefetch
                ] # (B, D)
                fill_value_prefetch = fill_value_prefetch[:, None, :] # (B, 1, D)
                flat_dat_prefetch = self.backend.where(
                    episode_id_eq[:, :self.prefetch_horizon, None],
                    flat_dat[:, :self.prefetch_horizon],
                    fill_value_prefetch
                )
            else:
                flat_dat_prefetch = None
            
            if self.postfetch_horizon > 0:
                num_eq_postfetch = self.backend.sum(episode_id_eq[:, -self.postfetch_horizon:], axis=1)            
                fill_idx_postfetch = self.prefetch_horizon + num_eq_postfetch
                fill_value_postfetch = flat_dat[
                    zero_to_B, 
                    fill_idx_postfetch
                ]
                fill_value_postfetch = fill_value_postfetch[:, None, :] # (B, 1, D)
                flat_dat_postfetch = self.backend.where(
                    episode_id_eq[:, self.prefetch_horizon:, None],
                    flat_dat[:, self.prefetch_horizon:],
                    fill_value_postfetch
                )
            else:
                flat_dat_postfetch = flat_dat[:, self.prefetch_horizon:]
            
            if flat_dat_prefetch is None:
                flat_dat = flat_dat_postfetch
            else:
                flat_dat = self.backend.concatenate([
                    flat_dat_prefetch, 
                    flat_dat_postfetch
                ], axis=1) # (B, T, D)
        else:
            episode_id_eq = None
            episode_id_at_step = None
        return flat_dat, episode_id_eq, episode_id_at_step

    def get_flat_at(self, idx : BArrayType):
        return self.get_flat_at_with_metadata(idx)[0]

    def get_flat_at_with_metadata(self, idx : BArrayType) -> Tuple[
        BArrayType,
        Optional[Dict[str, Any]]
    ]:
        unfilt_flat_dat, metadata = self._get_unfiltered_flat_with_metadata(idx)
        dat, episode_id_eq, episode_id_at_step = self.unfiltered_to_filtered_flat(unfilt_flat_dat)
        if episode_id_at_step is not None:
            if metadata is None:
                metadata = {}
            metadata.update({
                "slice_valid_mask": episode_id_eq,
                "episode_id": episode_id_at_step
            })
        
        return dat, metadata
    
    def get_at(self, idx : BArrayType) -> BatchT:
        return self.get_at_with_metadata(idx)[0]

    def get_at_with_metadata(self, idx : BArrayType) -> Tuple[
        BatchT,
        Optional[Dict[str, Any]]
    ]:
        flat_dat, metadata = self.get_flat_at_with_metadata(idx)
        dat = sfu.unflatten_data(self.sampled_space, flat_dat, start_dim=2)
        return dat, metadata