import abc
import os
import dataclasses
from typing import Generic, TypeVar, Optional, Any, Dict, Union, Tuple, Sequence, Callable
from unienv_interface.space import Space, Box, flatten_utils as space_flatten_utils, batch_utils as space_batch_utils
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from .common import TransitionsBase
import json

class TensorStorage(abc.ABC, Generic[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    save_ext : str = ".storage"

    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType] = None
    dtype : BDtypeType
    capacity : Optional[int] = None
    
    next_cursor : int = 0
    single_instance_shape : Tuple[int, ...]

    @abc.abstractmethod
    def get(self, index : Union[int, slice, BArrayType, None]) -> BArrayType:
        raise NotImplementedError
    
    @abc.abstractmethod
    def set(self, index : Union[int, slice, BArrayType, None], value : BArrayType):
        raise NotImplementedError

    def append(self, value : BArrayType):
        self.set(self.next_cursor, value)
        self.next_cursor = (self.next_cursor + 1) % self.capacity
    
    def extend(self, values : BArrayType):
        if self.capacity is None:
            raise NotImplementedError("Please override this method for storage without fixed capacity")

        b = values.shape[0]
        stop_index = self.next_cursor + b
        if stop_index <= self.capacity:
            self.set(slice(self.next_cursor, stop_index), values)
            self.next_cursor = stop_index % self.capacity
        else:
            if b >= self.capacity:
                self.set(None, values[-self.capacity:])
                self.next_cursor = 0
            else:
                self.set(slice(self.next_cursor, self.capacity), values[:self.capacity - self.next_cursor])
                self.set(slice(0, b - (self.capacity - self.next_cursor)), values[self.capacity - self.next_cursor:])
                self.next_cursor = b - (self.capacity - self.next_cursor)

    @abc.abstractmethod
    def dumps(self, path : Union[str, os.PathLike]) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def loads(self, path : Union[str, os.PathLike]) -> None:
        raise NotImplementedError

def index_with_offset(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    index : Union[int, slice, BArrayType, None],
    len_transitions : int,
    capacity : int,
    offset : int
) -> Union[int, BArrayType]:
    """
    Helpful function to convert replay buffer indices to data indices in the TensorStorage
    """
    if index is None:
        nonzero_index = backend.array_api_namespace.arange(len_transitions)
        data_index = (nonzero_index + offset) % capacity
        return data_index
    elif isinstance(index, int):
        assert index < len_transitions and index >= -len_transitions, f"Index {index} is out of bounds for length {len_transitions}"
        nonzero_index = (index + len_transitions) % len_transitions
        data_index = (nonzero_index + offset) % capacity
        return data_index
    elif isinstance(index, slice):
        nonzero_index = backend.array_api_namespace.asarray(list(range(len_transitions))[index])
        data_index = (nonzero_index + offset) % capacity
        return data_index
    else:
        assert len(index.shape) == 1, f"Index shape {index.shape} is not 1D"
        if index.shape == (len_transitions, ) and backend.dtype_is_boolean(
            index.dtype
        ):
            # Boolean mask, rotate the mask by offset
            rotated_mask = backend.array_api_namespace.roll(
                index,
                shift=offset,
                axis=0
            )
            return rotated_mask
        else:
            assert backend.array_api_namespace.min(index) >= -len_transitions and backend.array_api_namespace.max(index) < len_transitions, f"Index {index} is out of bounds for length {len_transitions}"
            nonzero_index = (index + len_transitions) % len_transitions
            data_index = (nonzero_index + offset) % capacity
            return data_index

StorageConstructorT = Callable[
    [
        ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType], # Backend
        Optional[BDeviceType], # Device
        BDtypeType, # DType
        Tuple[int, ...], # Single instance shape
        Optional[int], # Capacity
    ],
    TensorStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]
]

@dataclasses.dataclass
class ReplayBufferItemConfig:
    space: Space
    default_value : Any
    storage_constructor: Optional[StorageConstructorT] = None
    storage_constructor_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

class FlexibleReplayBuffer(Generic[BArrayType, BDeviceType, BDtypeType, BRNGType, ContextType, ObsType, ActType]):
    """
    Flexible replay buffer that stores transitions in a memory-efficient way
    It assumes that the transitions are stored in-order for a given episode, but supports if the transitions in different episodes are interleaved
    This is mostly suitable for any batched or unbatched environment data collection
    """
    def __init__(
        self,
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device : Optional[BDeviceType],
        default_storage_constructor : StorageConstructorT,
        capacity : Optional[int],
        episode_capacity : Optional[int],
        stepwise_storage_configs : Dict[str, ReplayBufferItemConfig],
        episode_storage_configs : Dict[str, ReplayBufferItemConfig],
        episode_id_index_map_storage_constructor : Optional[StorageConstructorT] = None,
        episode_id_index_map_storage_kwargs : Dict[str, Any] = {},
    ):
        self.backend = backend
        self.device = device

        # If we're recording episodic data, we need to have episode_id in the stepwise_storage_configs
        if len(episode_storage_configs) > 0:
            assert "episode_id" in stepwise_storage_configs, "episode_id must be present in stepwise_storage_configs"
            episode_id_cfg = stepwise_storage_configs["episode_id"]
            assert len(episode_id_cfg.space.shape) == 0, "episode_id must be a scalar"
            assert episode_id_cfg.space.backend.dtype_is_real_integer(episode_id_cfg.space.dtype), "episode_id must be an integer"

        self.stepwise_storages : Dict[str, TensorStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]] = {}
        self.stepwise_spaces : Dict[str, Space[Any, Any, BDeviceType, BDtypeType, BRNGType]] = {}
        self.stepwise_flat_spaces : Dict[str, Box[BArrayType, BDeviceType, BDtypeType, BRNGType]] = {}
        self.stepwise_default_values : Dict[str, Any] = {}
        for key, stepwise_cfg in stepwise_storage_configs.items():
            assert backend == stepwise_cfg.space.backend, f"Backend mismatch for {key}"
            space = stepwise_cfg.space if device is None else stepwise_cfg.space.to_device(device)
            storage_constructor = stepwise_cfg.storage_constructor or default_storage_constructor
            self.stepwise_storages[key] = storage_constructor(
                backend,
                device,
                space.dtype,
                space.shape,
                capacity,
                **stepwise_cfg.storage_constructor_kwargs
            )
            if capacity is not None:
                self.stepwise_storages[key].set(
                    None,
                    backend.array_api_namespace.broadcast_to(
                        stepwise_cfg.default_value,
                        (capacity, ) + space.shape
                    )
                )
            
            self.stepwise_spaces[key] = space
            self.stepwise_flat_spaces[key] = space_flatten_utils.flatten_space(space)
            self.stepwise_default_values[key] = stepwise_cfg.default_value
        
        self.episode_wise_storages : Dict[str, TensorStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]] = {}
        self.episode_wise_spaces : Dict[str, Space[Any, Any, BDeviceType, BDtypeType, BRNGType]] = {}
        self.episode_wise_flat_spaces : Dict[str, Box[BArrayType, BDeviceType, BDtypeType, BRNGType]] = {}
        self.episode_wise_default_values : Dict[str, Any] = {}
        for key, episode_cfg in episode_storage_configs.items():
            assert backend == episode_cfg.space.backend, f"Backend mismatch for {key}"
            space = episode_cfg.space if device is None else episode_cfg.space.to_device(device)
            storage_constructor = episode_cfg.storage_constructor or default_storage_constructor
            self.episode_wise_storages[key] = storage_constructor(
                backend,
                device,
                space.dtype,
                space.shape,
                episode_capacity,
                **episode_cfg.storage_constructor_kwargs
            )
            self.episode_wise_spaces[key] = space
            self.episode_wise_flat_spaces[key] = space_flatten_utils.flatten_space(space)
            self.episode_wise_default_values[key] = episode_cfg.default_value
        
        self.step_count = 0
        if len(episode_storage_configs) > 0 or episode_id_index_map_storage_constructor is not None:
            self.episode_id_index_map_storage = (default_storage_constructor or episode_id_index_map_storage_constructor)(
                backend,
                device,
                backend.default_integer_dtype,
                (),
                episode_capacity,
                **episode_id_index_map_storage_kwargs
            )
            self.episode_id_index_map_storage.set(
                None,
                -1
            )
        self.capacity = capacity
        self.episode_capacity = episode_capacity

    def len_transitions(self) -> int:
        return self.step_count

    def len_episodes(self) -> int:
        return self.backend.array_api_namespace.sum(
            self.episode_id_index_map_storage.get(
                None
            ) != -1
        )

    def _update_episode_ids(self, new_epsode_ids : BArrayType):
        assert self.episode_capacity is None or new_epsode_ids.shape[0] <= self.episode_capacity, "Episode IDs exceed capacity"
        assert len(new_epsode_ids.shape) == 1, "Episode IDs must be 1D"
        # Cast to integer
        new_epsode_ids = self.backend.array_api_namespace.astype(new_epsode_ids, self.backend.default_integer_dtype)
        new_epid_b = new_epsode_ids.shape[0]

        # Retrieve the current episode_id_index_map
        episode_id_index_map = self.episode_id_index_map_storage.get(None)

        # episode_id_index map is an array where each element is the episode_id or -1, and the index of the element gives the index to query episode-related data
        # We need to find indexes in episode_id_index_map which is no longer present in new_epsode_ids
        mask_cond = self.backend.array_api_namespace.reshape(
            episode_id_index_map,
            (-1, 1) # Shape (episode_capacity, 1)
        ) != self.backend.array_api_namespace.reshape(
            new_epsode_ids,
            (1, -1) # Shape (1, new_epid_b)
        ) # Shape (episode_capacity, new_epid_b)
        not_present_in_storage_masks = self.backend.array_api_namespace.all(
            mask_cond,
            axis=1
        ) # Shape (episode_capacity, )
        not_present_mask_in_new_episode_ids = self.backend.array_api_namespace.all(
            mask_cond,
            axis=0
        ) # Shape (new_epid_b, )
        present_mask_in_new_episode_ids = self.backend.array_api_namespace.logical_not(
            not_present_mask_in_new_episode_ids
        ) # Shape (new_epid_b, )
        not_present_b = self.backend.array_api_namespace.sum(not_present_in_storage_masks)
        not_present_new_b = self.backend.array_api_namespace.sum(not_present_mask_in_new_episode_ids)

        # Reset the portion of episodic data that is no longer present
        for key, storage in self.episode_wise_storages.items():
            storage.set(
                not_present_in_storage_masks,
                self.backend.array_api_namespace.broadcast_to(
                    self.episode_wise_default_values[key],
                    (not_present_b, ) + self.episode_wise_spaces[key].shape
                )
            )
        
        # Update the episode_id_index_map
        storage_override_num = min(not_present_b, not_present_new_b)
        storage_extend_num = max(not_present_new_b - storage_override_num, 0)

        not_present_in_storage_idxes = self.backend.array_api_namespace.nonzero(not_present_in_storage_masks)[0]
        not_present_idxes_in_new_episode_ids = self.backend.array_api_namespace.nonzero(not_present_mask_in_new_episode_ids)[0]
        if self.episode_capacity is not None:
            assert storage_extend_num <= 0, "This should never happen"
        self.episode_id_index_map_storage.set(
            not_present_in_storage_idxes[:storage_override_num],
            new_epsode_ids[not_present_idxes_in_new_episode_ids[:storage_override_num]]
        )
        if storage_override_num < not_present_b:
            self.episode_id_index_map_storage.set(
                not_present_in_storage_idxes[storage_override_num:],
                -1
            )
        if storage_extend_num > 0:
            self.episode_id_index_map_storage.extend(
                new_epsode_ids[not_present_idxes_in_new_episode_ids[storage_override_num:]]
            )
            for key, storage in self.episode_wise_storages.items():
                storage.extend(
                    self.backend.array_api_namespace.broadcast_to(
                        self.episode_wise_default_values[key],
                        (storage_extend_num, ) + self.episode_wise_spaces[key].shape
                    )
                )

    def get_stepwise(
        self,
        key : str,
        idx : Union[int, slice, BArrayType, None]
    ) -> BArrayType:
        assert key in self.stepwise_storages, f"Key {key} not found in stepwise_storages"
        storage = self.stepwise_storages[key]
        storage_idx = index_with_offset(
            self.backend,
            idx,
            self.step_count,
            self.capacity,
            0 if self.capacity is None or self.step_count < self.capacity else storage.next_cursor
        )
        return storage.get(storage_idx)

    def extend_stepwise(
        self,
        stepwise_data : Dict[str, BArrayType], # Shape (num_transitions, ...)
    ):
        assert set(stepwise_data.keys()) == set(self.stepwise_storages.keys()), "Keys mismatch"
        b = stepwise_data[next(iter(stepwise_data.keys()))].shape[0]
        for key, data in stepwise_data.items():
            assert data.shape[0] == b, "Batch size mismatch"
            self.stepwise_storages[key].extend(data)
        
        remaining_b = b
        while remaining_b > 0:
            if self.capacity is None or self.step_count + remaining_b < self.capacity:
                self.step_count += remaining_b
                remaining_b = 0
            else:
                self.step_count = self.capacity
        
        # Scan for episode id change
        if "episode_id" in stepwise_data:
            unique_episode_ids = self.backend.array_api_namespace.unique_values(
                stepwise_data["episode_id"]
            )
            need_epid_update = False
            current_epids = self.episode_id_index_map_storage.get(None)
            for i in range(unique_episode_ids.shape[0]):
                epid = unique_episode_ids[i]
                if not self.backend.array_api_namespace.any(current_epids == epid):
                    need_epid_update = True
                    break
            if need_epid_update:
                new_epids = self.backend.array_api_namespace.unique_values(
                    self.get_stepwise("episode_id", None)
                )
                self._update_episode_ids(new_epids)
    
    def set_episodic(
        self,
        episode_id : int,
        data : Dict[str, BArrayType]
    ) -> None:
        assert set(data.keys()).issubset(self.episode_wise_storages.keys()), "Keys mismatch"
        episode_idx_map = self.episode_id_index_map_storage.get(None)
        episode_idx = self.backend.array_api_namespace.nonzero(episode_idx_map == episode_id)[0]
        assert len(episode_idx) == 1, "Expected exactly one episode with the given ID"
        episode_idx = episode_idx[0]
        for key, data in data.items():
            storage = self.episode_wise_storages[key]
            storage.set(episode_idx, data)

    def dumps(self, path : Union[str, os.PathLike]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for key, storage in self.stepwise_storages.items():
            storage.dumps(os.path.join(path, f"stepwise_{key}{storage.save_ext}"))
        for key, storage in self.episode_wise_storages.items():
            storage.dumps(os.path.join(path, f"episode_wise_{key}{storage.save_ext}"))
        self.episode_id_index_map_storage.dumps(os.path.join(path, f"episode_id_index_map{self.episode_id_index_map_storage.save_ext}"))
        metadata = {
            "step_count": self.step_count,
            "capacity": self.capacity,
            "episode_capacity": self.episode_capacity
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    
    def loads(self, path : Union[str, os.PathLike]):
        for key, storage in self.stepwise_storages.items():
            storage.loads(os.path.join(path, f"stepwise_{key}{storage.save_ext}"))
        for key, storage in self.episode_wise_storages.items():
            storage.loads(os.path.join(path, f"episode_wise_{key}{storage.save_ext}"))
        self.episode_id_index_map_storage.loads(os.path.join(path, f"episode_id_index_map{self.episode_id_index_map_storage.save_ext}"))
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
            self.step_count = metadata["step_count"]
            self.capacity = metadata["capacity"]
            self.episode_capacity = metadata["episode_capacity"]

