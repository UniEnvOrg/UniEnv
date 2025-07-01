import abc
import os
import dataclasses
from typing import Generic, TypeVar, Optional, Any, Dict, Union, Tuple, Sequence, Callable, Type, Mapping
from typing_extensions import TypedDict
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space import Space, DictSpace, BoxSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.space.space_utils import serialization_utils as bsu
from unienv_interface.utils.symbol_util import get_class_from_full_name, get_full_class_name

from unienv_data.base import BatchBase, BatchT, IndexableType, SpaceStorage
import json
import copy
from .replay_buffer import ReplayBuffer

EpisodeBatchT = TypeVar("EpisodeBatchT")

class TrajectoryData(TypedDict, Generic[BatchT, EpisodeBatchT]):
    step_data: BatchT
    episode_data : Optional[EpisodeBatchT]

class TrajectoryReplayBuffer(BatchBase[TrajectoryData[BatchT, EpisodeBatchT], BArrayType, BDeviceType, BDtypeType, BRNGType], Generic[BatchT, EpisodeBatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    is_mutable = True

    # =========== Class Attributes ==========
    @staticmethod
    def create(
        step_data_instance_space : Space[BatchT, BDeviceType, BDtypeType, BRNGType],
        step_data_storage_cls : Type[SpaceStorage[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        *,
        step_data_capacity : Optional[int] = None,
        step_episode_id_storage_cls : Optional[Type[SpaceStorage[BArrayType, BArrayType, BDeviceType, BDtypeType, BRNGType]]] = None,
        step_episode_id_capacity : Optional[int] = None,
        step_episode_id_storage_kwargs : Dict[str, Any] = {},
        episode_data_instance_space : Optional[Space[EpisodeBatchT, BDeviceType, BDtypeType, BRNGType]] = None,
        episode_data_storage_cls : Optional[Type[SpaceStorage[EpisodeBatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]]] = None,
        episode_data_capacity : Optional[int] = None,
        episode_data_storage_kwargs : Dict[str, Any] = {},
        cache_path : Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ) -> "TrajectoryReplayBuffer[BatchT, EpisodeBatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        backend = step_data_instance_space.backend 
        step_data_buffer = ReplayBuffer.create(
            step_data_storage_cls,
            step_data_instance_space,
            cache_path=None if cache_path is None else os.path.join(cache_path, "step_data"),
            capacity=step_data_capacity,
            **kwargs
        )
        step_episode_id_kwargs = step_episode_id_storage_kwargs if step_episode_id_storage_cls is not None else kwargs
        step_episode_id_buffer = ReplayBuffer.create(
            step_episode_id_storage_cls or step_data_storage_cls,
            BoxSpace(
                backend,
                low=0,
                high=backend.inf,
                dtype=backend.default_integer_dtype,
                device=step_data_buffer.device,
                shape=()
            ),
            cache_path=None if cache_path is None else os.path.join(cache_path, "step_episode_ids"),
            capacity=step_episode_id_capacity,
            **step_episode_id_kwargs
        )
        if episode_data_instance_space is not None:
            assert episode_data_instance_space.backend == backend, \
                "Episode data instance space must have the same backend as step data instance space."
            assert episode_data_instance_space.device == step_data_instance_space.device, \
                "Episode data instance space must have the same device as step data instance space."
            episode_data_storage_kwargs = episode_data_storage_kwargs if episode_data_storage_cls is not None else kwargs
            episode_data_buffer = ReplayBuffer.create(
                episode_data_storage_cls or step_data_storage_cls,
                episode_data_instance_space,
                cache_path=None if cache_path is None else os.path.join(cache_path, "episode_data"),
                capacity=episode_data_capacity,
                **episode_data_storage_kwargs
            )
        else:
            episode_data_buffer = None
        return TrajectoryReplayBuffer(
            step_data_buffer,
            step_episode_id_buffer,
            episode_data_buffer,
            current_episode_id=0,
            episode_id_to_index_map={} if episode_data_buffer is not None else None,
        )

    @staticmethod
    def is_loadable_from(
        path : Union[str, os.PathLike]
    ) -> bool:
        if os.path.exists(os.path.join(path, "metadata.json")):
            with open(os.path.join(path, "metadata.json"), "r") as f:
                metadata = json.load(f)
            return metadata.get('type', None) == __class__.__name__
        return False

    @staticmethod
    def load_from(
        path : Union[str, os.PathLike],
        *,
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device: Optional[BDeviceType] = None,
        step_storage_kwargs: Dict[str, Any] = {},
        step_episode_id_storage_kwargs: Dict[str, Any] = {},
        episode_storage_kwargs: Dict[str, Any] = {},
        **storage_kwargs
    ) -> "TrajectoryReplayBuffer[BatchT, EpisodeBatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]":
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        assert metadata['type'] == __class__.__name__, f"Metadata type {metadata['type']} does not match expected type {__class__.__name__}"

        step_storage_kwargs.update(storage_kwargs)
        step_data_buffer = ReplayBuffer.load_from(
            os.path.join(path, "step_data"),
            backend=backend,
            device=device,
            **step_storage_kwargs
        )
        step_episode_id_storage_kwargs.update(storage_kwargs)
        step_episode_id_buffer = ReplayBuffer.load_from(
            os.path.join(path, "step_episode_ids"),
            backend=backend,
            device=device,
            **step_episode_id_storage_kwargs
        )
        episode_data_buffer = None
        if metadata.get("episode_id_to_index_map", None) is not None and os.path.exists(os.path.join(path, "episode_data")):
            episode_storage_kwargs.update(storage_kwargs)
            episode_data_buffer = ReplayBuffer.load_from(
                os.path.join(path, "episode_data"),
                backend=backend,
                device=device,
                **episode_storage_kwargs
            )
        else:
            episode_data_buffer = None
        
        
        # Convert JSON string keys back to int keys for episode_id_to_index_map
        if episode_data_buffer is not None:
            raw_map = metadata.get("episode_id_to_index_map")
            episode_id_to_index_map = (
                {int(k): v for k, v in raw_map.items()}
                if raw_map is not None else {}
            )
        else:
            episode_id_to_index_map = None

        return TrajectoryReplayBuffer(
            step_data_buffer,
            step_episode_id_buffer,
            episode_data_buffer,
            current_episode_id=metadata["current_episode_id"],
            episode_id_to_index_map=episode_id_to_index_map,
        )

    # ========== Instance Attributes and Methods ==========
    def dumps(self, path : Union[str, os.PathLike]):
        os.makedirs(path, exist_ok=True)
        step_data_path = os.path.join(path, "step_data")
        self.step_data_buffer.dumps(step_data_path)
        step_episode_id_path = os.path.join(path, "step_episode_ids")
        self.step_episode_id_buffer.dumps(step_episode_id_path)
        if self.episode_data_buffer is not None:
            episode_data_path = os.path.join(path, "episode_data")
            self.episode_data_buffer.dumps(episode_data_path)
        
        # Convert episode_id_to_index_map keys to strings for JSON serialization
        if self.episode_id_to_index_map is not None:
            episode_id_to_index_map = {
                str(ep_id): idx
                for ep_id, idx in self.episode_id_to_index_map.items()
            }
        else:
            episode_id_to_index_map = None

        metadata = {
            "type": __class__.__name__,
            "current_episode_id": self.current_episode_id,
            "episode_id_to_index_map": episode_id_to_index_map,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def __init__(
        self,
        step_data_buffer : ReplayBuffer[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType],
        step_episode_id_buffer : ReplayBuffer[BArrayType, BArrayType, BDeviceType, BDtypeType, BRNGType],
        episode_data_buffer : Optional[ReplayBuffer[EpisodeBatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]],
        current_episode_id: int = 0,
        episode_id_to_index_map : Optional[Dict[int, int]] = None,
    ):
        assert step_data_buffer.backend == step_episode_id_buffer.backend, \
            "Step data buffer and step episode ID buffer must have the same backend."
        assert step_data_buffer.device == step_episode_id_buffer.device, \
            "Step data buffer and step episode ID buffer must have the same device."
        if episode_data_buffer is not None:
            assert step_data_buffer.backend == episode_data_buffer.backend, \
                "Step data buffer and episode data buffer must have the same backend."
            assert step_data_buffer.device == episode_data_buffer.device, \
                "Step data buffer and episode data buffer must have the same device."
            assert episode_id_to_index_map is not None, \
                "Episode ID to index map must be provided if episode data buffer is not None."
        assert isinstance(step_episode_id_buffer.single_space, BoxSpace), \
            "Step episode ID buffer's single space must be a BoxSpace."
        assert step_episode_id_buffer.single_space.shape == (), \
            "Step episode ID buffer's single space must be a scalar BoxSpace."

        backend = step_data_buffer.backend

        single_space = DictSpace(
            backend,
            {
                "step_data": step_data_buffer.single_space,
            },
            device=step_data_buffer.device
        )
        if episode_data_buffer is not None:
            single_space["episode_data"] = episode_data_buffer.single_space
        
        metadata_space = copy.copy(step_data_buffer.single_metadata_space) if step_data_buffer.single_metadata_space is not None else DictSpace(backend, {}, device=step_data_buffer.device)
        metadata_space["episode_ids"] = step_episode_id_buffer.single_space
        super().__init__(
            single_space,
            metadata_space
        )

        self.step_data_buffer = step_data_buffer
        self.step_episode_id_buffer = step_episode_id_buffer
        self.episode_data_buffer = episode_data_buffer
        self.current_episode_id = current_episode_id
        self.episode_id_to_index_map = episode_id_to_index_map

    def __len__(self) -> int:
        return len(self.step_episode_id_buffer)

    def episode_id_to_episode_data_index(
        self,
        episode_id : Union[int, BArrayType],
    ) -> Union[int, BArrayType]:
        if isinstance(episode_id, int):
            return self.episode_id_to_index_map[episode_id]
        else:
            return self.backend.asarray([
                self.episode_id_to_index_map[ep_id]
                for ep_id in episode_id
            ])

    @property
    def capacity(self) -> Optional[int]:
        return self.step_episode_id_buffer.capacity

    @property
    def episode_capacity(self) -> Optional[int]:
        if self.episode_data_buffer is not None:
            return self.episode_data_buffer.capacity
        return None

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.step_data_buffer.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.step_data_buffer.device

    def get_flattened_at(self, idx):
        return self.get_flattened_at_with_metadata(idx)[0]

    def get_flattened_at_with_metadata(self, idx):
        episode_ids = self.step_episode_id_buffer.get_at(idx)
        step_data_flat, metadata = self.step_data_buffer.get_flattened_at_with_metadata(idx)
        
        # We do some tricks knowing how flat data is layed out for dictionary space
        if self.episode_data_buffer is not None:
            episode_data_flat = self.episode_data_buffer.get_flattened_at(
                self.episode_id_to_episode_data_index(episode_ids)
            )
            if isinstance(idx, int):
                data_flat = self.backend.concat([
                    step_data_flat,
                    episode_data_flat
                ], axis=0)
            else:
                data_flat = self.backend.concat([
                    step_data_flat,
                    episode_data_flat
                ], axis=1)
        else:
            data_flat = step_data_flat
        
        metadata = {} if metadata is None else copy.copy(metadata)
        metadata["episode_ids"] = episode_ids
        return data_flat, metadata

    def get_at(self, idx):
        return self.get_at_with_metadata(idx)[0]
    
    def get_at_with_metadata(self, idx):
        episode_ids = self.step_episode_id_buffer.get_at(idx)
        step_data, metadata = self.step_data_buffer.get_at_with_metadata(idx)
        
        data = {
            "step_data": step_data,
        }
        if self.episode_data_buffer is not None:
            episode_data = self.episode_data_buffer.get_at(
                self.episode_id_to_episode_data_index(episode_ids)
            )
            data["episode_data"] = episode_data

        metadata = {} if metadata is None else copy.copy(metadata)
        metadata["episode_ids"] = episode_ids
        return data, metadata
    
    def set_flattened_at(self, idx, value):
        """
        Set the flattened data at the specified index.
        """
        if isinstance(idx, int):
            unflat_data = sfu.unflatten_data(self.single_space, value)
        else:
            unflat_data = sfu.unflatten_data(self._batched_space, value, start_dim=1)
        self.set_at(idx, unflat_data)
    
    def set_at(self, idx, value):
        if "episode_ids" in value:
            self.step_episode_id_buffer.set_at(idx, value['episode_ids'])
            
        if "step_data" in value:
            step_data = value["step_data"]
            self.step_data_buffer.set_at(idx, step_data)
        
        if "episode_data" in value and self.episode_data_buffer is not None:
            episode_ids = value["episode_ids"] if "episode_ids" in value else self.step_episode_id_buffer.get_at(idx)
            episode_data = value["episode_data"]
            episode_ids_unique, unique_indices, _, _ = self.backend.unique_all(episode_ids)
            self.set_episode_data_at(
                episode_ids_unique,
                sbu.get_at(
                    self._batched_space['episode_data'],
                    episode_data,
                    unique_indices
                )
            )
    
    def set_episode_data_at(
        self,
        episode_id: Union[int, BArrayType],
        value: Any
    ) -> None:
        assert self.episode_data_buffer is not None, \
            "Episode data buffer is not set. Cannot set episode data."
        if isinstance(episode_id, int):
            if episode_id in self.episode_id_to_index_map:
                index = self.episode_id_to_index_map[episode_id]
            else:
                assert self.episode_data_buffer.capacity is None or len(self.episode_data_buffer) < self.episode_data_buffer.capacity, \
                    "Episode data buffer is full. Cannot set episode data."
                index = len(self.episode_id_to_index_map)
                self.episode_id_to_index_map[episode_id] = index
                self.episode_data_buffer.extend(
                    sbu.concatenate(
                        self._batched_space['episode_data'],
                        [value]
                    )
                )
        else:
            assert self.backend.is_backendarray(episode_id), \
                "Episode ID must be an integer or a backend array."
            assert len(episode_id.shape) == 1, \
                "Episode ID must be a 1D array."
            
            valid_ids = [] # Stores (rb_index, index_in_batch) tuples
            new_ids = [] # Stores (episode_id, index_in_batch) tuples
            for i in range(episode_id.shape[0]):
                ep_id = episode_id[i]
                if ep_id in self.episode_id_to_index_map:
                    valid_ids.append((self.episode_id_to_index_map[ep_id], i))
                else:
                    new_ids.append((ep_id, i))
            if len(new_ids) > 0:
                assert self.episode_data_buffer.capacity is None or len(self.episode_data_buffer) + len(new_ids) <= self.episode_data_buffer.capacity, \
                    "Episode data buffer is full. Cannot set episode data."
                start_index = len(self.episode_id_to_index_map)
                for ep_id, i in new_ids:
                    self.episode_id_to_index_map[ep_id] = start_index
                    start_index += 1
                self.episode_data_buffer.extend(
                    sbu.concatenate(
                        self._batched_space['episode_data'],
                        [value[i] for _, i in new_ids]
                    )
                )
            if len(valid_ids) > 0:
                rb_indices = self.backend.asarray([i for i, _ in valid_ids], device=self.device)
                indices_in_batch = self.backend.asarray([i for _, i in valid_ids], device=self.device)
                self.episode_data_buffer.set_at(
                    rb_indices,
                    sbu.get_at(
                        self._batched_space['episode_data'],
                        value,
                        indices_in_batch
                    )
                )

    def extend_flattened(self, value):
        try:
            unflattened_data = sfu.unflatten_data(self._batched_space, value, start_dim=1)
        except:
            # If we cannot decode the full data, we assume it's a single step data
            unflattened_data = {
                "step_data": sfu.unflatten_data(self._batched_space['step_data'], value, start_dim=1)
            }
        self.extend(unflattened_data)

    def extend(self, value):
        B = sbu.batch_size_data(value)
        if B == 0:
            return
        
        if not isinstance(value, Mapping) or "step_data" not in value:
            # If the value is not a mapping or does not contain "step_data", we assume it's a single step data
            value = {
                "step_data": value
            }

        if "episode_ids" in value:
            episode_ids = value["episode_ids"]
        else:
            episode_ids = self.backend.full(
                (B,),
                self.current_episode_id,
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )
        self.step_episode_id_buffer.extend(episode_ids)
        self.step_data_buffer.extend(value["step_data"])
        
        if "episode_data" in value and self.episode_data_buffer is not None:
            episode_ids_unique, unique_indices, _, _ = self.backend.unique_all(episode_ids)
            self.set_episode_data_at(
                episode_ids_unique,
                sbu.get_at(
                    self._batched_space['episode_data'],
                    value["episode_data"],
                    unique_indices
                )
            )
    
    def set_current_episode_data(
        self,
        value : EpisodeBatchT
    ) -> None:
        self.set_episode_data_at(
            self.current_episode_id,
            value
        )

    def mark_episode_end(self) -> None:
        self.current_episode_id += 1

    def clear(self):
        self.step_data_buffer.clear()
        self.step_episode_id_buffer.clear()
        if self.episode_data_buffer is not None:
            self.episode_data_buffer.clear()
            self.episode_id_to_index_map = {}
        self.current_episode_id = 0
    
    def close(self):
        self.step_data_buffer.close()
        self.step_episode_id_buffer.close()
        if self.episode_data_buffer is not None:
            self.episode_data_buffer.close()
