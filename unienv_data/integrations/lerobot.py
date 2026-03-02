from typing import Optional, Any, Dict, List, Tuple, Sequence, Union, Literal
from collections import OrderedDict
import importlib.util
import os

import numpy as np
import pyarrow.parquet as pq

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, BoxSpace, DictSpace, BinarySpace
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_data.base import BatchBase, BatchT, IndexableType

from unienv_data.third_party.lerobot import (
    parse_info_json,
    build_index,
    get_data_file_path,
    get_video_file_path,
    LeRobotSchema,
    LeRobotIndex,
    LeRobotFeature,
    LeRobotVersion,
    EpisodeMetadata,
)

__all__ = [
    "LeRobotAsUniEnvDataset",
]

# LeRobot dtype strings to numpy dtypes
_LEROBOT_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


def _normalize_image_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Normalize image shape to (H, W, C).
    LeRobot may store shapes as (C, H, W) or (H, W, C).
    """
    if len(shape) == 3 and shape[0] <= 4 and shape[1] > 4 and shape[2] > 4:
        return (shape[1], shape[2], shape[0])
    return shape


def _np_dtype_to_backend_dtype(backend: ComputeBackend, np_dtype: np.dtype):
    """Convert a numpy dtype to the equivalent backend dtype."""
    return backend.from_numpy(np.zeros(1, dtype=np_dtype)).dtype


def _feature_to_space(
    feature: LeRobotFeature,
    backend: ComputeBackend,
    device: Optional[Any] = None,
) -> Space:
    if feature.is_video or feature.is_image:
        shape = _normalize_image_shape(feature.shape)
        return BoxSpace(
            backend,
            low=0, high=255,
            shape=shape,
            dtype=_np_dtype_to_backend_dtype(backend, np.dtype(np.uint8)),
            device=device,
        )
    elif feature.dtype == "bool":
        return BinarySpace(
            backend,
            shape=feature.shape,
            dtype=None,
            device=device,
        )
    elif feature.dtype in _LEROBOT_DTYPE_MAP:
        np_dtype = np.dtype(_LEROBOT_DTYPE_MAP[feature.dtype])
        if np.issubdtype(np_dtype, np.floating):
            low, high = -np.inf, np.inf
        elif np.issubdtype(np_dtype, np.signedinteger):
            info = np.iinfo(np_dtype)
            low, high = info.min, info.max
        else:
            info = np.iinfo(np_dtype)
            low, high = info.min, info.max
        return BoxSpace(
            backend,
            low=low, high=high,
            shape=feature.shape,
            dtype=_np_dtype_to_backend_dtype(backend, np_dtype),
            device=device,
        )
    else:
        return BoxSpace(
            backend,
            low=-np.inf, high=np.inf,
            shape=feature.shape,
            dtype=_np_dtype_to_backend_dtype(backend, np.dtype(np.float32)),
            device=device,
        )


def _schema_to_flat_dictspace(
    schema: LeRobotSchema,
    backend: ComputeBackend,
    device: Optional[Any] = None,
    include_features: Optional[List[str]] = None,
    exclude_features: Optional[List[str]] = None,
) -> DictSpace:
    """
    Convert LeRobot features to a flat DictSpace.

    Feature names are kept as dot-separated strings (e.g. "observation.state")
    matching the Parquet column names. This is consistent with HFAsUniEnvDataset.
    """
    spaces: Dict[str, Space] = {}
    for feat_name, feature in schema.features.items():
        if include_features is not None and feat_name not in include_features:
            continue
        if exclude_features is not None and feat_name in exclude_features:
            continue
        spaces[feat_name] = _feature_to_space(feature, backend, device)
    return DictSpace(backend, spaces, device=device)


class LeRobotAsUniEnvDataset(BatchBase[BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """
    Wraps a local LeRobot dataset directory as a UniEnv BatchBase.

    Parses the LeRobot on-disk format natively (no ``lerobot`` package required).
    Supports v2.0, v2.1, and v3.0 dataset formats.

    Usage::

        from unienv_interface.backends.numpy import NumpyComputeBackend
        dataset = LeRobotAsUniEnvDataset.from_local(
            "/path/to/lerobot/dataset",
            backend=NumpyComputeBackend,
        )
        frame = dataset[42]
        batch = dataset[0:100]
    """

    is_mutable = False

    @staticmethod
    def from_local(
        dataset_dir: str,
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device: Optional[BDeviceType] = None,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        decode_video: bool = True,
        video_backend: Literal["pyav", "torchcodec", "auto"] = "auto",
    ) -> "LeRobotAsUniEnvDataset[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        """Load a LeRobot dataset from a local directory."""
        schema = parse_info_json(dataset_dir)
        index = build_index(dataset_dir, schema)

        effective_exclude_features = list(exclude_features) if exclude_features is not None else []
        if not decode_video:
            effective_exclude_features.extend(
                feat_name for feat_name, feat in schema.features.items() if feat.is_video
            )

        space = _schema_to_flat_dictspace(
            schema, backend, device,
            include_features=include_features,
            exclude_features=effective_exclude_features,
        )
        return LeRobotAsUniEnvDataset(
            dataset_dir=dataset_dir,
            schema=schema,
            index=index,
            space=space,
            _backend=backend,
            _device=device,
            decode_video=decode_video,
            video_backend=video_backend,
            include_features=include_features,
            exclude_features=effective_exclude_features,
        )

    @staticmethod
    def from_hub(
        repo_id: str,
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device: Optional[BDeviceType] = None,
        revision: Optional[str] = None,
        local_dir: Optional[str] = None,
        **kwargs,
    ) -> "LeRobotAsUniEnvDataset[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        """
        Download a LeRobot dataset from HuggingFace Hub and load it.

        Uses ``huggingface_hub.snapshot_download`` (not the ``lerobot`` package).
        """
        from huggingface_hub import snapshot_download

        dataset_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
        )
        return LeRobotAsUniEnvDataset.from_local(
            dataset_dir, backend, device, **kwargs
        )

    def __init__(
        self,
        dataset_dir: str,
        schema: LeRobotSchema,
        index: LeRobotIndex,
        space: DictSpace,
        _backend: ComputeBackend,
        _device: Optional[Any] = None,
        decode_video: bool = True,
        video_backend: Literal["pyav", "torchcodec", "auto"] = "auto",
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
    ):
        super().__init__(space)
        self._dataset_dir = dataset_dir
        self._schema = schema
        self._index = index
        self._device = _device
        self._decode_video = decode_video
        self._video_backend = video_backend

        # Determine active features based on filters
        active = list(self._schema.features.keys())
        if include_features is not None:
            active = [f for f in active if f in include_features]
        if exclude_features is not None:
            active = [f for f in active if f not in exclude_features]
        self._active_features = active

        self._tabular_features = [
            f for f in active if not self._schema.features[f].is_video
        ]
        self._video_features = [
            f for f in active if self._schema.features[f].is_video
        ] if decode_video else []

        # Simple Parquet cache: episode_index -> {col_name: np.ndarray}
        self._parquet_cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()
        self._parquet_cache_max = 16

    def __len__(self) -> int:
        return self._index.total_frames

    def get_at(self, idx: Union[IndexableType, BArrayType]) -> BatchT:
        return self.get_at_with_metadata(idx)[0]

    def get_at_with_metadata(
        self, idx: Union[IndexableType, BArrayType]
    ) -> Tuple[BatchT, Dict[str, Any]]:
        is_single = isinstance(idx, int)
        global_indices = self._normalize_index(idx)
        episode_groups = self._index.global_indices_to_episode_and_local(global_indices)
        batch_size = len(global_indices)

        # Pre-allocate numpy output arrays, convert to backend at the end
        np_result: Dict[str, np.ndarray] = {}
        for feat_name in self._active_features:
            feat = self._schema.features[feat_name]
            if feat.is_video or feat.is_image:
                shape = _normalize_image_shape(feat.shape)
                np_result[feat_name] = np.zeros((batch_size, *shape), dtype=np.uint8)
            elif feat.dtype == "bool":
                np_result[feat_name] = np.zeros((batch_size, *feat.shape), dtype=np.bool_)
            elif feat.dtype in _LEROBOT_DTYPE_MAP:
                np_result[feat_name] = np.zeros(
                    (batch_size, *feat.shape),
                    dtype=_LEROBOT_DTYPE_MAP[feat.dtype],
                )
            else:
                np_result[feat_name] = np.zeros((batch_size, *feat.shape), dtype=np.float32)

        # Load tabular features from Parquet
        for ep_idx, (local_indices, output_positions) in episode_groups.items():
            episode_meta = self._index.episodes[ep_idx]
            dataset_offset = int(episode_meta.dataset_from_index or 0)
            file_row_indices = local_indices + dataset_offset
            parquet_data = self._load_parquet_episode(ep_idx)
            for feat_name in self._tabular_features:
                if feat_name in parquet_data:
                    values = parquet_data[feat_name][file_row_indices]
                    feat_shape = self._schema.features[feat_name].shape
                    if values.ndim == 1 and feat_shape:
                        values = values.reshape(-1, *feat_shape)
                    np_result[feat_name][output_positions] = values

        # Load video features
        for feat_name in self._video_features:
            for ep_idx, (local_indices, output_positions) in episode_groups.items():
                episode_meta = self._index.episodes[ep_idx]
                dataset_offset = int(episode_meta.dataset_from_index or 0)
                file_frame_indices = local_indices + dataset_offset
                total_file_frames = (
                    int(episode_meta.dataset_to_index)
                    if episode_meta.dataset_to_index is not None
                    else int(max(episode_meta.length, int(file_frame_indices.max()) + 1))
                )
                video_path = get_video_file_path(
                    self._dataset_dir, self._schema, feat_name, episode_meta.index,
                    file_index=episode_meta.data_file_index,
                    chunk_index=episode_meta.data_chunk_index,
                )
                frames = self._decode_video_frames(
                    video_path, file_frame_indices, total_file_frames
                )
                np_result[feat_name][output_positions] = frames

        # Convert to backend arrays
        result = self._to_backend(np_result)

        if is_single:
            batched_space = sbu.batch_space(self.single_space, batch_size)
            result = sbu.get_at(batched_space, result, 0)

        return result, {}

    def get_column(self, nested_keys: Sequence[str]) -> "BatchBase":
        """Return a view over a single feature column, joining nested keys with dots."""
        single_key = ".".join(nested_keys)
        sub_space = self.single_space[single_key]
        from unienv_data.batches.subitem_batch import SubItemBatch
        return SubItemBatch(self, [single_key])

    # ========== Episode-level access ==========

    def get_episode(self, episode_index: int) -> BatchT:
        """Load all frames for a single episode."""
        start, end = self._index.episode_frame_range(episode_index)
        return self.get_at(slice(start, end))

    @property
    def num_episodes(self) -> int:
        return self._index.total_episodes

    @property
    def fps(self) -> int:
        return self._schema.fps

    @property
    def schema(self) -> LeRobotSchema:
        return self._schema

    @property
    def episode_metadata(self) -> List[EpisodeMetadata]:
        return self._index.episodes

    # ========== Internal helpers ==========

    def _normalize_index(self, idx: Union[IndexableType, BArrayType]) -> np.ndarray:
        total = self._index.total_frames
        if isinstance(idx, int):
            norm_idx = int(idx)
            if norm_idx < 0:
                norm_idx += total
            if norm_idx < 0 or norm_idx >= total:
                raise IndexError(f"Index {idx} out of bounds for dataset of length {total}")
            return np.array([norm_idx], dtype=np.int64)
        elif isinstance(idx, slice):
            return np.arange(*idx.indices(total), dtype=np.int64)
        elif idx is Ellipsis:
            return np.arange(total, dtype=np.int64)
        else:
            if self.backend.is_backendarray(idx):
                if self.backend.dtype_is_boolean(idx.dtype):
                    idx = self.backend.nonzero(idx)[0]
                arr = self.backend.to_numpy(idx)
            else:
                arr = np.asarray(idx)
            if arr.dtype.kind == "b":
                if arr.ndim != 1 or arr.shape[0] != total:
                    raise IndexError(
                        f"Boolean index must be 1D with length {total}, got shape {arr.shape}"
                    )
                return np.nonzero(arr)[0].astype(np.int64)

            arr = arr.ravel().astype(np.int64)
            negative_mask = arr < 0
            if np.any(negative_mask):
                arr = arr.copy()
                arr[negative_mask] += total
            if np.any(arr < 0) or np.any(arr >= total):
                raise IndexError(f"Index values out of bounds for dataset of length {total}: {arr}")
            return arr

    def _load_parquet_episode(self, episode_index: int) -> Dict[str, np.ndarray]:
        if episode_index in self._parquet_cache:
            # Move to end (most recently used)
            self._parquet_cache.move_to_end(episode_index)
            return self._parquet_cache[episode_index]

        ep_meta = self._index.episodes[episode_index]
        path = get_data_file_path(
            self._dataset_dir, self._schema, ep_meta.index,
            file_index=ep_meta.data_file_index,
            chunk_index=ep_meta.data_chunk_index,
        )
        table = pq.read_table(path)

        data: Dict[str, np.ndarray] = {}
        for col_name in table.column_names:
            if col_name not in self._tabular_features:
                continue
            col = table.column(col_name)
            if hasattr(col, 'combine_chunks'):
                col = col.combine_chunks()
            # Handle list columns (e.g. observation.state stored as fixed-size lists)
            col_type_str = str(col.type)
            if 'list' in col_type_str:
                data[col_name] = np.stack(col.to_pylist())
            else:
                data[col_name] = col.to_numpy(zero_copy_only=False)

        # LRU eviction
        if len(self._parquet_cache) >= self._parquet_cache_max:
            self._parquet_cache.popitem(last=False)
        self._parquet_cache[episode_index] = data

        return data

    def _decode_video_frames(
        self,
        video_path: str,
        local_indices: np.ndarray,
        total_frames: int,
    ) -> np.ndarray:
        from unienv_data.storages.video_storage import PyAvVideoReader, TorchCodecVideoReader

        if self._video_backend == "auto":
            if importlib.util.find_spec("torchcodec"):
                reader_cls = TorchCodecVideoReader
            else:
                reader_cls = PyAvVideoReader
        elif self._video_backend == "torchcodec":
            reader_cls = TorchCodecVideoReader
        else:
            reader_cls = PyAvVideoReader

        from unienv_interface.backends.numpy import NumpyComputeBackend
        with reader_cls(
            backend=NumpyComputeBackend,
            filename=video_path,
            buffer_pixel_format="rgb24",
            hwaccel="auto",
            seek_mode="exact",
            device=None,
        ) as reader:
            frames = reader.read(local_indices, total_frames)

        # reader.read returns backend arrays; ensure numpy for assembly
        if not isinstance(frames, np.ndarray):
            frames = NumpyComputeBackend.to_numpy(frames)
        return frames

    def _to_backend(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            arr = self.backend.from_numpy(data)
            if self._device is not None:
                arr = self.backend.to_device(arr, self._device)
            return arr
        elif isinstance(data, dict):
            return {k: self._to_backend(v) for k, v in data.items()}
        return data

    def close(self) -> None:
        self._parquet_cache.clear()
