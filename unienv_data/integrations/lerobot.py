from typing import Optional, Any, Dict, List, Tuple, Sequence, Union, Literal
from collections import OrderedDict
import importlib.util
import os
import threading

import numpy as np
import pyarrow as pa
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

try:
    import torch
except ImportError:
    torch = None

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


def _load_v3_video_locations(
    dataset_dir: str,
    video_features: Sequence[str],
) -> Dict[int, Dict[str, Tuple[int, int, float, float]]]:
    """
    Read the per-episode video placement of a v3.0 dataset from its episode metadata
    (``videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}`` columns).
    Features without these columns are left out (callers then fall back to the data file).
    """
    locations: Dict[int, Dict[str, Tuple[int, int, float, float]]] = {}
    episodes_dir = os.path.join(dataset_dir, "meta", "episodes")
    if not video_features or not os.path.isdir(episodes_dir):
        return locations
    paths = sorted(
        os.path.join(root, fname)
        for root, _, fnames in os.walk(episodes_dir)
        for fname in fnames if fname.endswith(".parquet")
    )
    for path in paths:
        column_names = set(pq.read_schema(path).names)
        for feat_name in video_features:
            keys = [f"videos/{feat_name}/{c}" for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
            if "episode_index" not in column_names or not all(k in column_names for k in keys):
                continue
            table = pq.read_table(path, columns=["episode_index", *keys]).to_pydict()
            for i, ep_idx in enumerate(table["episode_index"]):
                values = [table[k][i] for k in keys]
                if any(v is None for v in values):
                    continue
                locations.setdefault(int(ep_idx), {})[feat_name] = (
                    int(values[0]), int(values[1]), float(values[2]), float(values[3])
                )
    return locations


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
        video_hwaccel: Optional[Any] = "auto",
    ) -> "LeRobotAsUniEnvDataset[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        """Load a LeRobot dataset from a local directory.

        ``video_hwaccel`` is passed to the video readers: ``"auto"`` (default) uses a hardware
        decoder when one is available, ``None`` decodes in software, which is cheaper to open
        and often faster for small frames read at random positions (the usual training case).
        """
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
            video_hwaccel=video_hwaccel,
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
        video_hwaccel: Optional[Any] = "auto",
    ):
        super().__init__(space)
        self._dataset_dir = dataset_dir
        self._schema = schema
        self._index = index
        self._device = _device
        self._decode_video = decode_video
        self._video_backend = video_backend
        self._video_hwaccel = video_hwaccel

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

        # Simple Parquet cache: episode_index -> {col_name: np.ndarray | torch.Tensor}
        # (the columns of the episode's whole data file, shared with `_parquet_file_cache`)
        self._parquet_cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        # Data file path -> its materialized tabular columns (LRU). v3.0 packs many episodes
        # into one file; they share one read instead of one read per episode.
        self._parquet_file_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        # Data file path -> global frame index of its first row (None if the file has no "index" column)
        self._data_file_first_index: Dict[str, Optional[int]] = {}
        # v3.0: episode_index -> video feature -> (chunk_index, file_index, from_timestamp, to_timestamp)
        self._video_locations: Optional[Dict[int, Dict[str, Tuple[int, int, float, float]]]] = None
        self._parquet_cache_max = 16
        # Open video readers, video path -> reader (LRU); see `_checkout_video_reader`.
        self._video_reader_cache_max = 16
        self._reset_video_readers()
        self._use_torch_fastpath = self.backend.simplified_name == "pytorch" and torch is not None

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
        use_torch_fastpath = self._use_torch_fastpath

        if use_torch_fastpath:
            result: Dict[str, Any] = {}
            for feat_name in self._active_features:
                feat = self._schema.features[feat_name]
                if feat.is_video or feat.is_image:
                    shape = _normalize_image_shape(feat.shape)
                else:
                    shape = feat.shape
                feat_space = self.single_space.spaces[feat_name]
                result[feat_name] = self.backend.zeros(
                    (batch_size, *shape),
                    dtype=feat_space.dtype,
                    device=self._device,
                )
        else:
            # Pre-allocate numpy output arrays, convert to backend at the end.
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
            parquet_data = self._load_parquet_episode(ep_idx)
            file_row_indices = local_indices + self._episode_row_offset(ep_idx)
            row_index_tensor = None
            output_index_tensor_cpu = None
            if use_torch_fastpath and torch is not None:
                row_index_tensor = torch.from_numpy(file_row_indices).to(dtype=torch.long)
                output_index_tensor_cpu = torch.from_numpy(output_positions).to(dtype=torch.long)
            for feat_name in self._tabular_features:
                if feat_name in parquet_data:
                    cached_values = parquet_data[feat_name]
                    if use_torch_fastpath and torch is not None:
                        values = cached_values[row_index_tensor.to(
                            device=cached_values.device if hasattr(cached_values, "device") else "cpu"
                        )]
                        target = result[feat_name]
                        if values.ndim == 1 and target.ndim > 1:
                            # Scalar features (shape [1]) are stored as primitive columns.
                            values = values.reshape(-1, *target.shape[1:])
                        if values.device != target.device:
                            values = values.to(target.device)
                        output_index_tensor = output_index_tensor_cpu.to(device=target.device)
                        target[output_index_tensor] = values
                    else:
                        values = cached_values[file_row_indices]
                        feat_shape = self._schema.features[feat_name].shape
                        if values.ndim == 1 and feat_shape:
                            values = values.reshape(-1, *feat_shape)
                        np_result[feat_name][output_positions] = values

        # Load video features
        for feat_name in self._video_features:
            for ep_idx, (local_indices, output_positions) in episode_groups.items():
                episode_meta = self._index.episodes[ep_idx]
                video_location = self._get_video_location(episode_meta.index, feat_name)
                if video_location is not None:
                    # v3.0 packs videos independently of the data files: use the episode's
                    # video file and its start timestamp within that file.
                    chunk_index, file_index, from_timestamp, to_timestamp = video_location
                    video_fps = self._video_fps(feat_name)
                    first_frame = int(round(from_timestamp * video_fps))
                    file_frame_indices = local_indices + first_frame
                    total_file_frames = max(
                        int(round(to_timestamp * video_fps)),
                        first_frame + int(episode_meta.length),
                    )
                else:
                    dataset_offset = int(episode_meta.dataset_from_index or 0)
                    file_frame_indices = local_indices + dataset_offset
                    total_file_frames = (
                        int(episode_meta.dataset_to_index)
                        if episode_meta.dataset_to_index is not None
                        else int(max(episode_meta.length, int(file_frame_indices.max()) + 1))
                    )
                    chunk_index = episode_meta.data_chunk_index
                    file_index = episode_meta.data_file_index
                video_path = get_video_file_path(
                    self._dataset_dir, self._schema, feat_name, episode_meta.index,
                    file_index=file_index,
                    chunk_index=chunk_index,
                )
                frames = self._decode_video_frames(
                    video_path, file_frame_indices, total_file_frames
                )
                if use_torch_fastpath and torch is not None:
                    frame_tensor = self.backend.from_numpy(frames)
                    if self._device is not None:
                        frame_tensor = self.backend.to_device(frame_tensor, self._device)
                    target = result[feat_name]
                    output_index_tensor = torch.from_numpy(output_positions).to(
                        dtype=torch.long,
                        device=target.device,
                    )
                    target[output_index_tensor] = frame_tensor
                else:
                    np_result[feat_name][output_positions] = frames

        if not use_torch_fastpath:
            # Convert to backend arrays.
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

    def _load_parquet_episode(self, episode_index: int) -> Dict[str, Any]:
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
        data = self._load_parquet_file(path)

        # LRU eviction
        if len(self._parquet_cache) >= self._parquet_cache_max:
            self._parquet_cache.popitem(last=False)
        self._parquet_cache[episode_index] = data

        return data

    def _load_parquet_file(self, path: str) -> Dict[str, Any]:
        """Materialized tabular columns of a whole data file (cached per file, LRU)."""
        data = self._parquet_file_cache.get(path)
        if data is not None:
            self._parquet_file_cache.move_to_end(path)
            return data

        # Only read the active tabular columns (plus "index", for the file's row offset).
        wanted = set(self._tabular_features) | {"index"}
        columns = [name for name in pq.read_schema(path).names if name in wanted]
        table = pq.read_table(path, memory_map=True, columns=columns)
        if path not in self._data_file_first_index:
            self._data_file_first_index[path] = (
                int(table.column("index")[0].as_py())
                if "index" in table.column_names and table.num_rows > 0
                else None
            )

        data = {}
        for col_name in table.column_names:
            if col_name not in self._tabular_features:
                continue
            feat_shape = self._schema.features[col_name].shape
            data[col_name] = self._materialize_tabular_column(table.column(col_name), feat_shape)

        if len(self._parquet_file_cache) >= self._parquet_cache_max:
            _, evicted = self._parquet_file_cache.popitem(last=False)
            # Keep at most `_parquet_cache_max` files alive: drop the episodes still holding it.
            for ep in [ep for ep, cached in self._parquet_cache.items() if cached is evicted]:
                del self._parquet_cache[ep]
        self._parquet_file_cache[path] = data
        return data

    def _episode_row_offset(self, episode_index: int) -> int:
        """Row of the episode's first frame inside its (possibly packed) data file."""
        ep_meta = self._index.episodes[episode_index]
        if ep_meta.dataset_from_index is None:
            return 0
        # v3.0 `dataset_from_index` is a global frame index; packed files restart their rows
        # at 0, so subtract the global index of the file's first row when it is known.
        path = get_data_file_path(
            self._dataset_dir, self._schema, ep_meta.index,
            file_index=ep_meta.data_file_index,
            chunk_index=ep_meta.data_chunk_index,
        )
        if path not in self._data_file_first_index:
            self._load_parquet_episode(episode_index)
        first_index = self._data_file_first_index.get(path)
        return int(ep_meta.dataset_from_index) - (first_index or 0)

    def _video_fps(self, feat_name: str) -> float:
        video_info = self._schema.features[feat_name].video_info
        if video_info is not None and video_info.fps:
            return float(video_info.fps)
        return float(self._schema.fps)

    def _get_video_location(
        self, episode_index: int, feat_name: str
    ) -> Optional[Tuple[int, int, float, float]]:
        if self._schema.codebase_version != LeRobotVersion.V3_0:
            return None
        if self._video_locations is None:
            self._video_locations = _load_v3_video_locations(self._dataset_dir, self._video_features)
        return self._video_locations.get(episode_index, {}).get(feat_name)

    def _materialize_tabular_column(
        self,
        column: pa.ChunkedArray,
        feat_shape: Tuple[int, ...],
    ) -> Any:
        """Convert a parquet column to cached tensor/array with a fast torch path."""
        arr = column.combine_chunks() if hasattr(column, "combine_chunks") else column

        if pa.types.is_fixed_size_list(arr.type):
            return self._materialize_fixed_size_list_column(arr, feat_shape)
        return self._materialize_primitive_column(arr)

    def _materialize_primitive_column(self, arr: pa.Array) -> Any:
        if self._use_torch_fastpath and arr.null_count == 0:
            try:
                return torch.utils.dlpack.from_dlpack(arr)
            except Exception:
                pass
        np_arr = arr.to_numpy(zero_copy_only=False)
        if self._use_torch_fastpath:
            return torch.as_tensor(np_arr.copy())
        return np_arr

    def _materialize_fixed_size_list_column(
        self,
        arr: pa.Array,
        feat_shape: Tuple[int, ...],
    ) -> Any:
        values = arr.values
        if self._use_torch_fastpath and arr.null_count == 0 and values.null_count == 0:
            try:
                torch_values = torch.utils.dlpack.from_dlpack(values)
                return torch_values.reshape(-1, *feat_shape)
            except Exception:
                pass
        np_values = values.to_numpy(zero_copy_only=False).reshape(-1, *feat_shape)
        if self._use_torch_fastpath:
            return torch.as_tensor(np_values.copy())
        return np_values

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
        reader = self._checkout_video_reader(video_path, reader_cls)
        try:
            with reader:
                frames = reader.read(local_indices, total_frames)
        except BaseException:
            # Do not keep a reader in an unknown decoding state.
            reader.close()
            raise
        self._checkin_video_reader(video_path, reader)

        # reader.read returns backend arrays; ensure numpy for assembly
        if not isinstance(frames, np.ndarray):
            frames = NumpyComputeBackend.to_numpy(frames)
        return frames

    # ========== Video reader cache ==========
    # Opening a video container (and its hardware decoding context) costs far more than
    # seeking to and decoding a few frames, so readers are kept open across calls. A reader
    # is taken out of the cache while in use (concurrent reads, e.g. from a CombinedBatch
    # thread pool, open their own) and belongs to the process that opened it: after a fork
    # the inherited readers are neither reused (shared file offsets, device contexts) nor
    # closed by the child, and pickling drops them.

    def _reset_video_readers(self) -> None:
        self._video_reader_cache: OrderedDict[str, Any] = OrderedDict()
        self._video_reader_lock = threading.Lock()
        self._video_reader_pid = os.getpid()
        # Readers inherited through fork: referenced so that they are never closed here.
        self._inherited_video_readers: List[Any] = []

    def _owned_video_reader_cache(self) -> "OrderedDict[str, Any]":
        if self._video_reader_pid != os.getpid():
            inherited = self._inherited_video_readers + list(self._video_reader_cache.values())
            self._reset_video_readers()
            self._inherited_video_readers = inherited
        return self._video_reader_cache

    def _checkout_video_reader(self, video_path: str, reader_cls) -> Any:
        cache = self._owned_video_reader_cache()
        with self._video_reader_lock:
            reader = cache.pop(video_path, None)
        if reader is not None and type(reader) is not reader_cls:
            reader.close()
            reader = None
        if reader is None:
            from unienv_interface.backends.numpy import NumpyComputeBackend
            reader = reader_cls(
                backend=NumpyComputeBackend,
                filename=video_path,
                buffer_pixel_format="rgb24",
                hwaccel=self._video_hwaccel,
                seek_mode="exact",
                device=None,
            )
        return reader

    def _checkin_video_reader(self, video_path: str, reader: Any) -> None:
        cache = self._owned_video_reader_cache()
        to_close = []
        with self._video_reader_lock:
            if video_path in cache or self._video_reader_cache_max <= 0:
                to_close.append(reader)  # another thread returned a reader for this file first
            else:
                cache[video_path] = reader
                while len(cache) > self._video_reader_cache_max:
                    to_close.append(cache.popitem(last=False)[1])
        for r in to_close:
            r.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in ("_video_reader_cache", "_video_reader_lock", "_video_reader_pid", "_inherited_video_readers"):
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_video_readers()

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
        self._parquet_file_cache.clear()
        cache = getattr(self, "_video_reader_cache", None)
        if cache is None:
            return
        if self._video_reader_pid == os.getpid():
            with self._video_reader_lock:
                readers = list(cache.values())
                cache.clear()
            for reader in readers:
                reader.close()
        else:
            self._owned_video_reader_cache()
