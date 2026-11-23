"""Write UniEnv data as LeRobot datasets (format v3.0), natively: the ``lerobot`` package is not needed.

:class:`LeRobotDatasetWriter` writes frame by frame (``add_frame`` / ``save_episode`` / ``finalize``, like
LeRobot's own ``LeRobotDataset``); :func:`export_lerobot` writes a whole replay buffer or dataset. The output
is what LeRobot writes: chunked data parquet files, one MP4 per camera and file holding consecutive episodes,
``meta/info.json``, ``meta/tasks.parquet``, ``meta/episodes/...parquet`` with per-episode statistics, and
``meta/stats.json``; ``LeRobotDataset`` and :class:`~unienv_data.integrations.lerobot.LeRobotAsUniEnvDataset`
read it.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from unienv_interface.space import BinarySpace, BoxSpace, DictSpace, Space
from unienv_interface.utils.png import encode_png

__all__ = ["LeRobotDatasetWriter", "export_lerobot", "lerobot_features"]

CODEBASE_VERSION = "v3.0"
#: Columns every LeRobot dataset has; the writer fills them in
STANDARD_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
CHUNK_FILE = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DATA_PATH = "data/" + CHUNK_FILE + ".parquet"
VIDEO_PATH = "videos/{video_key}/" + CHUNK_FILE + ".mp4"
EPISODES_PATH = "meta/episodes/" + CHUNK_FILE + ".parquet"
_STAGING = ".unienv-staging"
_ARROW_TYPES = {
    "float16": pa.float16(), "float32": pa.float32(), "float64": pa.float64(),
    "int8": pa.int8(), "int16": pa.int16(), "int32": pa.int32(), "int64": pa.int64(),
    "uint8": pa.uint8(), "uint16": pa.uint16(), "uint32": pa.uint32(), "uint64": pa.uint64(), "bool": pa.bool_(),
}
#: Video encoders, preferred first: (PyAV codec, LeRobot's codec name, encoder options as LeRobot sets them)
_ENCODERS = (
    ("libsvtav1", "av1", {"g": 2, "crf": 30, "preset": 12}),
    ("libx264", "h264", {"g": 2, "crf": 23, "preset": "medium"}),
)


# ---------------------------------------------------------------------------- features


@dataclass
class _Feature:
    name: str  # LeRobot feature key
    path: Tuple[str, ...]  # where in a frame
    kind: str  # "video", "image" or "value"
    dtype: str  # LeRobot dtype
    shape: Tuple[int, ...]  # LeRobot shape: images (H, W, C), scalars (1,)
    names: Optional[List[str]]
    space: Space
    layout: str = "HWC"  # images: the space's layout
    bgr: bool = False  # images: the space stores BGR(A)

    def schema(self) -> Dict[str, Any]:
        if self.kind in ("video", "image"):
            return {"dtype": self.kind, "shape": list(self.shape), "names": ["height", "width", "channels"]}
        return {"dtype": self.dtype, "shape": list(self.shape), "names": self.names}


def _leaves(space: Space, path: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], Space]:
    if isinstance(space, DictSpace):
        found: Dict[Tuple[str, ...], Space] = {}
        for key, sub in space.spaces.items():
            found.update(_leaves(sub, path + (str(key),)))
        return found
    return {path: space}


def _numpy_dtype(space: Space) -> np.dtype:
    return np.asarray(space.backend.to_numpy(space.backend.zeros((1,), dtype=space.dtype, device=space.device))).dtype


def _describe_leaf(name: str, path: Tuple[str, ...], space: Space, videos: bool) -> Union[_Feature, str]:
    """The feature a leaf becomes, or why it cannot be written."""
    md = dict(getattr(space, "metadata", None) or {})
    if isinstance(space, BinarySpace):
        shape = tuple(space.shape) or (1,)
        return _Feature(name, path, "value", "bool", shape, None, space)
    if not isinstance(space, BoxSpace):
        if type(space).__name__ == "TextSpace":
            return "text: give it as each frame's task instead"
        return f"{type(space).__name__} has no LeRobot feature type"
    dtype = _numpy_dtype(space)
    shape = tuple(space.shape)
    kind = md.get("kind")
    if kind == "depth":
        return "depth images are not supported"
    layout = str(md.get("layout", "")).upper()
    looks_like_image = dtype == np.uint8 and len(shape) == 3 and (
        shape[-1] in (3, 4) if layout != "CHW" else shape[0] in (3, 4))
    if kind == "image" or (kind is None and looks_like_image and min(shape[:2] if layout != "CHW" else shape[1:]) >= 16):
        if dtype != np.uint8 or len(shape) != 3:
            return f"images must be uint8 with 3 axes, got {dtype} {shape}"
        height, width, channels = (shape[1], shape[2], shape[0]) if layout == "CHW" else shape
        fmt = str(md.get("format", "rgb")).lower()
        if channels != 3 and not (channels == 4 and fmt in ("rgba", "bgra")):
            return f"images must have 3 (RGB) channels, got {channels}"
        if videos and (height % 2 or width % 2):
            return f"video frames need an even height and width, got {height}x{width} (use videos=False)"
        return _Feature(name, path, "video" if videos else "image", "uint8", (height, width, 3), None, space,
                        layout="CHW" if layout == "CHW" else "HWC", bgr=fmt in ("bgr", "bgra"))
    if dtype.name not in _ARROW_TYPES:
        return f"dtype {dtype} has no LeRobot equivalent"
    names = md.get("names")
    if not (isinstance(names, (list, tuple)) and len(shape) == 1 and len(names) == shape[0]):
        names = None
    return _Feature(name, path, "value", dtype.name, shape or (1,), list(names) if names else None, space)


def _features(space: Space, features: Optional[Mapping[str, Union[str, Sequence[str]]]], videos: bool,
              ) -> Tuple[List[_Feature], Dict[str, str]]:
    leaves = _leaves(space)
    found: List[_Feature] = []
    skipped: Dict[str, str] = {}
    if features is None:
        targets = {".".join(path) or "observation": path for path in leaves}
    else:
        targets = {}
        for name, path in features.items():
            path = tuple(p for p in path.split("/") if p) if isinstance(path, str) else tuple(path)
            if path not in leaves:
                raise KeyError(f"features[{name!r}] = {'/'.join(path)!r} is not a leaf of the space; its leaves are "
                               f"{sorted('/'.join(p) for p in leaves)}")
            targets[name] = path
    for name, path in targets.items():
        if name in STANDARD_FEATURES:
            if features is not None:
                raise ValueError(f"{name!r} is a column every LeRobot dataset has; the writer fills it in")
            skipped[name] = "a standard LeRobot column, filled in by the writer"
            continue
        described = _describe_leaf(name, path, leaves[path], videos)
        if isinstance(described, str):
            if features is not None:
                raise ValueError(f"Cannot write {name!r} ({'/'.join(path)}): {described}")
            skipped[name] = described
        else:
            found.append(described)
    if not found:
        raise ValueError(f"Nothing to write: no leaf of the space becomes a LeRobot feature ({skipped})")
    return found, skipped


def lerobot_features(space: Space, *, features: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                     videos: bool = True) -> Dict[str, Dict[str, Any]]:
    """The LeRobot features (``info.json``'s ``features``, without the standard columns) ``space`` is written as."""
    return {f.name: f.schema() for f in _features(space, features, videos)[0]}


# ---------------------------------------------------------------------------- statistics


def _num_samples(length: int, low: int = 100, high: int = 10_000, power: float = 0.75) -> int:
    """How many frames image statistics sample (LeRobot's heuristic)."""
    if length < low:
        low = length
    return max(low, min(int(length ** power), high))


def _downsample(image: np.ndarray, target: int = 150, threshold: int = 300) -> np.ndarray:
    """Images (C, H, W) at least ``threshold`` pixels a side are strided down to about ``target`` (as LeRobot)."""
    _, height, width = image.shape
    if max(height, width) < threshold:
        return image
    factor = int(width / target) if width > height else int(height / target)
    return image[:, ::factor, ::factor]


def _stats(array: np.ndarray, axis: Union[int, Tuple[int, ...]], keepdims: bool, count: int) -> Dict[str, np.ndarray]:
    array = array.astype(np.float64)
    stats = {
        "min": np.min(array, axis=axis, keepdims=keepdims), "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims), "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([count], dtype=np.int64),
    }
    for q in QUANTILES:
        stats[f"q{int(q * 100):02d}"] = np.quantile(array, q, axis=axis, keepdims=keepdims)
    if keepdims:  # images: drop the sample axis, keep (C, 1, 1)
        stats = {k: (v[0] if k != "count" else v) for k, v in stats.items()}
    return stats


def _aggregate(per_episode: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Dataset statistics from per-episode ones, as LeRobot aggregates them."""
    means = np.stack([s["mean"] for s in per_episode])
    variances = np.stack([s["std"] ** 2 for s in per_episode])
    counts = np.stack([s["count"] for s in per_episode])
    total = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = counts[..., None]
    mean = (means * counts).sum(axis=0) / total
    variance = ((variances + (means - mean) ** 2) * counts).sum(axis=0) / total
    result = {"min": np.min(np.stack([s["min"] for s in per_episode]), axis=0),
              "max": np.max(np.stack([s["max"] for s in per_episode]), axis=0),
              "mean": mean, "std": np.sqrt(variance), "count": total}
    for q in QUANTILES:  # quantiles of quantiles: LeRobot keeps a conservative envelope
        key = f"q{int(q * 100):02d}"
        values = np.stack([s[key] for s in per_episode])
        result[key] = np.min(values, axis=0) if q <= 0.5 else np.max(values, axis=0)
    return result


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------- videos


def _video_encoder(preferred: Optional[str]) -> Tuple[str, str, Dict[str, Any]]:
    import av

    candidates = _ENCODERS if preferred is None else [e for e in _ENCODERS if preferred in (e[0], e[1])] or [
        (preferred, preferred, {"g": 2})]
    for codec, name, options in candidates:
        try:
            av.codec.Codec(codec, "w")
            return codec, name, dict(options)
        except Exception:
            continue
    raise RuntimeError(f"No usable video encoder among {[c[0] for c in candidates]} in this FFmpeg build; "
                       "pass videos=False to store frames as PNG images instead")


class _EpisodeVideo:
    """One camera's frames of the current episode, encoded to a temporary MP4 as they arrive."""

    def __init__(self, path: Path, feature: _Feature, fps: int, codec: str, options: Dict[str, Any]) -> None:
        import av

        self.path = path
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.height, self.stream.width = feature.shape[0], feature.shape[1]
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {k: str(v) for k, v in options.items()}
        self.fps = fps
        self.frames = 0

    def add(self, image: np.ndarray) -> None:
        import av

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        frame.pts = self.frames
        frame.time_base = Fraction(1, self.fps)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def _concatenate_videos(inputs: Sequence[Path], output: Path) -> None:
    """Concatenate MP4s of the same encoding into ``output`` by stream copy (FFmpeg's concat demuxer)."""
    import av

    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False) as listing:
        listing.write("ffconcat version 1.0\n")
        for path in inputs:
            listing.write(f"file '{Path(path).resolve()}'\n")
    joined = output.with_suffix(".joining.mp4")
    try:
        source = av.open(listing.name, mode="r", format="concat", options={"safe": "0"})
        target = av.open(str(joined), mode="w", options={"movflags": "faststart"})
        streams = {}
        for stream in source.streams.video:
            streams[stream.index] = target.add_stream_from_template(template=stream, opaque=True)
            streams[stream.index].time_base = stream.time_base
        for packet in source.demux():
            if packet.stream.index not in streams or packet.dts is None:
                continue
            packet.stream = streams[packet.stream.index]
            target.mux(packet)
        source.close()
        target.close()
        os.replace(joined, output)
    finally:
        os.unlink(listing.name)
        if joined.exists():
            joined.unlink()


# ---------------------------------------------------------------------------- the writer


class _Files:
    """Rotation through ``chunk-XXX/file-YYY`` files once one exceeds a size."""

    def __init__(self, chunks_size: int, limit_mb: float) -> None:
        self.chunks_size, self.limit = chunks_size, limit_mb * 1024 * 1024
        self.chunk = self.file = 0
        self.started = False

    def current(self) -> Tuple[int, int]:
        return self.chunk, self.file

    def advance(self) -> None:
        self.file += 1
        if self.file >= self.chunks_size:
            self.chunk, self.file = self.chunk + 1, 0


class LeRobotDatasetWriter:
    """Writes a LeRobot dataset (v3.0) frame by frame.

    Example::

        with LeRobotDatasetWriter.create("datasets/pick", step_space, fps=20) as writer:
            for episode in episodes:
                for frame in episode:
                    writer.add_frame(frame, task="pick up the bowl")
                writer.save_episode()

    ``space`` is the space of one frame. Each leaf becomes a LeRobot feature named by its key path joined with
    dots (``obs/state`` becomes ``obs.state``), or as ``features`` maps LeRobot names to key paths
    (``{"observation.state": "obs/state", "observation.images.front": "obs/front", "action": "action"}``;
    only those are written). Image leaves (space metadata ``kind="image"``, or unannotated ``uint8`` ``H x W x 3``
    boxes) become ``video`` features (MP4, AV1 as LeRobot encodes them, or H.264), or with ``videos=False`` PNG
    ``image`` features; other boxes numeric features with their dtype, shape and ``names`` metadata; binary
    leaves ``bool`` features. Text, depth and other leaves are skipped (:attr:`skipped` says why); give the
    language instruction as each frame's ``task``. LeRobot's standard columns (``timestamp``, ``frame_index``,
    ``episode_index``, ``index``, ``task_index``) are filled in.

    :meth:`finalize` writes the metadata; without it (or the ``with`` block ending) the dataset is incomplete.
    """

    def __init__(self, root: Path, space: Space, fps: int, features: List[_Feature], skipped: Dict[str, str], *,
                 robot_type: Optional[str], video_codec: Optional[str], chunks_size: int,
                 data_files_size_in_mb: float, video_files_size_in_mb: float) -> None:
        self.root, self.space, self.fps = root, space, fps
        self._features = features
        self.skipped = skipped
        self.robot_type = robot_type
        self.chunks_size = chunks_size
        self.data_files_size_in_mb, self.video_files_size_in_mb = data_files_size_in_mb, video_files_size_in_mb
        self._videos = [f for f in features if f.kind == "video"]
        self._encoder = _video_encoder(video_codec) if self._videos else None
        self._staging = root / _STAGING
        self._staging.mkdir(parents=True, exist_ok=True)
        self._tasks: Dict[str, int] = {}
        self._episodes: List[Dict[str, Any]] = []
        self._episode_stats: List[Dict[str, Dict[str, np.ndarray]]] = []
        self._num_frames = 0
        # Data files: an open parquet writer appending episodes (row groups) until the file is full
        self._data = _Files(chunks_size, data_files_size_in_mb)
        self._data_writer: Optional[pq.ParquetWriter] = None
        # Videos: per camera, the frames already in its current file
        self._video_files = {f.name: _Files(chunks_size, video_files_size_in_mb) for f in self._videos}
        self._video_frames = {f.name: 0 for f in self._videos}
        self._finalized = False
        self._reset_episode()

    @classmethod
    def create(cls, root: Union[str, os.PathLike], space: Space, fps: int, *,
               features: Optional[Mapping[str, Union[str, Sequence[str]]]] = None, videos: bool = True,
               robot_type: Optional[str] = None, video_codec: Optional[str] = None, chunks_size: int = 1000,
               data_files_size_in_mb: float = 100, video_files_size_in_mb: float = 200,
               overwrite: bool = False) -> "LeRobotDatasetWriter":
        """Start a dataset in the directory ``root`` (which must not exist, unless ``overwrite``).

        ``video_codec``: ``"av1"`` / ``"libsvtav1"`` (LeRobot's default) or ``"h264"`` / ``"libx264"``; by default
        the first available. Data and video files hold consecutive episodes up to the given sizes.
        """
        root = Path(root)
        if root.exists():
            if not overwrite:
                raise FileExistsError(f"{root} exists; pass overwrite=True to replace it")
            shutil.rmtree(root)
        if not (isinstance(fps, int) and fps > 0):
            raise ValueError(f"fps must be a positive integer, got {fps!r}")
        found, skipped = _features(space, features, videos)
        return cls(root, space, fps, found, skipped, robot_type=robot_type, video_codec=video_codec,
                   chunks_size=chunks_size, data_files_size_in_mb=data_files_size_in_mb,
                   video_files_size_in_mb=video_files_size_in_mb)

    # ------------------------------------------------------------------ properties

    @property
    def features(self) -> Dict[str, Dict[str, Any]]:
        """``info.json``'s features: the written ones, then the standard columns."""
        written = {}
        for feature in self._features:
            schema = feature.schema()
            if feature.kind == "video":
                codec, name, options = self._encoder
                schema["info"] = {"video.height": feature.shape[0], "video.width": feature.shape[1],
                                  "video.codec": name, "video.pix_fmt": "yuv420p", "video.fps": self.fps,
                                  "video.channels": 3, "has_audio": False,
                                  **{f"video.{k}": v for k, v in options.items()}, "is_depth_map": False}
            written[feature.name] = schema
        return {**written, **{k: dict(v) for k, v in STANDARD_FEATURES.items()}}

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def episode_frames(self) -> int:
        """Frames added to the episode in progress."""
        return self._length

    # ------------------------------------------------------------------ frames

    def _reset_episode(self) -> None:
        self._length = 0
        self._columns: Dict[str, List[Any]] = {f.name: [] for f in self._features if f.kind != "video"}
        self._episode_tasks: List[str] = []
        self._samples: Dict[str, List[np.ndarray]] = {f.name: [] for f in self._features if f.kind in ("video", "image")}
        self._open_videos: Dict[str, _EpisodeVideo] = {}

    def _check_open(self) -> None:
        if self._finalized:
            raise RuntimeError("The dataset is finalized; start a new writer to write more")

    def _image(self, feature: _Feature, value: np.ndarray) -> np.ndarray:
        """An RGB ``(H, W, 3)`` uint8 image from a leaf's value."""
        image = np.asarray(value)
        if feature.layout == "CHW":
            image = np.moveaxis(image, 0, -1)
        if feature.bgr:
            image = image[..., [2, 1, 0]]
        image = image[..., :3]
        if image.shape != feature.shape:
            raise ValueError(f"{feature.name}: expected an image of shape {feature.shape}, got {image.shape}")
        return np.ascontiguousarray(image.astype(np.uint8, copy=False))

    def _leaf(self, feature: _Feature, data: Any) -> np.ndarray:
        value = data
        for key in feature.path:
            value = value[key]
        return np.asarray(feature.space.backend.to_numpy(value))

    def add_frames(self, frames: Any, task: Union[str, Sequence[str]]) -> None:
        """Add a batch of consecutive frames (a leading axis on every leaf) to the episode in progress;
        ``task`` is their instruction (one string, or one per frame)."""
        self._check_open()
        self._add({f.name: self._leaf(f, frames) for f in self._features}, task)

    def add_frame(self, frame: Any, task: str) -> None:
        """Add one frame (data of the frame space) to the episode in progress; ``task`` is its instruction."""
        self._check_open()
        self._add({f.name: self._leaf(f, frame)[None] for f in self._features}, task)

    def _add(self, arrays: Dict[str, np.ndarray], task: Union[str, Sequence[str]]) -> None:
        counts = {array.shape[0] for array in arrays.values()}
        if len(counts) != 1:
            raise ValueError(f"The features have different numbers of frames: { {k: v.shape[0] for k, v in arrays.items()} }")
        count = counts.pop()
        tasks = [task] * count if isinstance(task, str) else list(task)
        if len(tasks) != count or not all(isinstance(t, str) for t in tasks):
            raise ValueError(f"task must be a string, or one string per frame ({count})")
        for feature in self._features:
            array = arrays[feature.name]
            if feature.kind == "value":
                expected = (count,) + (feature.shape if feature.shape != (1,) or array.ndim > 1 else ())
                if array.size != int(np.prod(expected)):
                    raise ValueError(f"{feature.name}: expected {count} frames of shape {feature.shape}, got {array.shape}")
                values = array.reshape(count, *feature.shape)
                values = values.reshape(count) if feature.shape == (1,) else values
                self._columns[feature.name].append(values.astype(np.dtype(feature.dtype), copy=False))
                continue
            for i in range(count):
                image = self._image(feature, array[i])
                self._samples[feature.name].append(_downsample(np.moveaxis(image, -1, 0)))
                if feature.kind == "image":
                    index = self._length + i
                    self._columns[feature.name].append({"bytes": encode_png(image), "path": f"frame-{index:06d}.png"})
                else:
                    video = self._open_videos.get(feature.name)
                    if video is None:
                        codec, _, options = self._encoder
                        path = self._staging / f"{feature.name}-{len(self._episodes)}.mp4"
                        video = self._open_videos[feature.name] = _EpisodeVideo(path, feature, self.fps, codec, options)
                    video.add(image)
        self._episode_tasks.extend(tasks)
        self._length += count

    def clear_episode(self) -> None:
        """Discard the frames of the episode in progress."""
        for video in self._open_videos.values():
            video.close()
            video.path.unlink(missing_ok=True)
        self._reset_episode()

    def save_episode(self) -> int:
        """Write the episode in progress and start the next one; returns its episode index."""
        self._check_open()
        length = self._length
        if length == 0:
            raise ValueError("The episode has no frames")
        episode = len(self._episodes)
        start = self._num_frames
        names: List[str] = []  # tasks join the dataset only once an episode using them is saved
        for text in self._episode_tasks:
            if text not in self._tasks:
                self._tasks[text] = len(self._tasks)
            if text not in names:
                names.append(text)
        frame_index = np.arange(length, dtype=np.int64)
        standard = {
            "timestamp": (frame_index / self.fps).astype(np.float32),
            "frame_index": frame_index,
            "episode_index": np.full(length, episode, dtype=np.int64),
            "index": np.arange(start, start + length, dtype=np.int64),
            "task_index": np.asarray([self._tasks[t] for t in self._episode_tasks], dtype=np.int64),
        }
        record: Dict[str, Any] = {"episode_index": episode, "tasks": names, "length": length}
        # Data: one row group appended to the current file
        table = self._table(standard)
        chunk, file = self._write_data(table)
        record.update({"data/chunk_index": chunk, "data/file_index": file,
                       "dataset_from_index": start, "dataset_to_index": start + length})
        # Videos: the episode's MP4 appended to the camera's current file
        for feature in self._videos:
            video = self._open_videos.pop(feature.name)
            video.close()
            chunk, file, first = self._append_video(feature.name, video.path, length)
            record.update({f"videos/{feature.name}/chunk_index": chunk, f"videos/{feature.name}/file_index": file,
                           f"videos/{feature.name}/from_timestamp": first / self.fps,
                           f"videos/{feature.name}/to_timestamp": (first + length) / self.fps})
        record.update({"meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
        stats = self._stats(standard, length)
        for name, values in stats.items():
            for key, value in values.items():
                record[f"stats/{name}/{key}"] = value
        self._episodes.append(record)
        self._episode_stats.append(stats)
        self._num_frames += length
        self._reset_episode()
        return episode

    def _table(self, standard: Dict[str, np.ndarray]) -> pa.Table:
        columns, fields = [], []
        for feature in self._features:
            if feature.kind == "video":
                continue
            values = self._columns[feature.name]
            if feature.kind == "image":
                array = pa.array(values, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
            else:
                data = np.concatenate(values, axis=0)
                array = pa.array(data.reshape(-1), type=_ARROW_TYPES[feature.dtype])
                for size in reversed(feature.shape if feature.shape != (1,) else ()):
                    array = pa.FixedSizeListArray.from_arrays(array, size)
            columns.append(array)
            fields.append(pa.field(feature.name, array.type))
        for name, values in standard.items():
            array = pa.array(values, type=_ARROW_TYPES[STANDARD_FEATURES[name]["dtype"]])
            columns.append(array)
            fields.append(pa.field(name, array.type))
        schema = pa.schema(fields, metadata={b"huggingface": json.dumps({"info": {"features": self._hf_features()}})
                                             .encode()})
        return pa.Table.from_arrays(columns, schema=schema)

    def _hf_features(self) -> Dict[str, Any]:
        """The Hugging Face ``datasets`` features of the data files (as LeRobot declares them)."""
        hf: Dict[str, Any] = {}
        for name, schema in self.features.items():
            dtype, shape = schema["dtype"], tuple(schema["shape"])
            if dtype == "video":
                continue
            if dtype == "image":
                hf[name] = {"_type": "Image"}
            elif shape == (1,):
                hf[name] = {"dtype": dtype, "_type": "Value"}
            elif len(shape) == 1:
                hf[name] = {"feature": {"dtype": dtype, "_type": "Value"}, "length": shape[0], "_type": "List"}
            else:
                hf[name] = {"shape": list(shape), "dtype": dtype, "_type": f"Array{len(shape)}D"}
        return hf

    def _write_data(self, table: pa.Table) -> Tuple[int, int]:
        files = self._data
        if self._data_writer is not None:
            path = self.root / DATA_PATH.format(chunk_index=files.chunk, file_index=files.file)
            if path.stat().st_size >= files.limit:
                self._data_writer.close()
                self._data_writer = None
                files.advance()
        if self._data_writer is None:
            path = self.root / DATA_PATH.format(chunk_index=files.chunk, file_index=files.file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._data_writer = pq.ParquetWriter(str(path), table.schema)
        self._data_writer.write_table(table)
        return files.current()

    def _append_video(self, name: str, episode_path: Path, length: int) -> Tuple[int, int, int]:
        """Append an episode's MP4 to the camera's current file; returns (chunk, file, first frame in the file)."""
        files = self._video_files[name]
        path = self.root / VIDEO_PATH.format(video_key=name, chunk_index=files.chunk, file_index=files.file)
        if path.exists() and path.stat().st_size >= files.limit:
            files.advance()
            self._video_frames[name] = 0
            path = self.root / VIDEO_PATH.format(video_key=name, chunk_index=files.chunk, file_index=files.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        first = self._video_frames[name]
        if path.exists():
            _concatenate_videos([path, episode_path], path)
            episode_path.unlink()
        else:
            _concatenate_videos([episode_path], path)  # remux with faststart
            episode_path.unlink()
        self._video_frames[name] = first + length
        return files.chunk, files.file, first

    def _stats(self, standard: Dict[str, np.ndarray], length: int) -> Dict[str, Dict[str, np.ndarray]]:
        stats: Dict[str, Dict[str, np.ndarray]] = {}
        for feature in self._features:
            if feature.kind == "value":
                data = np.concatenate(self._columns[feature.name], axis=0).reshape(length, *feature.shape)
                stats[feature.name] = _stats(data, 0, False, length)
            else:
                frames = self._samples[feature.name]
                picks = np.round(np.linspace(0, len(frames) - 1, _num_samples(len(frames)))).astype(int)
                sample = np.stack([frames[i] for i in picks]).astype(np.float64) / 255.0
                stats[feature.name] = _stats(sample, (0, 2, 3), True, length)
        for name, values in standard.items():
            stats[name] = _stats(values.reshape(length, 1), 0, False, length)
        return stats

    # ------------------------------------------------------------------ metadata

    def finalize(self) -> None:
        """Write the metadata; the dataset is then complete. An episode in progress is discarded."""
        if self._finalized:
            return
        if self._length:
            self.clear_episode()
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None
        if not self._episodes:
            raise ValueError("No episode was saved")
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        episodes_path = self.root / EPISODES_PATH.format(chunk_index=0, file_index=0)
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        columns = {key: [episode[key] for episode in self._episodes] for key in self._episodes[0]}
        pq.write_table(pa.table({k: pa.array(_json(v)) for k, v in columns.items()}), str(episodes_path))
        self._write_tasks(meta / "tasks.parquet")
        stats = {name: _aggregate([episode[name] for episode in self._episode_stats])
                 for name in self._episode_stats[0]}
        (meta / "stats.json").write_text(json.dumps(_json(stats), indent=4))
        info = {
            "codebase_version": CODEBASE_VERSION, "robot_type": self.robot_type, "fps": self.fps,
            "total_episodes": len(self._episodes), "total_frames": self._num_frames, "total_tasks": len(self._tasks),
            "chunks_size": self.chunks_size, "data_files_size_in_mb": self.data_files_size_in_mb,
            "video_files_size_in_mb": self.video_files_size_in_mb, "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if self._videos else None,
            "splits": {"train": f"0:{len(self._episodes)}"}, "features": self.features,
        }
        (meta / "info.json").write_text(json.dumps(info, indent=4))
        shutil.rmtree(self._staging, ignore_errors=True)
        self._finalized = True

    def _write_tasks(self, path: Path) -> None:
        ordered = sorted(self._tasks.items(), key=lambda item: item[1])
        try:
            import pandas as pd
            frame = pd.DataFrame({"task_index": [i for _, i in ordered]}, index=pd.Index([t for t, _ in ordered],
                                                                                           name="task"))
            pq.write_table(pa.Table.from_pandas(frame), str(path))
        except ImportError:  # same columns, without pandas' index metadata
            pq.write_table(pa.table({"task_index": pa.array([i for _, i in ordered], pa.int64()),
                                     "task": pa.array([t for t, _ in ordered], pa.string())}), str(path))

    def push_to_hub(self, repo_id: str, *, private: bool = False, token: Optional[str] = None,
                    branch: Optional[str] = None, commit_message: str = "Upload dataset (UniEnv)",
                    license: Optional[str] = "apache-2.0", tags: Sequence[str] = ()) -> str:
        """Upload the finalized dataset to the Hugging Face Hub as a dataset repository; returns its URL.

        Adds a dataset card if the directory has none, and tags the revision with the codebase version
        (``v3.0``), which LeRobot looks for when it loads datasets from the Hub.
        """
        if not self._finalized:
            raise RuntimeError("finalize() the dataset before pushing it")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        if branch is not None:
            api.create_branch(repo_id, branch=branch, repo_type="dataset", exist_ok=True)
        card = self.root / "README.md"
        if not card.exists():
            card.write_text(self._card(repo_id, license, tags))
        api.upload_folder(repo_id=repo_id, folder_path=str(self.root), repo_type="dataset", revision=branch,
                          commit_message=commit_message, ignore_patterns=[f"{_STAGING}/*"])
        api.create_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset", revision=branch, exist_ok=True)
        return f"https://huggingface.co/datasets/{repo_id}"

    def _card(self, repo_id: str, license: Optional[str], tags: Sequence[str]) -> str:
        front = ["---"]
        if license:
            front.append(f"license: {license}")
        front += ["task_categories:", "- robotics", "tags:", "- LeRobot", "- UniEnv", *[f"- {t}" for t in tags],
                  "configs:", "- config_name: default", "  data_files: data/*/*.parquet", "---"]
        return "\n".join(front) + (
            f"\n\n# {repo_id}\n\nA [LeRobot](https://github.com/huggingface/lerobot) dataset (format {CODEBASE_VERSION}) "
            f"written with [UniEnv](https://github.com/UniEnvOrg/UniEnv): {len(self._episodes)} episodes, "
            f"{self._num_frames} frames at {self.fps} fps, {len(self._tasks)} tasks.\n\n"
            f"```json\n{json.dumps(self.features, indent=2)}\n```\n")

    def __enter__(self) -> "LeRobotDatasetWriter":
        return self

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        if kind is None:
            self.finalize()
        else:  # keep what was saved, discard the episode in progress
            self.clear_episode()
            if self._data_writer is not None:
                self._data_writer.close()
                self._data_writer = None


# ---------------------------------------------------------------------------- export


def _episode_ranges(source: Any) -> List[Tuple[int, int, Any]]:
    """``(start, end, record)`` per episode of a replay buffer or LeRobot dataset."""
    segments = getattr(source, "get_episode_segments", None)
    if callable(segments):
        try:
            found = segments()
        except Exception:
            found = None
        if found:
            return [(int(s), int(e), source.get_episode_data(int(i))) for s, e, i in found]
    plain = getattr(source, "get_segments", None)
    found = plain() if callable(plain) else None
    if found:
        return [(int(s), int(e), None) for s, e in found]
    raise ValueError("Cannot tell the episodes of this source: pass episodes=[(start, end), ...]")


def _task_of(record: Any, task: Union[None, str, Callable[[Any], str]]) -> str:
    if callable(task):
        return str(task(record))
    if isinstance(task, str):
        return task
    if isinstance(record, str):
        return record
    if isinstance(record, Mapping):
        for key in ("task", "language_instruction", "instruction"):
            value = record.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
                return str(value.reshape(-1)[0])
    return ""


def export_lerobot(source: Any, root: Union[str, os.PathLike], fps: int, *,
                   task: Union[None, str, Callable[[Any], str]] = None,
                   episodes: Optional[Sequence[Tuple[int, int]]] = None,
                   features: Optional[Mapping[str, Union[str, Sequence[str]]]] = None, videos: bool = True,
                   robot_type: Optional[str] = None, batch_frames: int = 256, **create_kwargs: Any,
                   ) -> LeRobotDatasetWriter:
    """Write ``source`` (a replay buffer, a LeRobot dataset, or any UniEnv batch) as a LeRobot dataset.

    Episodes are the source's episodes (a replay buffer's episode segments, or its segments; a LeRobot
    dataset's episodes) or ``episodes=[(start, end), ...]``. Each episode's task is ``task`` (a string, or a
    function of the episode record), by default its record's ``"task"`` / ``"language_instruction"`` /
    ``"instruction"`` (or the record itself if it is a string). Other arguments are those of
    :meth:`LeRobotDatasetWriter.create`. Returns the finalized writer (e.g. to ``push_to_hub``).
    """
    ranges = [(int(s), int(e), None) for s, e in episodes] if episodes is not None else _episode_ranges(source)
    writer = LeRobotDatasetWriter.create(root, source.single_space, fps, features=features, videos=videos,
                                         robot_type=robot_type, **create_kwargs)
    with writer:
        for start, end, record in ranges:
            text = _task_of(record, task)
            for first in range(start, end, batch_frames):
                writer.add_frames(source.get_at(slice(first, min(first + batch_frames, end))), text)
            writer.save_episode()
    return writer
