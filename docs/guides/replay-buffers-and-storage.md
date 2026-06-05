# Replay Buffers And Storage

`unienv_data` brings the same typed-space approach to stored data.

## `BatchBase`

`BatchBase` is the common interface for dataset-like collections of structured samples.

It gives you a uniform API for:

- indexed reads
- structured or flattened access
- metadata-aware retrieval
- column views and sliced views
- extension from other batches

Because a batch knows the `Space` of one sample, UniEnv can keep structured data and flattened tensor views in sync.

## `ReplayBuffer`

`ReplayBuffer` is the main mutable buffer abstraction.

Key properties:

- ring-buffer overwrite semantics
- persistence metadata
- pluggable backing storage
- optional multiprocessing-safe counters and locking
- step-wise `append(value)` with implicit segment starts
- `mark_segment_end()` to finalize the current segment
- `get_segments()` for segment-aware storages, or `None` for legacy/default ones

Typical creation pattern:

```python
import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space.spaces import BoxSpace, DictSpace
from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.parquet import ParquetStorage

space = DictSpace(
    NumpyComputeBackend,
    {
        "obs": BoxSpace(NumpyComputeBackend, 0.0, 1.0, np.float32, shape=(4,)),
        "action": BoxSpace(NumpyComputeBackend, -1.0, 1.0, np.float32, shape=(2,)),
    },
)

buffer = ReplayBuffer.create(
    ParquetStorage,
    space,
    cache_path="cache/replay",
    capacity=100_000,
)
```

Load a persisted buffer with:

```python
restored = ReplayBuffer.load_from(
    "cache/replay",
    backend=NumpyComputeBackend,
)
```

Use `append(value)` for one unbatched step/sample at a time. The first append after creation, `clear()`, or the previous `mark_segment_end()` opens a segment implicitly.

```python
import torch

from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space.spaces import BoxSpace
from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.pytorch import PytorchTensorStorage

space = BoxSpace(PyTorchComputeBackend, -10, 10, torch.int64, shape=(2,))
buffer = ReplayBuffer.create(
    PytorchTensorStorage,
    space,
    capacity=4,
    is_memmap=False,
)

buffer.append(torch.tensor([1, 2], dtype=torch.int64))
buffer.append(torch.tensor([3, 4], dtype=torch.int64))
buffer.mark_segment_end()

assert buffer.get_segments() is None
```

For segment-aware storages, `ReplayBuffer.get_segments()` returns logical chronological half-open ranges. With fixed capacity, round-robin overwrite is supported and the ranges stay in logical buffer order.

> Visibility note: the base storage append path writes through the active index and may make values visible immediately, but some implementations (notably `VideoStorage`) buffer data until `mark_segment_end()`.

## Storage Backends

UniEnv includes multiple storage implementations, each useful for a different tradeoff.

- `ParquetStorage`: columnar storage for NumPy-friendly structured data
- `HDF5Storage`: HDF5-backed persistence
- `FlattenedStorage`: storage built around flattened samples, often paired with another inner storage
- `PytorchTensorStorage`: tensor or memmap-backed PyTorch storage
- `NPZStorage`: `.npz`-based storage
- `ImageStorage` and `VideoStorage`: media-oriented storage
- `DictStorage`: structured composition over multiple sub-storages

The storage layer is intentionally separate from the replay-buffer semantics, so you can swap persistence formats without changing higher-level code.

`VideoStorage` streams frames into a temporary file for the active segment and only registers the final file on `mark_segment_end()`. Do not read from the active segment before finalizing it.

```python
from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.video_storage import VideoStorage

rb = ReplayBuffer.create(
    VideoStorage,
    space,  # e.g. BoxSpace(..., shape=(H, W, 3))
    cache_path="cache/video",
    capacity=None,
    codec="libx264rgb",          # RGB video
    file_pixel_format="rgb24",
    buffer_pixel_format="rgb24",
    file_ext="mkv",
)

for frame in frames:
    rb.append(frame)
rb.mark_segment_end()

print(rb.storage.get_segments())  # backend-specific physical file ranges
print(rb.get_segments())          # logical half-open ranges, e.g. [(0, 3)]
```

For depth or other single-channel videos, prefer `ffv1` with a 2D `(H, W)` space and a grayscale pixel format such as `gray16le`. For RGB video, use an RGB-capable codec such as `libx264rgb` (or `h264`/`libx264` with a matching YUV pixel format like `yuv420p`). Set pixel formats explicitly; there is no automatic RGB fallback.

## Trajectories And Samplers

Beyond single-transition replay:

- `StepSampler` samples fixed-size batches from stored data
- `MultiprocessingSampler` supports higher-throughput sampling workflows

Use these when the consumer of data is a learner or evaluation process rather than interactive environment code. Episode-level grouping now comes from `ReplayBuffer` segment APIs.

## Practical Advice

Use `ReplayBuffer` when you need mutation and online accumulation.

Use read-only `BatchBase` wrappers and integrations when the data already exists elsewhere.

Pick the storage backend based on the data you need to persist:

- mostly numeric tabular data: `ParquetStorage`
- existing HDF5 workflows: `HDF5Storage`
- PyTorch-heavy pipelines or memmap tensors: `PytorchTensorStorage`
- image or video artifacts: `ImageStorage` and `VideoStorage`
