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

## Trajectories And Samplers

Beyond single-transition replay:

- `TrajectoryReplayBuffer` organizes episode-level access
- `StepSampler` samples fixed-size batches from stored data
- `MultiprocessingSampler` supports higher-throughput sampling workflows

Use these when the consumer of data is a learner or evaluation process rather than interactive environment code.

## Practical Advice

Use `ReplayBuffer` when you need mutation and online accumulation.

Use read-only `BatchBase` wrappers and integrations when the data already exists elsewhere.

Pick the storage backend based on the data you need to persist:

- mostly numeric tabular data: `ParquetStorage`
- existing HDF5 workflows: `HDF5Storage`
- PyTorch-heavy pipelines or memmap tensors: `PytorchTensorStorage`
- image or video artifacts: `ImageStorage` and `VideoStorage`
