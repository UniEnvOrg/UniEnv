# Getting Started

This page is the fastest way to orient yourself in the `UniEnvPy` package.

## Install

Install the base package:

```bash
pip install unienv
```

Install optional extras when you need them:

```bash
pip install "unienv[gymnasium,video]"
```

The project also exposes optional integrations and storage backends that depend on external packages such as `torch`, `jax`, `pyarrow`, `h5py`, `datasets`, or `huggingface_hub`. Install those separately when you use the matching modules.

## The Two Main Namespaces

- `unienv_interface`: online interaction APIs such as spaces, environments, worlds, wrappers, and transformations
- `unienv_data`: offline data APIs such as batches, replay buffers, storage backends, samplers, and dataset adapters

If you are only building environments, most of your work will start in `unienv_interface`.

If you are only building dataset or replay-buffer tooling, most of your work will start in `unienv_data`.

## Mental Model

UniEnv is easiest to understand as four layers:

1. **Spaces** describe what valid data looks like.
2. **Environments and worlds** produce or consume that data over time.
3. **Wrappers and transformations** reshape the interface without changing your core implementation.
4. **Replay buffers and dataset adapters** store, sample, and serve the resulting trajectories or transitions.

## Smallest Useful Example

```python
import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space.spaces import BoxSpace, DictSpace
from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.parquet import ParquetStorage

backend = NumpyComputeBackend

transition_space = DictSpace(
    backend,
    {
        "obs": BoxSpace(backend, 0.0, 1.0, np.float32, shape=(4,)),
        "action": BoxSpace(backend, -1.0, 1.0, np.float32, shape=(2,)),
    },
)

buffer = ReplayBuffer.create(
    ParquetStorage,
    transition_space,
    cache_path="cache/demo_buffer",
    capacity=1_000,
)

print(buffer.single_space)
```

This already shows the basic pattern:

- define the structure of one sample with a `Space`
- build storage around that space
- keep higher-level code generic over the exact backend or storage implementation

## Where To Go Next

- Read [Environments](concepts/environments.md) if you are implementing runtime interaction.
- Read [Spaces and Backends](concepts/spaces-and-backends.md) if you need structured data validation or cross-backend movement.
- Read [Replay Buffers and Storage](guides/replay-buffers-and-storage.md) if your next step is logging or offline learning.

## Local Development

Inside the repository:

```bash
pip install -e .[dev,gymnasium,video]
pytest
```

## Documentation Build

This site is built from hand-written Markdown in `docs/` plus generated API pages for `unienv_interface` and `unienv_data`.

To serve it locally after installing the MkDocs toolchain:

```bash
mkdocs serve
```
