# UniEnv

![UniEnv overview](static/unienv_overview.webp)

UniEnv is a Python framework for building robot environments and robot data pipelines on top of a shared set of abstractions.

Documentation: <https://unienvorg.github.io/UniEnv/>

It gives you:

- a common environment interface for simulation and real-robot control
- a functional environment variant for explicit state passing
- backend-aware spaces, wrappers, and transformations
- world and node composition utilities for multi-component environments
- replay buffers, storages, samplers, and dataset adapters for offline data workflows

The project is designed around one idea: environment code and dataset code should not have to be rewritten every time the simulator, robot, or tensor library changes.

## What Is In The Package

- `unienv_interface`: environments, worlds, nodes, spaces, wrappers, and transformations
- `unienv_data`: batches, replay buffers, storage backends, samplers, and dataset integrations
- backend portability through [XBArray](https://github.com/UniEnvOrg/XBArray) and DLPack-based conversion paths

## Installation

Install the base package:

```bash
pip install unienv
```

Install optional extras when needed:

```bash
pip install "unienv[gymnasium,video]"
```

Some integrations and storage backends rely on their own ecosystem packages such as `pyarrow`, `h5py`, `datasets`, `huggingface_hub`, `torch`, or `jax`. Install the ones that match the features you plan to use.

## Quick Example

The example below shows the core building blocks without depending on a specific simulator:

```python
import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space.spaces import BoxSpace, DictSpace
from unienv_interface.transformations import RescaleTransformation
from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.parquet import ParquetStorage

backend = NumpyComputeBackend

transition_space = DictSpace(
    backend,
    {
        "obs": BoxSpace(backend, 0.0, 1.0, np.float32, shape=(4,)),
        "action": BoxSpace(backend, -1.0, 1.0, np.float32, shape=(2,)),
        "reward": BoxSpace(backend, -np.inf, np.inf, np.float32, shape=()),
    },
)

action_transform = RescaleTransformation(new_low=0.0, new_high=1.0)
normalized_action_space = action_transform.get_target_space_from_source(
    transition_space["action"]
)

buffer = ReplayBuffer.create(
    ParquetStorage,
    transition_space,
    cache_path="cache/demo_buffer",
    capacity=10_000,
)

print(normalized_action_space)
print(buffer.single_space)
```

In practice, you would pair these components with your own `Env`, `FuncEnv`, `World`, or `WorldNode` implementations, then add wrappers and storage backends as needed.

## Documentation

Documentation: <https://unienvorg.github.io/UniEnv/>

Start with:

- `docs/getting-started.md` for installation and the package map
- `docs/concepts/` for the core abstractions
- `docs/guides/` for wrappers, replay buffers, and dataset integrations

## Development

For local development:

```bash
git clone https://github.com/UniEnvOrg/UniEnv
cd UniEnv/UniEnvPy
pip install -e .[dev,gymnasium,video]
pytest
```

If you want to exercise optional backends or integrations during development, install their matching dependencies explicitly before running the relevant tests.

## Cite

If you use UniEnv in research, cite:

```bibtex
@software{cao_unienv,
  author = {Yunhao Cao AND Cory Fan AND Meryl Zhang AND Sabrina Liu AND Kuan Fang},
  title = {{UniEnv: Unifying Robot Environments and Data APIs}},
  year = {2025},
  month = oct,
  url = {https://github.com/UniEnvOrg/UniEnv},
  license = {MIT}
}
```

## Acknowledgements

UniEnv is influenced by [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and OpenAI Gym, and builds on ideas from the [Array API Standard](https://data-apis.org/array-api/latest/) and [DLPack](https://github.com/dmlc/dlpack).
