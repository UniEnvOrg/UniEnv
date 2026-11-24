# Contributing an Adaptor

An adaptor connects one simulator, robot, sensor, dataset or policy to UniEnv: it exposes the source through
UniEnv's interfaces and describes its data with [spaces](../concepts/spaces-and-backends.md). Once it does,
everything else (wrappers, replay buffers, [remote execution](remote-execution.md), [MCP](mcp.md), other
people's policies) works with it unchanged.

Adaptors live in their own repositories, one per source, so each can pin its own dependencies. The ones in
[UniEnvOrg](https://github.com/UniEnvOrg) are the best templates; see the
[ecosystem list](https://unienvorg.github.io/#ecosystem).

## Pick The Interface

| You are adapting | Implement | Examples |
| --- | --- | --- |
| A simulator or robot API that already runs whole episodes | an [`Env`](../concepts/environments.md) | `unienv-maniskill`, `unienv-droid` (`DroidEnv`) |
| A physics engine, or a robot / sensor that should combine with others | a `World` and [`WorldNode`s](../concepts/world-composition.md) | `unienv-genesis`, `unienv-mjwarp`, `unienv-isaacsim`, `unienv-zed`, `unienv-rohand` |
| Recorded data | a batch (`unienv_data`), plus a `DataTransformation` into the matching env's schema | `unienv-droid` (`unienv_droid_data`), `unienv-libero` |
| A model that acts | a [`Policy`](../concepts/policies.md) | `unienv-lerobot` |

An `Env` is the quickest to write. Worlds and nodes take more work, but let users assemble their own scenes
(your robot, someone else's camera, their task) and are the way to go for engines.

## Describe The Data

The spaces are the adaptor's contract: users, buffers and policies read them instead of your code.

- **Spaces describe exactly what you return**, on your `backend` and `device`. A batched adaptor
  (`batch_size` not `None`) includes the leading batch dimension in its spaces.
- **Describe every leaf with [space metadata](../concepts/spaces-and-backends.md#space-metadata)**: `kind`,
  `units`, `frame`, `rotation`, `names`, `source` and `fps` where they apply. The builders keep the keys
  consistent:

    ```python
    import numpy as np
    from unienv_interface.backends.numpy import NumpyComputeBackend as backend
    from unienv_interface.space import BoxSpace
    from unienv_interface.space.space_utils.metadata_utils import image_metadata, proprio_metadata

    joints = BoxSpace(backend, -3.14, 3.14, np.float32, shape=(7,),
                      metadata=proprio_metadata(units="rad", names=[f"joint{i}" for i in range(1, 8)]))
    wrist = BoxSpace(backend, 0, 255, np.uint8, shape=(480, 640, 3),
                     metadata=image_metadata(format="rgb", layout="HWC", source="wrist_cam", fps=15))
    ```

- **Report only what the source measures.** If a quantity is not available, leave the key out instead of
  filling it with zeros or constants.
- **Match existing schemas where one exists.** A simulated twin of a supported robot, or a dataset recorded
  on it, should use the same keys, shapes, units and metadata as its real-robot adaptor, so policies move
  between them unchanged.
- **Keep extra outputs out of the way.** Privileged simulator state and debugging values belong in `info` or
  under their own key, not mixed into the observation a policy sees.

## Repository Layout

```
unienv-<name>/
├── unienv_<name>/          # the package; optional extras as unienv_<name>_data, unienv_<name>_collection, ...
├── tests/
├── examples/               # optional, short runnable scripts
├── README.md               # what it adapts, install, a minimal example, the observation / action spaces
├── CHANGELOG.md
├── LICENSE                 # Apache-2.0
├── NOTICE                  # Copyright (c) <year> <authors> and UniEnv Contributors
├── pyproject.toml
└── .github/workflows/      # tests, and publishing to PyPI on a GitHub release
```

```toml
[project]
name = "unienv_<name>"
version = "0.0.1a1"
license = "Apache-2.0"
requires-python = ">=3.10"
dependencies = ["unienv>=1.0.0", "<the simulator / SDK>"]

[project.urls]
Repository = "https://github.com/UniEnvOrg/unienv-<name>"
Issues = "https://github.com/UniEnvOrg/unienv-<name>/issues"

[tool.setuptools.packages.find]
include = ["unienv_<name>*"]
```

- **License:** UniEnv and its adaptors are Apache-2.0. Put the license text in `LICENSE` and the copyright
  line in `NOTICE`, listing third-party code you include there too.
- **No `CITATION.cff`:** the UniEnv citation lives only on [unienvorg.github.io](https://unienvorg.github.io/#cite),
  so authors are listed in one place.
- **Heavy or proprietary dependencies** (a simulator that needs a GPU, a robot SDK) stay out of `unienv`
  itself; that is why adaptors are separate packages.

## Test It

UniEnv's own tests check every environment against a shared contract: reset and step outputs inside the
declared spaces, batched shapes, masked resets, determinism under a seed, `sample_action`, wrapper
attribute lookup and closing. It lives in `tests_common/` of the UniEnv repository and is not part of the
installed package yet, so copy `env_contract.py` and `trees.py` from there into your `tests/`, and in
`env_contract.py` change `from tests_common.trees import ...` to `from trees import ...`:

```python
from env_contract import check_env_contract  # tests/env_contract.py and tests/trees.py, copied from UniEnv


def test_env_contract():
    check_env_contract(lambda: make_my_env(batch_size=4))
```

Also host the adaptor through the remote layer once: this catches spaces that cannot be serialized, which
remote execution and MCP need.

```python
from unienv_interface.remote import RemoteServer


def test_remote_round_trip():
    with RemoteServer() as server:
        server.register("env", make_my_env())
        with server.connect() as client:
            env = client.env("env")
            context, obs, info = env.reset(seed=0)
            assert env.observation_space.contains(obs)
            env.step(env.sample_action())
```

Tests that need a GPU, a simulator license or hardware should skip cleanly when it is missing
(`pytest.importorskip`, a marker), so the rest still runs in CI.

## Publish And Get Listed

1. **Publish to PyPI from GitHub releases.** Copy `.github/workflows/python-publish.yml` from an existing
   adaptor: it builds on a published release and uploads with PyPI trusted publishing (add the repository
   as a trusted publisher on PyPI once).
2. **Move it into UniEnvOrg (optional).** Open an issue on [UniEnv](https://github.com/UniEnvOrg/UniEnv/issues)
   to discuss moving the adaptor into the organization; a transferred repository keeps its history.
3. **Add it to the website.** The ecosystem cards are in `site/index.html` of
   [UniEnvOrg.github.io](https://github.com/UniEnvOrg/UniEnvOrg.github.io): copy a card into the matching
   group (*Simulation*, *Policies*, *Robots, sensors & data*) and open a pull request. List only public
   repositories.
