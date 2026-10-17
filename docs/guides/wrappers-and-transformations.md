# Wrappers And Transformations

UniEnv separates pure data mappings from environment-level adaptation.

## Transformations

Transformations live under `unienv_interface.transformations`.

Use them when you want to describe a mapping between spaces and their data without tying that logic to one specific environment instance.

Common examples:

- `RescaleTransformation`
- `IdentityTransformation`
- `DictTransformation`
- `FlattenDictTransformation`
- `BatchifyTransformation`
- `UnBatchifyTransformation`
- `CropTransformation`
- `ImageResizeTransformation`
- `IterativeTransformation`
- `ChainedTransformation`

This layer is useful when the same data mapping should be shared by wrappers, storage, or standalone preprocessing code.

## Wrappers

Wrappers live under `unienv_interface.wrapper` and operate on `Env` objects.

Important wrappers include:

- `ActionRescaleWrapper`
- `FlattenActionWrapper`
- `FlattenContextObservationWrapper`
- `FrameStackWrapper`
- `TimeLimitWrapper`
- `ControlFrequencyLimitWrapper`
- `ToBackendOrDeviceWrapper`
- `BatchifyWrapper`
- `UnBatchifyWrapper`
- `EpisodeRenderStackWrapper`
- `EpisodeVideoWrapper`

## Typical Wrapper Stack

```python
from unienv_interface.wrapper import (
    ActionRescaleWrapper,
    FlattenActionWrapper,
    FrameStackWrapper,
    TimeLimitWrapper,
)

env = ActionRescaleWrapper(env, new_low=-1.0, new_high=1.0)
env = FlattenActionWrapper(env)
env = FrameStackWrapper(env, obs_stack_size=3)
env = TimeLimitWrapper(env, time_limit=200)
```

The pattern is intentionally incremental: each wrapper changes one aspect of the interface while preserving the underlying environment contract.

## Backend Conversion

`ToBackendOrDeviceWrapper` moves the environment-facing data to another backend or device.

That is useful when:

- the simulator produces NumPy arrays but the policy expects PyTorch tensors
- the environment logic runs on CPU but the learner consumes data on GPU
- you want to keep one environment implementation and swap consumers later

## Video And Render Capture

`EpisodeRenderStackWrapper` and `EpisodeVideoWrapper` are built for episode-level recording.

The render stack wrapper can accumulate:

- single image streams
- batched render outputs
- nested mapping outputs, such as multi-camera render dictionaries

The video wrapper can then write those captured frames to disk, including one file per flattened render key for mapping-based outputs.

The recorded frames come from `env.render()` by default, or from the observations with `frame_source="observation"`. When the space describing the frames (`frame_space`, which defaults to the observation space for observation frames) carries [space metadata](../concepts/spaces-and-backends.md#space-metadata), only the leaves with `kind == "image"` are recorded, and each is converted to `(H, W, 3)` RGB according to its declared `layout` (`"CHW"` is moved channel-last) and `format` (`"bgr"`, `"rgba"`, `"gray"`, ...):

```python
wrapper = EpisodeVideoWrapper(env, store_dir="videos", frame_source="observation")
# one video per camera leaf, e.g. episode_0_step_0_key_cameras.wrist.webm
```

Without any `kind` metadata the wrappers keep their previous behaviour: every render frame is recorded as an RGB image, and for observation frames every `(H, W, 3)` leaf is.

## Choosing Between A Wrapper And A Transformation

Choose a transformation if:

- the logic is fundamentally a space/data mapping
- you want it to be reusable outside environment execution
- you need to serialize or compose the mapping itself

Choose a wrapper if:

- the logic changes how an environment is presented
- the mapping needs access to `reset`, `step`, or `render`
- you are adapting an existing environment for a downstream policy or evaluator
