# Dataset Integrations

UniEnv can wrap external dataset ecosystems behind the same `BatchBase` interface used by replay buffers and storages.

That means downstream code can often stay unchanged whether its data comes from:

- a live replay buffer
- a local dataset on disk
- a Hugging Face dataset
- a PyTorch `Dataset`

## Hugging Face Datasets

`HFAsUniEnvDataset` adapts a `datasets.Dataset` into a UniEnv batch.

Highlights:

- infers a UniEnv space from the first sample
- configures the Hugging Face dataset format to match the requested backend
- supports column selection through the batch column-view helpers

Use it when your training or evaluation data is already managed through the Hugging Face datasets stack.

## LeRobot Datasets

`LeRobotAsUniEnvDataset` reads local or Hub-hosted LeRobot datasets without requiring the upstream `lerobot` Python package.

Highlights:

- supports multiple LeRobot dataset versions
- reconstructs a `DictSpace` from dataset features
- can include or exclude named features
- can optionally decode video features
- describes the features through [space metadata](../concepts/spaces-and-backends.md#space-metadata): image and video features carry `{"kind": "image", "format": "rgb", "layout": "HWC", "fps": <dataset fps>, "source": <feature key>}`, `observation.state` and `action` carry `kind` `"proprio"` / `"action"`
- every frame's metadata holds its episode, laid out like [replay buffer episode records](replay-buffers-and-storage.md#episode-records): `episode_id` (LeRobot's `episode_index`) and `episode` = `{"task", "task_index", "length"}`, so `sample_with_metadata()` returns each frame's task text; `get_episode_ids`, `get_episode_data` and `get_episode_segments` read them without loading frames

```python
from unienv_data.samplers import StepSampler

sampler = StepSampler(dataset, batch_size=32)
batch, metadata = sampler.sample_with_metadata()
metadata["episode"]["task"]      # object array of task strings, one per frame
metadata["episode_id"]           # LeRobot episode_index, on the dataset's backend
```

An episode that lists several tasks records its first one; `task_index` is -1 for a task missing from `meta/tasks`.

This is the most direct path for bringing LeRobot-formatted robotics datasets into UniEnv-native tooling.

With `video_backend="auto"` (the default) videos decode with TorchCodec when it is installed. If TorchCodec cannot open a video that PyAV can, which happens when its build does not match the installed torch or FFmpeg, the reader warns once and uses PyAV from then on.

### Writing LeRobot Datasets

UniEnv also writes LeRobot datasets (format v3.0) without the `lerobot` package. The output has the same layout LeRobot itself writes: chunked data parquet files, one MP4 per camera holding consecutive episodes, and the `meta/` files (`info.json`, `tasks.parquet`, the episodes index with per-episode statistics, `stats.json`). `LeRobotDataset` and `LeRobotAsUniEnvDataset` both load it, and `push_to_hub` uploads it.

Write frame by frame, for example while collecting:

```python
from unienv_data.integrations.lerobot import LeRobotDatasetWriter

with LeRobotDatasetWriter.create("out/pick_place", env.observation_space, fps=10) as writer:
    for episode in episodes:
        for observation in episode:
            writer.add_frame(observation, task="pick up the cube")
        writer.save_episode()          # or writer.clear_episode() to drop it
writer.push_to_hub("me/pick_place")    # after the with block has finalized the dataset
```

Or write a whole replay buffer or dataset at once:

```python
from unienv_data.integrations.lerobot import export_lerobot

writer = export_lerobot(replay_buffer, "out/pick_place", fps=10)
writer.push_to_hub("me/pick_place", private=True)
```

`export_lerobot` takes its episodes from the source: a replay buffer's [episode records](replay-buffers-and-storage.md#episode-records) (or its segments), or a LeRobot dataset's episodes. You can also pass `episodes=[(start, end), ...]`. Each episode's task is the `task` argument, which is either a string or a function of the episode record. Without it, the task is read from the record's `task`, `language_instruction` or `instruction` field.

Features:

- **Names.** By default every leaf of the space is a feature, named by its keys joined with dots (`{"obs": {"state": ...}}` becomes `obs.state`). To pick and rename features, pass `features={"observation.state": "obs/state", "observation.images.front": "obs/front", "action": "action"}`; only the mapped leaves are written. `lerobot_features(space, features=...)` shows the result without writing anything.
- **Images.** Leaves with `kind="image"` [metadata](../concepts/spaces-and-backends.md#space-metadata) are images, as are unannotated `uint8` `H×W×3` leaves. CHW layout and BGR format are converted. Images are written as videos (AV1 like LeRobot, or H.264 when no AV1 encoder is available; needs PyAV), which require an even height and width. With `videos=False` they are written as PNGs inside the parquet files.
- **Other leaves.** Box leaves become `float32`/`int64`/... features, with their `names` metadata, and binary leaves become `bool`. Text and other leaves are skipped, and `writer.skipped` records why. The standard columns (`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`) are always filled in by the writer, so a source that already has them, such as a LeRobot dataset, does not copy them.

## PyTorch Dataset Adapters

UniEnv provides both directions:

- `PyTorchAsUniEnvDataset`: wrap an existing PyTorch `Dataset` as a UniEnv batch
- `UniEnvAsPyTorchDataset`: expose a UniEnv batch as a PyTorch `Dataset`

This is useful when the training stack expects PyTorch dataloaders but the storage or replay layer is written against UniEnv.

## Choosing An Adapter

Use a UniEnv adapter when you want:

- one indexing interface across multiple dataset sources
- space-aware structured samples instead of ad hoc dictionaries
- the ability to plug the result into UniEnv samplers or downstream wrappers

Use the original upstream dataset API directly when you need ecosystem-specific features that are outside the UniEnv batch contract.
