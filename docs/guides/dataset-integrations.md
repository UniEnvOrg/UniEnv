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

This is the most direct path for bringing LeRobot-formatted robotics datasets into UniEnv-native tooling.

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
