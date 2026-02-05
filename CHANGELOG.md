# Changelog

All notable changes to this project will be documented in this file.

## 0.0.1b9 (2026-02-05)

- Bumped required `xbarray` package version to `0.0.1a14` and updated relative imports to make `PyTorchComputeBackend` capitalization consistent between UniEnv and XBArray.

## 0.0.1b8 (2026-02-02)
- Optimized `VideoStorage` to support using different backends (`torchcodec`, `pyav`) for reading videos based on user preference and system capabilities.
- (Breaking) Renamed `UniEnvPyTorchDataset` to `UnienvAsPyTOrchDataset` for better consistency with other dataset naming conventions.
- Added `PyTorchAsUniEnvDataset`
- Adedd huggingface intergration with `HFAsUniEnvDataset` for loading standard Hugging Face Hub datasets in the `dataset` package format (`pyarrow` datatables).

## 0.0.1b7 (2026-01-22)

- Bugfix `BatchifyTransformation`.

## 0.0.1b6 (2026-01-21)

- **Deprecated, use version 0.0.1b7 instead.**
- There's now an optional `nan_to` parameter for the `RescaleTransformation` class that allows users to specify a value to replace NaN values in the input data during rescaling. If not provided, NaN values will remain unchanged.
- Flipped `is_memmap` parameter default value to `True` in `PyTorchTensorStorage` to improve usability when storing large datasets.
- `PyTorchTensorStorage` now supports saving and loading data in the `BinarySpace` format.
- Optimized the behavior of reading `TextSpace` data from `HDF5Storage` to produce a numpy array of strings.
- Introduced a `ToBackendOrDeviceStorage` for converting data to a specified compute backend or device when reading from the storage.
- Introduced `ImageStorage`, `VideoStorage`, `NPZStorage` for storing image, video, and numpy `.npz` files respectively. Notably, the `VideoStorage` supports hardware-accelerated video encoding and decoding using `imageio` and `av` libraries.
- Minor bugfixes, documentation improvements, and code optimizations.

## 0.0.1b5 (2025-12-09)

- Added a `is_multiprocessing_safe` attribute to `ReplayBuffer` and `StorageBase` class that tells users whether the workload of creating the replay buffer before forking the python process (commonly used for pytorch dataloading) can be used.
- Add backward compatibility for the `FlattenedStorage` (the file it was implemented is moved to a different location, now we've added a compatibility mapping to map to the correct implementation file when loading from a legacy stored replay buffer)
- Bugfixed `flatten_data` implementations for `BinarySpace`

## 0.0.1b4 (2025-11-21)

Add DictStorage for mapping dictionary replay buffers with different keys.

## 0.0.1b3 (2025-11-13)

Removed integration with mujoco-playground and maniskill and moved them over to the `UniEnvAdaptors` package.

## 0.0.1b1 and 0.0.1b2 (2025-09-04)

Released first beta version of UniEnvPy on PyPI with unfinished world system support.