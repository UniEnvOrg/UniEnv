# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

A test-coverage push (line coverage 64% → 92%, ~3000 tests) surfaced and fixed a large number of bugs. Behaviour changes worth noting first:

- **Batch views no longer close the batch they wrap.** `close()` on a frame/slice-stacked, sub-index, sub-item, backend/device, transformed or combined batch leaves the source open (previously garbage-collecting a FrameStack view emptied its `ReplayBuffer`). Wrapper *storages* (`DictStorage`, `TransformedStorage`, `ToBackendOrDeviceStorage`) still close their children on an explicit `close()`.
- `SyncVecEnv`/`AsyncVecEnv` `reset(seed=s)` now seeds sub-environment `i` with `s + i`.
- `TimeLimitWrapper` never marks a terminated instance as truncated (batched now matches unbatched).
- `BoxSpace`/`DynamicBoxSpace` `contains` returns `False` for non-array values instead of raising; spaces built with Python scalar dtypes (JAX defaults `float`/`int`) store the concrete dtype.
- `ReplayBuffer.get_column` returns a live view that tracks the parent buffer; structural writes on the view raise.
- `ImageCompressTransformation` budgets now include the JPEG header (new `header_bytes`; old configs keep their shapes) and raise `ValueError` when an image cannot fit.

Spaces and transformations:
- Fixed `BatchedSpace` (membership checks, sampling, `shape`, `data_to`) and `GraphSpace` (default unbounded node/edge counts, `to()`, node-feature sampling, `data_to`, single-node graphs, equality ignoring `batch_shape`); graphs of different sizes can be batched.
- Fixed flattening of `TextSpace`, `UnionSpace`, `DynamicBoxSpace` and `DictSpace` extended via `__setitem__`; flattened data now always lies in `flatten_space`.
- Integer boxes with infinite bounds sample uniformly instead of returning a constant; `UnionSpace`/`TextSpace` sampling works on PyTorch/CUDA; zero-size boxes are supported.
- Fixed `construct_space_from_data_stream`, `batch_differing_spaces` shapes, and `concatenate`/`BatchifyTransformation` for Text and Union spaces.
- gymnasium space conversion: inclusive Discrete/MultiDiscrete bounds, numpy scalar data, MultiBinary and OneOf round-trips.
- `DictIncludeKeyTransformation` keeps sibling nested keys; `DictExcludeKeyTransformation` is order-independent; `CropTransformation` supports lower-rank bounds and keeps the dtype; `IterativeTransformation.direction_inverse` works for source-dependent inverses (new `LeafwiseTransformation`).
- The remote codec rejects malformed dict entries and dtypes and applies its depth limit consistently.

Environments and wrappers:
- `AsyncVecEnv` can be constructed and reports worker errors as chained `RuntimeError`s; vec-env info dicts are stacked per key and may contain strings.
- `ActionTransformWrapper`/`ActionRescaleWrapper` actually transform actions; `FlattenContextObservationWrapper` flattens instead of unflattening.
- Masked resets work and return only the selected rows in `FrameStackWrapper`, `FuncFrameStackWrapper`, `BatchifyWrapper`, `WorldEnv` and `FuncWorldEnv`.
- `FuncFrameStackWrapper` supports action stacking; `FuncTransformWrapper` works with transformations; transform wrappers keep the context space when only other transformations are given.
- `BatchifyWrapper`/`UnBatchifyWrapper` batch the reward too; `ControlFrequencyLimitWrapper` no longer runs at twice the requested rate.
- Episode video wrappers forward reset masks, don't re-post finished episodes on close, and support `format="gif"`.
- `ToGymnasiumEnv`/`FromGymnasiumEnv` work end to end and pass gymnasium's `check_env`; `ToGymnasiumEnv` observations no longer share the episode context object between calls (required by gymnasium 1.4's `check_env`).
- World timestep compatibility uses a tolerance-based ratio, so periods such as 0.1/0.02 are accepted.
- `Wrapper.rng` works with JAX keys; `FuncEnvBasedEnv.render_fps` works without a renderer; `ToBackendOrDeviceWrapper.device` defaults to the inner env's device; `next_seed_rng` returns valid non-negative seeds; `symbol_util`, `array_serialization`, `serialize_dtype` and `dict_util` edge cases fixed.

Calibration:
- The `calibration` extra now requires `opencv-contrib-python>=4.7,<5`: OpenCV 5.0's Python bindings no longer provide `cv2.calibrateHandEye`, so `solve_hand_to_base` raises a clear `ImportError` there instead of an `AttributeError`.

Storages, replay buffer and batches:
- Fixed `ReplayBuffer.get_flattened_at*` ignoring the ring-buffer offset after wrap-around.
- `HDF5Storage`: h5py kwargs, compression/chunks for non-dict spaces, unbounded creation, `dumps` to another path, column views (incl. closing the parent file), negative/empty index arrays.
- `DictStorage` custom `nested_separator` and nested `*` keys; `TransformedStorage` flattened accessors; `ParquetStorage` boolean masks; string columns in raw HDF5/Parquet loaders; empty selections return empty batches; `create`/`load_from` honour subclasses; memory-mapped tensor views pickle correctly.
- `VideoStorage`: falls back to software decoding when a hardware decoder fails to initialise; correct auto-codec priority with per-base caching, duplicate-index reads, exact frames in approximate seek mode, PyAV for non-rgb24 buffers.
- Fixed `SliceStackedBatch`/`FrameStackedBatch` flattened access, metadata, negative indices and the JAX episode-end mask; `SubIndexedBatch` int indexing; `SubItemBatch` flattened/metadata access; `CombinedBatch.set_at` with dicts; `ToBackendOrDeviceBatch` device-only conversion; `TransformedBatch.extend` with space-dependent inverses.
- `MultiprocessingSampler` seeds each worker independently (no more duplicate batches), works with spawn/forkserver, stops workers when an epoch iterator is abandoned, and closes faster.
- `HFAsUniEnvDataset` index handling and `get_column`; `UniEnvAsPyTorchDataset.collate_fn` for batched DataLoaders; `ImageCompressTransformation` grayscale/channel-first round-trips and any `init_quality`.
- LeRobot v3: episode metadata in `meta/episodes/chunk-XXX/` is discovered, rows and video frames in packed files are located correctly, and the torch fast path handles scalar columns.

## 0.0.1b14 (2026-10-16)

- Fixed `UnBatchifyWrapper` construction: `sbu.unbatch_spaces` takes only the space, so the extra batch-size argument made `UnBatchifyWrapper(env)` raise `TypeError`.
- Hardened `unienv_utils.calibration` after review: `solve_camera_relative` now rejects outlier paired observations with tianji's MAD-based gating (new `reject_outliers` / `min_inliers` parameters; `PairResult` reports `n_inliers` and `inlier_mask`, and residuals are computed over the inliers); the hand-eye low-diversity warning no longer misses rotations about the same axis in opposite directions; calibration JSON writes unavailable residual values as `null` and refuses non-finite numbers (`allow_nan=False`).
- Added `root_node` support to `CombinedWorldNode`, `FlatCombinedWorldNode`, `CombinedFuncWorldNode` and `FlatCombinedFuncWorldNode`. A `root_node` is passed separately (same world, not listed in `nodes`) and its data merges at the top level of the aggregated context/observation/action dicts instead of nesting under its own name:
  - Sole-root contributor per channel follows the existing `direct_return` rule (`True`: the root's own space/data passes through unwrapped, any space type; `False`: wrapped under the reserved key `ROOT_NODE_KEY = ""`). Mixed root + named children requires a `DictSpace` root space whose keys merge next to the child entries; a root key equal to a child node name raises `ValueError`.
  - Actions: the whole action is forwarded to a sole-root action provider, while in the mixed case the incoming mapping's keys are routed to children by name and every remaining key accumulates into the root's cached action. Each consumer is dispatched at its own control-rate/update-rate ratio tick with the usual hold-last semantics.
  - The root rides all child machinery (priority union, lifecycle dispatch, reward summation, termination/truncation OR-ing, info, render, close, `get_nodes_by_fn`) while staying out of the public `nodes` list and out of `get_node`. In the functional variants the root's node state is persisted under `ROOT_NODE_KEY` (`""`), keeping `CombinedNodeStateT` a plain `Dict[str, Any]`.
  - The flat variants treat the root as one more flat contributor for context/observation (keys merge at the top level) and route action keys that no child's action space claims to the root; without a root node such unclaimed keys raise `ValueError`.
- Added `unienv_utils`, a top-level package for general utilities, and `unienv_utils.calibration`, a unified ChArUco camera calibration toolkit (requires the new optional `calibration` extra: `pip install unienv[calibration]`, i.e. `opencv-contrib-python`). Importing `unienv_utils` itself never imports OpenCV; `unienv_utils.calibration` raises a clear `ImportError` when OpenCV with `aruco` is missing. The subpackage is a generalized lift of the battle-tested tianji calibration maths and supports both the legacy (`detectMarkers` / `interpolateCornersCharuco` / `estimatePoseCharucoBoard`) and the new (`ArucoDetector` / `CharucoDetector` / `matchImagePoints`) OpenCV APIs:
  - `board`: `CharucoBoardConfig` (validated frozen dataclass, defaults = tianji's 9x14 board with 20 mm checkers / 16 mm markers / `DICT_5X5_100`, plus `tianji_default()`), `resolve_aruco_dict`, `make_board` and `render_board` for printing the board to scale.
  - `detect`: `detect_charuco` with the tuned tianji detector parameters and `estimate_board_pose` (`T_cam_board` via PnP).
  - `hand_eye`: `solve_hand_to_base` for the eye-to-hand case (fixed camera, board on the gripper) with the `cv2.calibrateHandEye` input swap that returns `T_base_cam` directly, plus `T_gripper_board` recovery, translation/rotation consistency residual stats, minimum-pose validation and a warning for degenerate (low orientation diversity) capture sets.
  - `camera_pair`: `solve_camera_relative` for the relative pose of two rigidly mounted cameras from paired board observations, with quaternion-mean averaging and residuals, plus `transfer_to_base`.
  - `intrinsics`: `calibrate_intrinsics_charuco` wrapping `cv2.aruco.calibrateCameraCharuco` (falling back to `cv2.calibrateCamera` on the points matched by `CharucoBoard.matchImagePoints` on newer OpenCV) with per-view reprojection errors.
  - `format`: canonical `unienv_calibration/v1` JSON (`save_calibration`, `load_calibration`, `validate_calibration_dict`) covering `T_base_cam`, intrinsics (K, distortion, resolution, source), solver, residuals, board and creation timestamp.
  - `transforms`: rigid-transform helpers (`invert_T`, `compose`, `pose6d_to_T` / `T_to_pose6d` using the extrinsic XYZ (`sxyz`) Euler convention, quaternion averaging, `validate_T`).
  - CLI: `python -m unienv_utils.calibration print-board --out board.png` renders the printable ChArUco PNG for a given resolution and board geometry.
  - Tested by the new `tests_utils/` suite, entirely on synthetic data (no cameras, no GPU).

## 0.0.1b12 & 0.0.1b13 (2026-04-22)

- Bugfixed random read errors in `VideoStorage` when using the `pyav` backend by implementing a more robust seeking mechanism that handles edge cases such as seeking to non-keyframes and reaching the end of the video stream. 

## 0.0.1b11 (2026-03-05)

- Added support for rendering videos with `EpisodeRenderStackWrapper`, `EpisodeVideoWrapper`, `EpisodeWandbVideoWrapper` when the underlying environment provides potentially nested `Mapping[str, np.ndarray]` observations that contains image. The wrapper will automatically detect the image data in the observation and use it for rendering one video for each image observation key.
- Added `FlattenDictTransformation` and `UnflattenDictTransformation` for flattening and unflattening nested dictionary data.

## 0.0.1b10 (2026-02-18)

- Bumped required `xbarray` package version to `0.0.1a16` to include the latest bugfixes and performance improvements.
- Major improvements to the closing management of existing data storage backends through de-referencing.
- Major improvements to the `World` and `FuncWorld` system to support better lifecycle management.

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