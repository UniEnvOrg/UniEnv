# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added a LeRobot dataset writer (format v3.0, no `lerobot` package needed), in `unienv_data.integrations.lerobot`. `LeRobotDatasetWriter.create(root, space, fps, ...)` writes frame by frame with `add_frame` / `add_frames`, `save_episode` / `clear_episode` and `finalize`. `export_lerobot(source, root, fps)` writes a whole replay buffer (episodes and tasks come from its episode records) or LeRobot dataset. `push_to_hub` uploads the result and tags it `v3.0`. Features are named by their joined keys (`obs.state`) unless a `features` mapping is given. Images are written as AV1/H.264 videos, or as PNGs with `videos=False`. LeRobot's `LeRobotDataset` loads the output, statistics included.
- `LeRobotAsUniEnvDataset` with `video_backend="auto"` falls back to PyAV, with a warning, when TorchCodec is installed but cannot load (a build that does not match torch or FFmpeg). Before, every video read failed.
- Added `unienv_interface.utils.png.encode_png`, a dependency-free PNG encoder. It was moved out of `unienv_mcp`.

## 1.0.0 (2026-11-23)

First stable release. Requires `xbarray>=1.0.0`.

- The wheel no longer installs the repository's `docs` and `benchmarks` folders as top-level packages (earlier wheels did).

- Added `unienv_mcp` (extra `unienv[mcp]`, command `unienv-mcp`): environments and policies as MCP servers for LLM agents. `UniEnvMCPServer.add_env` exposes an environment as tools `E.describe` / `open` / `reset` / `step` / `render` / `close`, with state in instances addressed by id (a factory builds one per `open`, closed after `idle_timeout`; an object is one shared instance), each on its own thread. `E.observe` shows an instance's state again without acting (e.g. after the model's context was compacted), and `E.describe` lists the open instances and the camera names. `add_policy` exposes `P.run`, a closed-loop run on an env instance from its current state (policies as skills; `task` replaces the context for the policy; progress, cancellation, keyframes, or `wait=false` with `runs.status` / `runs.cancel`), and `P.suggest`. Action tool schemas come from the action spaces, out-of-bounds actions are tool errors listing the violations (or clipped), and image leaves of observations are returned as images: tools take `images` (`"all"`, `"none"` or camera paths), and `ContentOptions` sets the default cameras, a per-result image limit and first-env-only images for batched environments. `physical=True` annotates acting tools as destructive for real systems. `E.host` serves an open instance with UniEnv remote execution for other processes (any `tcp://`, `unix://` or `ws://` address; on by default, `hosting=False` removes it), with every reset and step, the model's and policy runs' included, published to remote subscribers in order; `P.host` serves the policy paired with an instance; `E.unhost` stops. `unienv-mcp serve module:attr` serves from code, `unienv-mcp bridge <address>` serves what a `RemoteServer` hosts; stdio or Streamable HTTP. No nodes and no access control.
- Added `RemoteServer.publish(name, method, value)`: publish a `reset` / `reload` / `step` result the server did not produce itself (e.g. an env driven on the server's side) to the env's subscribers; `register(..., publish_results=False)` for objects that publish all their own events.
- Added `space_to_json_schema`, `to_json`, `from_json` and `bound_violations` (`unienv_interface.space.space_utils`): JSON Schemas of spaces and their data as plain JSON (vectors with `names` metadata as objects keyed by name), for LLM tool calling.
- Added `RemoteServer.register_factory(name, factory, release_after=60)`: the object `factory()` builds is hosted only while clients use it. It is built on its own worker when a client first opens it (getting its proxy, calling or subscribing to it; spaces and attributes are read then) and closed there `release_after` seconds after the last client that opened it disconnects or releases it (`RemoteClient.release(name)` or `close()` on the proxy; `0`: at once, `None`: never). Reopening within the grace period gets the same object, later a newly built one. Additive to protocol version 2: clients of earlier versions don't see factory-hosted objects, and nothing else changes.
- `LeRobotAsUniEnvDataset` frames carry their episode in their metadata, laid out like `ReplayBuffer.with_episode_data()`: `episode_id` (LeRobot's `episode_index`) and `episode` = `{"task", "task_index", "length"}`; `get_episode_ids`, `get_episode_data` and `get_episode_segments` read them without loading frames. `get_episode(episode_index)` now looks the episode up by LeRobot's `episode_index` (the same as before unless the indices have gaps).
- **Fixed:** LeRobot episode tasks: v3.0 datasets read no tasks at all (`meta/tasks.parquet` was ignored, so every episode had task `""`), and v2.x datasets whose `episodes.jsonl` lists tasks by text (as LeRobot writes them) gave every episode task 0. Checked against the Hub metadata of `lerobot/libero_10_image` (379 episodes, 10 tasks; v2.1 and v3.0 agree).
- Added per-episode records to `ReplayBuffer`: `ReplayBuffer.create(..., episode_space=..., episode_storage_cls=...)` stores one record of `episode_space` per segment (task, instruction, success, ...) in a second storage, passed when the segment closes (`mark_segment_end(episode_data=...)`, `extend(value, episode_data=...)`, `commit_segment(writer, episode_data=...)`; required for every segment). `get_episode_ids`, `get_episode_data` and `get_episode_segments` read them; records stay available while any step of their episode is in the buffer, through wrap-around; `dumps()` / `load_from()` keep them. Steps carry their episode only on request, through the read-only view `with_episode_data()` (`ReplayBufferEpisodeView`; metadata `episode_id` and `episode`), e.g. for samplers.
- A segment built with `ReplayBuffer.append` can take its episode record with its first step instead (`append(value, episode_data=...)`, readable while the segment is open; the segment then closes with a bare `mark_segment_end()`).
- `ReplayBufferCollectionWrapper` writes episode records into buffers with an `episode_space`: by default from the episode's context, return, length, `terminated` / `truncated`, `info["success"]` and env slot (keys of a `DictSpace`), or the context alone for any other space; `episode_builder(EpisodeSummary)` for custom records; `episode_data_at="auto" | "start" | "end"` picks whether a record is written with the episode's first step (no context held while it runs; the default whenever the record needs only `context` / `env_index`) or when it ends. Episodes cut by a reset or a finalizing buffer swap are recorded as `truncated`.
- **Changed:** `ReplayBufferCollectionWrapper` in shared mode (unbatched or `batch_size=1`) now closes the open segment on `reset()` / `reset_wait()`; before, an episode cut by a reset ran on into the next one's segment.
- Added remote policies: `RemoteServer.register` hosts a `Policy` (kind `"policy"`) and `RemoteClient.policy(name)` returns a `RemotePolicy`, itself a `Policy`, whose `reset`, `observe_transition`, `get_action` and `visualize_action` run on the server: contexts and observations are rebuilt on the hosted policy's backend / device, actions on the client's. A `FuncPolicy` is hosted as `FuncPolicyBasedPolicy(func_policy)` (registering a bare one is a `TypeError`).
- Added the policy interfaces in `unienv_interface.policy` (also exported from `unienv_interface`), merged from UniAct (`c5691a5`; the UniAct repository has since been deleted): the stateful `Policy` / `PolicyWrapper` and functional `FuncPolicy` / `FuncPolicyWrapper` interfaces, and `FuncPolicyBasedPolicy` to run a `FuncPolicy` as a `Policy`. Changed from UniAct: a policy's state changes only in `reset` and `observe_transition`, which also takes `dt` (the step's duration in seconds of env time, required), and `compute_action(observation)` became `get_action()` (`FuncPolicy.get_action(state) -> (state, action)`, the returned state for bookkeeping such as RNG keys); `reset`, `observe_transition` and `get_action` are abstract; `FuncPolicy` has a `batch_size` and batched spaces like `FuncEnv` (`initial()` no longer takes `batch_size`, `FuncPolicyBasedPolicy` uses the functional policy's); a masked `reset` receives the full-batch context and observation; a name typo that kept `funcpolicy` from importing is fixed.
- Added `ParquetStorage.create(..., array_encoding=...)`: `"list"` (default, unchanged: fixed-size lists that Hugging Face `datasets`, pandas, DuckDB, ... read as arrays) or `"binary"` (one fixed-size binary value per row holding the array's little-endian bytes), for all array columns or per column (`{"image": "binary", "*": "list"}`). Binary columns write and read ~10x faster for images (128x128x3, 10k steps: `dump+close` 9.5 s -> 0.46 s, unbounded appends 1.1 ms -> 0.14 ms per step) and compress better, but other tools see opaque bytes and earlier UniEnv versions cannot read them. The encodings are recorded in `_metadata.json` (only when a column is binary); `load_from` / `load_replay_buffer_from_raw_parquet` read both. The storage benchmarks have a `parquet-binary` case.
- **Fixed:** a masked reset through `ReplayBufferCollectionWrapper` (`reset(mask=...)` / `reset_wait()`) wrote the reset rows into the observation and context arrays the previous `step()` / `reset()` had returned (in place on NumPy / PyTorch), so a caller still holding e.g. the terminal observation of the reset envs saw it overwritten. The merge now goes into a copy.
- `ParquetStorage` no longer decodes the zero-filled pieces `create()` just wrote when it first writes into them (~0.4 s per 1024 128x128x3 images): fixed-capacity appends ~6x, `extend` 15-20x faster.
- `HDF5Storage`: row writes read the dataset shape once from the file dataspace and reuse memory dataspaces (appends ~1.4x faster); unbounded storages resize their cached leaf datasets instead of walking the HDF5 groups on every append (~2.6x).
- Faster space utilities: `flat_dim` / `flatten_data` / `unflatten_data` of a `DictSpace` cache its flattening layout (sorted keys and offsets) per `start_dim`, invalidated by in-place changes to the space or a nested one (`flat_dim` ~6x, flatten / unflatten 1.1-1.5x); `set_at` assigns into torch tensors in place without `array_api_extra.at` (~2x per leaf), as it already did for NumPy; `BoxSpace.contains` reduces with the array's own `min` / `max` (1.2-1.6x on small arrays).
- **Fixed:** `PyAvVideoReader` random access into B-frame video (e.g. x265 in MP4) could return a later frame: the demuxer picks the seek keyframe by decode time, which precedes its presentation time by the reorder delay, so a target in the first frames after a keyframe landed on the next keyframe (x265 keyint 15: ~24% of random single-frame reads off by 1-3 frames; keyint 5: ~80%). A seek that lands past the target now seeks again further back.
- Added remote data transformations: `RemoteServer.register` hosts a `DataTransformation` (kind `"transformation"`), with the `backend` / `device` its calls run on (default numpy), and `RemoteClient.transformation(name)` returns a `RemoteDataTransformation` whose `transform` / `get_target_space_from_source` run on the server: the source space and data are rebuilt on the hosted backend, the result on the client's. For transformations whose dependencies live in another environment (e.g. a grasp synthesizer pinned to an older CUDA toolchain). The proxy has no inverse and cannot be serialized.

### Fixed (pre-1.0 review)

**Spaces, backends and serialization**

- `BoxSpace` / `DynamicBoxSpace` equality compared integer bounds with a relative tolerance, so e.g. `[0, 100000]` and `[0, 100001]` boxes were "equal" (and vectorized envs with such per-env spaces got the first env's bounds). Integer bounds now compare exactly; floating bounds still tolerate rounding.
- `json_to_space(..., JaxComputeBackend, ...)` raised `Unknown dtype string: float64` (or `int64`) for spaces saved from numpy when JAX runs without x64 (its default); such dtypes now load as the 32-bit dtype JAX stores them as.
- moving an int64 `BoxSpace` / `DynamicBoxSpace` to JAX without x64 (`to()` or `json_to_space`) wrapped its bounds around (`[0, 2**32 + 5]` became `[0, 5]`, full-range boxes crashed with `low is greater than high`); bounds are now clipped to the int32 range.
- flattening a `DictSpace` / `TupleSpace` with a negative `start_dim` below -1 concatenated along the wrong axis (silently wrong shapes such as `(4, 12)` instead of `(2, 24)`) or crashed, and unflattening failed.
- flattening and unflattening a `DynamicBoxSpace` dropped trailing values equal to `fill_value` even below `shape_low` (`[3, 0]` came back as `[3]`, which is not in the space).
- `flatten_data` / `flatten_space` crashed with "cannot reshape array of size 0" on an empty batch (e.g. extending or indexing a storage / replay buffer with zero items).
- flattening a `TextSpace` (alone or inside a `DictSpace` / `TupleSpace`) on CUDA produced CPU tensors, so flattening a dict with a text entry crashed with "Expected all tensors to be on the same device".
- `batch_utils.set_at` on a batched `DynamicBoxSpace` crashed with an integer index when the value was shorter than the batch slot, and a length-1 value silently broadcast over the whole slot; shorter values are now padded with `fill_value`.
- batches of `DynamicBoxSpace` data built by `concatenate` (padded with `fill_value`) were rejected by the batched space's `contains` when `fill_value` lies outside `[low, high]` (e.g. the default `0` with `low=1`); `fill_value` padding beyond `shape_low` is now accepted by `contains` and kept by `clip`.
- `space_to_json` and `serialize_scalar_or_array_value` raised `Unknown dtype` for float16 and bfloat16 data (e.g. `ReplayBuffer.dumps` with a float16 `BoxSpace`); half-precision dtypes now serialize and load back.
- `BoxSpace.contains` / `DynamicBoxSpace.contains` raised a device-mismatch `RuntimeError` for data on another device (depending on the bounds layout) instead of returning a bool.
- `GraphSpace.sample` and `GraphSpace.contains` crashed with "zero-size array to reduction operation" for an empty batch.
- `TextSpace` with an empty charset sampled alphanumeric strings it does not contain; it now samples `""` (and raises if `min_length > 0`).
- `BoxSpace` rejected bounds with fewer dimensions than the box (e.g. `low` of shape `(3,)` for a `(2, 3)` box) although they broadcast to it.
- `dict_util.flatten_keys_in_mapping` dropped empty nested dicts / `DictSpace`s, so `unflatten_keys_in_mapping` lost those keys.

**World composition and transformations**

- `RescaleTransformation.direction_inverse` reshaped the source box's stored bounds instead of broadcasting them, so a box whose low and high are stored at different sizes (e.g. per-row low, scalar high, as xbarray 1.0 abbreviates them) could not be inverted (`cannot reshape array of size 1`).
- `CombinedWorldNode` / `CombinedFuncWorldNode` accepted control timesteps that are not integer multiples of the smallest one (e.g. 0.02 / 0.03 / 0.06) and then delivered actions to the 0.03 s node every 0.04 s. Such combinations now raise at construction.
- `WorldEnv` / `FuncWorldEnv` accepted a control timestep that is not an integer multiple of the update timestep: control 0.03 / update 0.02 silently simulated 0.04 s per step, and control 0.02 / update 0.04 ran zero world steps. This now raises at construction; float multiples such as 0.05 / 0.01 are still accepted.
- Inside `CombinedWorldNode` / `CombinedFuncWorldNode`, child nodes without their own `control_timestep` / `update_timestep` received `dt=None` in `pre_environment_step` / `post_environment_step`. They now receive the step's dt, the same value a bare node gets from `WorldEnv`.
- `FuncWorldEnv.initial()` / `reload()` lost the state of nested `CombinedFuncWorldNode` children that initialize at different priorities, which raised `KeyError` on the first observation. Nested states are now merged recursively.
- `ImageResizeTransformation` crashed on PyTorch for inputs with two or more leading batch dimensions and for `uint8` images on CUDA, and returned `float32` (outside the target space) for integer images on JAX. Integer images now keep their dtype on every backend.
- `ImageResizeTransformation` gave different results on NumPy (plain bilinear without antialiasing, via OpenCV) than on PyTorch / JAX. All backends now use antialiased bilinear resizing (PyTorch `antialias=True` semantics) and agree to float precision. NumPy no longer needs OpenCV for this.
- `FlatCombinedFuncWorldNode` raised a spurious `ValueError` when a root-node key equalled a child node *name* (e.g. child `arm` with key `arm_pos` and root key `arm`), even though the flat keys do not collide.
- `RescaleTransformation` silently dropped `new_dtype` when serialized without a source space (e.g. scalar bounds), so the restored transformation output the source dtype.
- `WorldEnv.reset(...)` / `FuncWorldEnv` with several nodes raised `TypeError` for any extra reset keyword argument (e.g. `options=`), while a single node accepted it. `CombinedFuncWorldNode.reset` silently dropped such arguments. Extra kwargs are now forwarded to every child, and `pernode_kwargs` overrides them per node.
- `DictTransformation` with nested mapping keys (e.g. `"a"` and `"a/x"`) transformed data against the original sub-space instead of the already-transformed one, producing values outside its own target space. Its inverse now also undoes the entries in reverse order.
- `ChainedTransformation.direction_inverse()` without a source space raised (e.g. for a chain of `FlattenDictTransformation` or `BatchifyTransformation`), although each member can be inverted without one.
- Combining several `EnvAsWorldNode`s built without `world=` crashed on the first `WorldEnv.step()` with `AttributeError: 'NoneType' object has no attribute 'batch_size'`.
- A `root_node` whose name equals a child node name silently overwrote that child's `info` (and collided in render output and `pernode_kwargs`). This now raises `ValueError` at construction.
- `FlattenDictTransformation` dropped empty nested dicts, so flattening followed by its inverse did not restore the original space or data.

**Environments, wrappers and policies**

- `TransformWrapper`, `ContextObservationTransformWrapper` and `FuncTransformWrapper` transformed the rows of a masked `reset(mask=...)` with the full-batch space, so space-dependent transformations such as `RescaleTransformation` on spaces with per-env bounds (e.g. a `SyncVecEnv` whose sub-env bounds differ) returned a full-size batch of wrong values or raised a broadcast error. The rows are now transformed with the matching masked sub-space.
- reading `FuncEnvBasedEnv.render_fps` before the first `reset()` called `render_init` with `state=None`, which raised for renderers that need the state (e.g. through `FuncFrameStackWrapper`: `'NoneType' object has no attribute 'env_state'`). This also broke `ControlFrequencyLimitWrapper(env)` without `control_dt` and `AsyncVecEnv` construction. Before the first reset the declared fps is reported, and `FuncFrameStackWrapper.render_init` accepts `state=None`.
- `EpisodeRenderStackWrapper` / `EpisodeVideoWrapper` / `EpisodeWandbVideoWrapper` with `frame_source="observation"` and `rendered_index` indexed the compacted rows of a masked reset with the full-batch index, which raised (`Failed to index render tensor`) or recorded another env's frame. A masked reset that did not reset the recorded env also appended a duplicate frame. The recorded env's row is now located among the reset rows, and no frame is captured when the recorded env was not reset.
- `SyncVecEnv` / `AsyncVecEnv` built the dtype of stacked `info` scalars from the first sub-env only: `[0, 0.75]` became `[0, 0]`, NumPy booleans and integers became floats, and Python integers beyond int32 raised on PyTorch. The dtype now fits all values (bool, float, or int64 when needed), and lists that mix numbers with other values (e.g. `None`) are kept as lists.
- `SyncVecEnv.reset(mask=...)` and `AsyncVecEnv.reset(mask=...)` without shared memory raised `need at least one array to stack` for an all-False mask. With shared memory, `info` came back as `[]`. They now return 0-row context and observation with an empty `info` dict, like `BatchifyWrapper` and `FuncEnvBasedEnv`.
- `wrapper.sample_action()` (and the other `sample_*` helpers) pinned the wrapped env's rng on the wrapper, so a later `reset(seed=...)` that reseeds the env's rng (e.g. `SyncVecEnv`) no longer made wrapped sampling reproducible. Wrappers now sample through the wrapped env unless their own rng was set explicitly (`wrapper.rng = ...`).
- `FuncEnvBasedEnv.has_wrapper_attr` / `get_wrapper_attr` / `set_wrapper_attr` only looked at the outermost functional layer. Attributes of a `FuncEnv` wrapped in a `FuncEnvWrapper` (e.g. `FuncFrameStackWrapper`) were not found, and `set_wrapper_attr` silently created a new attribute on the wrapper.
- `EpisodeVideoWrapper(stream=True)` did not count episodes that ended inside `step()`, so every file was named `episode_0_step_<n>` (unlike `stream=False`).
- `set_wrapper_attr("render_mode" | "render_fps" | "backend" | "device" | "batch_size", ...)` on `Wrapper`, `FuncEnvWrapper`, `PolicyWrapper`, `FuncPolicyWrapper` and `FuncPolicyBasedPolicy` raised `property ... has no setter`. These read-only forwarding layers are now skipped, and the value is set on the layer that owns the attribute.
- `FuncEnvBasedEnv.close()` never called `render_close()` for functional envs whose `render_init` returns `render_state=None` (e.g. the world FuncEnv composer), and it left the renderer marked as started after `close()`.

**Remote execution**

- a client could make a `RemoteServer` allocate far more memory than `max_frame_size` (about 190x) by sending a small JPEG-compressed array. Servers now reject compressed arrays in requests (compression is server-to-client only), and the JPEG decoder checks image count and size before allocating.
- one failed `accept()` (e.g. `EMFILE` when out of file descriptors) or a failed connection setup stopped a TCP / Unix `RemoteServer` listener for good: the server kept running but never accepted another client. The listener now logs the error and retries.
- after `RemoteClient.release(name)` (or `close()` on a factory proxy), closing a subscription to that env raised `RemoteError: PermissionError`, and subscribing again never delivered events. Releasing now ends the env's subscriptions (`next()` raises `StopIteration`), the server drops them, and subscribing again works.
- a connection that closed while its `open` of a factory-hosted object was running could stay counted as a user, so the object was never released. Likewise, a `subscribe` processed as the connection closed left the dead connection subscribed, and every later step encoded an event for it.
- a client that sent requests and then stopped reading blocked the env's worker forever, stalling every other client of that env. New `RemoteServer(send_timeout=60.0)` closes a TCP / Unix connection whose sends make no progress for that long (`None` waits forever, the old behaviour).
- a `RemoteClient` with the JAX backend (x64 disabled) could not connect to a server hosting any env with a float64 / int64 space (`WireError: Invalid space: Unknown dtype string: float64`), and float64 / int64 results raised `TypeError` after the call had run. Torch clients could not receive float16 or uint16/32/64 data. Data and spaces now use the closest dtype the backend has (wider when lossless, e.g. float16 -> float32 or uint16 -> int32; otherwise narrower, as JAX does, e.g. float64 -> float32). A resource whose spaces still cannot be rebuilt no longer breaks the connection: only getting its proxy raises `TypeError`.
- with `compress={"image": ...}`, a two-alternative `UnionSpace` observation was compressed using the wrong alternative's metadata, so an exact label map could come back JPEG-compressed (lossy). Alternatives are now followed by their index.
- with `compress={"image": ...}`, a step returning a namedtuple observation failed with `TypeError: __new__() missing ... argument` instead of returning the observation.
- a `SystemExit` (or other `BaseException`) raised by a hosted object or a factory killed its worker thread: that call and every later call to the env hung, and the object was never closed. The error is now reported to the caller and the worker keeps running.
- when one subscriber's event could not be sent (e.g. too large uncompressed), later subscribers of the same env received nothing, even those whose JPEG-compressed event would fit.
- a `RemoteClient` with a smaller `max_frame_size` than its server lost its whole connection ("malformed frame") on the first larger reply or event. The client now sends its limit in `hello`. A too-large reply becomes an error for that call only, and a too-large event is skipped for that client. Older clients keep the previous behaviour.
- some malformed requests (an invalid non-finite `float` tag, absurd array shapes, an unhashable `open` / `release` name, undecodable compressed arrays) crashed the server's reader thread with an unhandled `TypeError` / `ValueError` traceback. They now raise `WireError`, and the connection is closed cleanly.
- a successful `reset` of a factory-hosted env raised `KeyError: 'version'` when another thread released that env while the reset was running.
- NumPy string scalars (`np.str_` / `np.bytes_`, e.g. an element of a string array in an `info` dict) could not be sent (`TypeError: Cannot send arrays of dtype <U4`). They now travel as `str` / `bytes`.
- `RemoteServer.listen("tcp://[::1]:port")` (any IPv6 address) failed with `Address family for hostname not supported`.

**Storages**

- In multiprocessing mode, a growing (`capacity=None`) replay buffer backed by `NPZStorage` or `ImageStorage` lost the samples it extended and could not read rows other processes had appended, because each process kept its own storage length. These storages now report `is_multiprocessing_safe=False` unless they have a fixed capacity or are read-only, so `ReplayBuffer(..., capacity=None, multiprocessing=True)` refuses them. Their `set()` now raises when the index selects fewer rows than the value holds; before, the extra rows were silently dropped.
- One invalid frame (wrong dtype or shape) appended to a `VideoStorage` discarded every frame already appended to the open segment. The replay buffer kept counting them, so later reads failed with "No episode files are available". A `DictStorage` was left with its children out of step, and every following append failed. Invalid frames are now rejected with `InvalidVideoFrameError` (a `ValueError`) before they reach the encoder, the open segment is kept, and `DictStorage` takes the rejected sample back from its other children.
- `HDF5Storage` text fields held `max_length` bytes rather than `max_length` characters. Valid text with multi-byte characters came back cut short (`"日ab"` → `"日a"`), or the whole read failed with `UnicodeDecodeError`. New files reserve room for any UTF-8 text of `max_length` characters. Existing files still load, and a string that was stored cut inside a character reads back without the broken last character.
- `clear()` on an empty growing (`capacity=None`) `HDF5Storage` or `ParquetStorage` raised `AssertionError: Length must be greater than 0`. This includes calling `ReplayBuffer.clear()` twice in a row.
- `VideoStorage.set(-1, frame)` no longer writes a stray `-1_-1` file and a bogus `(-1, -1)` segment; it raises, as `get(-1)` already did. `set()` with a repeated index (e.g. `[3, 3]`) now stores the last value for that row, as array assignment does; before, it stored the first.
- `HDF5Storage.set()` with a repeated index now stores the last value for that row, matching NumPy and the other storages; before, it stored the first.
- An `HDF5Storage` column view (`get_column`) kept the length the storage had when the view was created. After the storage grew, `len(column)` was stale and negative index arrays returned the wrong rows.
- The `"*"` key in `HDF5Storage.create(compression={...}, compression_level={...})` was ignored, so keys not listed by name were stored uncompressed (it already worked for `chunks`).

**Replay buffers, batches and samplers**

- Stopping a `MultiprocessingSampler` epoch iteration early and then exiting the script hung at interpreter exit (the requests no worker had taken kept the work queue's feeder thread waiting). Closing a sampler now releases its work queue.
- A script that used a `MultiprocessingSampler` and ended without calling `close()` hung forever at exit. Running samplers are now closed by an exit hook, before `multiprocessing` joins their (non-daemonic) workers. The `daemon` default is unchanged.
- An exception raised while a `MultiprocessingSampler` worker produced a batch (e.g. a storage read error) killed the worker silently, and `sample()` then blocked forever. The error is now re-raised in the main process as `SamplerWorkerError`, with the worker's exception and traceback attached. If a worker dies, the next fetch raises instead of waiting.
- `MultiprocessingSampler.manual_seed()` did not affect the batches that followed, because running workers kept their RNGs and their queued batches. Reseeding now restarts the `sample*` workers, so the same seed reproduces the same batches.
- `CombinedBatch` cached its children's lengths when it was built: samples added to a child later (e.g. an online replay buffer combined with an offline one) were never returned, and a child that shrank made reads fail. The view now follows its children's current lengths.
- In `ReplayBufferCollectionWrapper`, recording from a CUDA environment into a CUDA replay buffer failed with a device-mismatch error when the buffer stored `env_index` (and other converted scalars such as `reward`, the done flags or default episode-record fields). These values are now created on the buffer's device.
- `ReplayBufferCollectionWrapper.set_replay_buffers(..., dump_old=True)` did not dump the old buffers when none of them had an open segment: never in shared mode with a batched env, and never right after an episode ended.
- In shared mode with `batch_size=1`, `ReplayBufferCollectionWrapper.reset(mask=[False])` (e.g. calling `reset(mask=done)` every step) ended the open episode, and triggered the reset-boundary dump, although no env slot was reset. Every step was recorded as its own episode.
- With `SliceStackedBatch` / `FrameStackedBatch` and the episode-id valid mask, frames past either end of the batch (clamped to the first / last sample) were reported valid, unlike with the episode-end mask (e.g. `[T, T, T]` instead of `[T, F, F]` for the last step of the final episode with `postfetch_horizon=2`). Both built-in episode masks now mark such frames invalid. Frames before the first sample are therefore now also invalid under the episode-end mask; custom mask functions and stacking without a mask are unchanged.
- `ReplayBufferCollectionWrapper.set_replay_buffers(..., replay_mode="auto")` resolved the mode from the old buffers, so swapping from one shared buffer to one buffer per env slot (or back) raised a layout error.
- In per-env mode, `dump_on_reset_boundary` only dumped a buffer when the reset cut its episode short. Resetting a slot whose episode had already ended (terminated / truncated) now dumps too, as in shared mode.
- Indexing a `ReplayBuffer` (or its episode view, `get_episode_ids`, `set_at`, flattened access) with a NumPy integer or a 0-d NumPy / torch integer array failed with `Index shape () is not 1D`. Such indices now select one sample, like a Python `int`.

**Dataset integrations and utilities**

- `LeRobotAsUniEnvDataset` failed with "cannot reshape array of size 1 into shape (H,W,3)" on datasets whose images are stored in the parquet files (`dtype: "image"`, e.g. datasets recorded without videos); image features are now decoded (PNG/JPEG bytes or file path) to `(H, W, C)` uint8 frames.
- `LeRobotAsUniEnvDataset` failed with "cannot reshape array of size 1" (numpy) or "can't convert np.ndarray of type numpy.object_" (PyTorch) on 2-D features (HF `Array2D` columns, unless `datasets` happened to be imported) and on variable-length list columns; list columns of any depth are now read and length-checked.
- with the PyTorch backend, `LeRobotAsUniEnvDataset` raised "Index put requires the source and destination dtypes match" when a parquet column's dtype differed from `info.json` (e.g. float64 stored, float32 declared); values are cast to the declared dtype as on the numpy backend.
- LeRobot v3.0 datasets whose data files have no `index` column returned out-of-range or wrong rows for every episode outside the first data file.
- the video properties of LeRobot v2.1 / v3.0 datasets (stored under `features[key]["info"]`) were ignored, so a video frame rate differing from the dataset fps was not used; both `"info"` and the v2.0 `"video_info"` key are read.
- pickling a view of a view of a memory-mapped tensor (e.g. a row of a `PytorchTensorStorage` slice sent through a DataLoader worker or `multiprocessing` queue) rebuilt the wrong rows; every index is now kept.
- `HFAsUniEnvDataset.create()` failed with "Box dtype must be a real floating or integer type" on Hugging Face datasets with a string column when using the numpy backend; text columns are `TextSpace`s on every backend.
- `HFAsUniEnvDataset` returned nested (struct) columns as a list / object array of per-row dicts for index arrays and slices instead of a dict of batched arrays matching its space; an empty index now returns `(0, *shape)` arrays.
- `PyTorchAsUniEnvDataset` silently returned other rows for out-of-range index arrays (e.g. index `len` returned row 0); it now raises `IndexError`.
- wrapping a `UniEnvAsPyTorchDataset` in `PyTorchAsUniEnvDataset` failed with "string indices must be integers" for index arrays and slices.
- `ImageCompressTransformation(compression_ratio=<float>)` failed with "Expect all shape elements to be an integer".
- `render_board` (and `python -m unienv_utils.calibration print-board`) drew the ChArUco board about 4.5% too small with the default 20 px margin (19.1 mm instead of 20 mm checkers at 5000 px/m); the margins are now added around the board, so the image is `2 * margin_px` larger per axis and the checkers have the specified size.

## 0.0.1b16 (2026-10-17)

- **Changed:** `RemoteServer.register(name, obj)` now hosts only `obj`; it no longer exposes the world and nodes an env is built from. Use the new `RemoteServer.register_recursive(name, obj)` for that (components mode). Descriptor links (a node proxy's `world`, `nodes`, `get_node`) name only hosted objects, and objects sharing a world still share one worker.
- Extended the space metadata vocabulary with `frame`, `rotation`, `control`, `mode`, `normalized`, `range` and `names`, recommended `kind` values, the `"HW"` layout and the `"bgra"` format, plus a convention for vectors that pack several quantities.
- Added metadata builders (`metadata`, `image_metadata`, `depth_metadata`, `proprio_metadata`, `action_metadata`) and `annotate` to attach metadata to nested spaces by path.
- Added `metadata_issues` / `validate_metadata` to check space metadata against the vocabulary, and `common_metadata`.
- Fixed `CombinedWorldNode`, `FlatCombinedWorldNode` and their functional counterparts dropping the metadata of merged dict spaces.
- Added `Space.clip(x)` on every space (#4): projects a value of a superspace onto the subspace (`target.clip(x)` with `target.is_subspaceeq(source)`, so `target.contains(target.clip(x))`), dropping extra dict keys, clamping bounds, trimming over-long dynamic axes/text/graphs, or raising `ValueError` when a projection would require inventing data (missing keys, too-short text or dynamic axes, graphs below the minimum node/edge count).
- `BoxSpace.clip` casts to the box dtype, returns an array for 0-d boxes on NumPy, and is faster for boxes whose coordinates share one bound; `DynamicBoxSpace.clip` also trims axes longer than `shape_high`.
- Fixed `DynamicBoxSpace.is_subspaceeq` for per-element bounds on a fixed-size axis, `GraphSpace.is_subspaceeq` comparing edge counts of edge-less spaces, `GraphSpace.contains` raising on unexpected edge features, and `UnionSpace.contains` rejecting non-int64 NumPy integer indices.

## 0.0.1b15 (2026-10-16)

Requires `xbarray>=0.0.1a20` (faster NumPy random draws and `clip`, int shapes for PyTorch/JAX random draws).

**Remote execution was rewritten as a lightweight RPC layer** (protocol version 2; not compatible with the previous protocol):

- Frames are strict JSON plus raw binary array attachments (no pickle; malformed input is rejected), sent over TCP, Unix domain sockets, WebSocket (optional `websockets`, `network` extra) or in-memory. `RemoteServer.register(name, obj)` hosts an `Env`, `World` or `WorldNode` (a `WorldEnv` also exposes its world and nodes); each registered root runs on one worker thread; `RemoteClient.connect("tcp://…" | "unix://…" | "ws://…")` returns proxies (`RemoteEnv`, `RemoteWorld`, `RemoteWorldNode`) implementing the normal interfaces. Remote exceptions are re-raised as `RemoteError` with the remote traceback; a lost connection raises `ConnectionError` (an interrupted call may or may not have run on the server).
- Removed: sessions, ownership/acquire/release, begin/complete/abort operations, fault states, revisions, snapshot caches, idle eviction, factory and detached domains, `UncertainOutcomeError`, `Codec` and `RemoteWorldEnv`. Components mode is a plain local `WorldEnv` over remote world/node proxies (one call per hook).
- Observers: `client.subscribe(env)` yields each reset/step result (`SubscriptionEvent`), latest-wins.
- Opt-in JPEG compression of image observations, selected by space metadata: `RemoteClient.connect(addr, compress={"image": "jpeg:90"})` (worth it over real networks; raw is faster on loopback).
- `env.step` round trips drop from 303 / 930 / ~3200 µs (previous layer over WebSocket) to roughly 50 / 55 / 160 µs over TCP (small observation / 84×84×3 image / 64×84×84×3 batch).

**Space metadata:**

- Every space takes an optional, read-only, JSON-serializable `metadata=` mapping (`space.metadata`, `with_metadata()`, `SpaceMetadata`) with a documented, unenforced vocabulary (`kind`, `format`, `layout`, `units`, `source`, `fps`, `label`). It is ignored by `==`, `is_subspace` and `is_subspaceeq`, persisted by `space_to_json`/`json_to_space` and replay buffers, kept by `to()`, the batch utilities and meaning-preserving transformations/wrappers, and dropped by `flatten_space`. Helpers: `metadata_utils.spaces_equal_with_metadata`, `find_spaces_by_metadata`.
- `LeRobotAsUniEnvDataset` annotates image/video features (`kind`, `format`, `layout`, `fps`, `source`) and `observation.state`/`action` (`kind`).
- Episode render/video/wandb wrappers can record observations (`frame_source="observation"`), pick image leaves by `kind == "image"` and convert CHW/BGR/RGBA/gray to RGB, falling back to the previous behaviour without metadata.
- `unienv_data.storages.dict_storage_kwargs_from_space` / `create_dict_storage_from_space` build `DictStorage` layouts that send image leaves to `ImageStorage` (or a configured storage) and everything else to `HDF5Storage`.

A performance audit of the core code (benchmarks now live in `benchmarks/`) produced the following. Behaviour changes worth noting first:

- **`AsyncVecEnv` uses shared memory by default** (`shared_memory=True`) for observations, contexts and actions; groups that cannot be shared (spaces other than Box/Binary leaves in Dict/Tuple) fall back to the pipe with an INFO log, see `env.shared_groups`. Observations/contexts must match their declared spaces exactly (shape and dtype) in shared mode; pass `shared_memory=False` for the previous pipe-only behaviour.
- **`AsyncVecEnv` no longer builds an environment in the parent process**; workers report spaces, backend, device and metadata, and worker construction errors are raised by the constructor.
- **`BoxSpace`/`DynamicBoxSpace.sample` draws different values for a given seed** for integer boxes (all backends) and for float boxes on JAX; NumPy/PyTorch float samples are unchanged.
- **`HDF5Storage` chooses dataset layouts itself**: contiguous for fixed-capacity uncompressed data, whole-row ~64 KiB chunks otherwise (`chunks=(1,)` gives a per-step layout; explicit chunk tuples are honoured). Old and new files load in both versions; gzip-compressed files are somewhat larger.
- **`ParquetStorage` writes only complete pieces while growing**; the partial last piece is written by `dumps()` / `close()`.
- **`get_column` on `SliceStackedBatch`/`FrameStackedBatch`/`CombinedBatch`** reads only the selected column and returns a dedicated view class instead of `SubItemBatch` (values, spaces, metadata and flat layout are unchanged).
- `ControlFrequencyLimitWrapper` reports late actions through rate-limited `logging` warnings (`late_warning_interval`) instead of printing every step.
- In-place changes to spaces (`DictSpace.__setitem__`/`__delitem__`, mutating `DictSpace.spaces`, reassigning attributes) and reassigned transformation attributes invalidate cached spaces/constants; arrays or containers held by a transformation should be replaced rather than mutated in place.

Fixes:
- `MultiprocessingSampler` workers started with `fork` no longer deadlock after the parent ran a parallel torch op (workers run torch ops single-threaded).
- `FrameStackWrapper`/`FuncFrameStackWrapper` resets no longer modify stacked observations returned earlier (or, for the functional wrapper, the state passed in).
- `DictStorage` mixing a `VideoStorage` with other children no longer fails on the second `append()`.
- `ReplayBuffer(multiprocessing=True)` readers always see a consistent `(count, offset)` pair, and the buffer works with `spawn`/`forkserver` processes.
- Bounded float and integer `BoxSpace.sample` work inside `jax.jit`.
- Video readers never retry a hardware decoder that failed to initialise (e.g. CUDA hidden via `CUDA_VISIBLE_DEVICES`); forked workers previously segfaulted retrying it. `hwaccel="auto"` moves on to the next available device.

New:
- `AsyncVecEnv(shared_memory=..., copy=...)`: host shared memory for NumPy, PyTorch CPU and JAX CPU; zero-copy CUDA IPC for PyTorch CUDA and JAX GPU (JAX via DLPack, requires PyTorch); host staging for TPUs and other devices. `copy=False` returns buffers valid until the next `step`/`reset`. `ctx=None` keeps the platform default start method; CUDA sharing works with `fork` as long as the parent had not initialised CUDA before forking.
- `MultiprocessingSampler(shared_memory=True)` (off by default) returns `sample()`/`sample_with_metadata()` batches through pre-allocated shared memory.
- `LeRobotAsUniEnvDataset.from_local(video_hwaccel=...)` (default `"auto"`, unchanged).

Performance:
- Spaces: faster `BoxSpace` construction/`sample`/`contains`, `flat_dim`, `unflatten_data`, `set_at`, `batch_space`/`unbatch_spaces`, `ChainedTransformation` and `RescaleTransformation` (cached, pickle-safe plans; ~5x geometric mean across benchmarked operations).
- Wrappers: frame stacking 2–4x faster for NumPy/PyTorch Box/Binary observations; `ToBackendOrDeviceWrapper` skips no-op device moves.
- `AsyncVecEnv` with shared memory: ~3x (NumPy images) and ~20–35x (PyTorch CPU) more steps/s than the pipe with 8 workers; PyTorch CUDA ~1.3–1.5x.
- `HDF5Storage`: ~50x faster appends and ~200x faster random reads with the default layout; slice-based sparse IO.
- `ReplayBuffer`: interval-arithmetic segment bookkeeping, slice-based extends, faster `PytorchTensorStorage.append`; collection wrapper steps ~2.7x faster.
- File-backed storages (images, NPZ, video) decode on a shared, bounded IO thread pool; `VideoStorage` decodes forward within a GOP for sparse reads (use `encoder_options={"g": "15"}` for random-access sampling).
- Batches: column views of stacked/combined batches up to ~100x faster; slice/frame-stack fills re-gather only rows with invalid frames; `CombinedBatch` reuses its thread pool and memoises batched spaces.
- LeRobot: per-file parquet cache with column projection (~100x on tabular reads) and reused video readers (~95x on video reads).
- `ImageDecompressTransformation` decodes JPEG batches with torchvision when available (bit-identical to PIL).

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
- New `ReplayBuffer.open_segment_writer()` / `commit_segment()`: build several segments at once (e.g. one per env of a batched rollout) and append each at the tail as `extend()` would. `VideoStorage` writers encode frames as they arrive instead of staging them; other storages buffer in memory by default.
- `VideoStorage` takes `encoder_options` (passed to the encoder, persisted in the metadata).
- `VideoStorage` encodes 1-2 frame libx265 clips without B-frames: x265 derived their dts from uninitialised memory, so muxing intermittently failed with "pts/dts pair unsupported".
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