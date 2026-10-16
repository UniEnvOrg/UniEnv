from .env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, batch_utils as sbu
from unienv_interface.space.space_utils.serialization_utils import space_to_json, json_to_space
from ._shared_memory import (
    ParentSharedBuffers, WorkerSharedBuffers, check_leaf_layouts_match, select_mode, space_leaves,
)
from typing import Any, Dict, Generic, Literal, Optional, SupportsFloat, Tuple, TypeVar, Callable, Iterable, Mapping, Sequence, List
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection as MPConnection
from multiprocessing.context import BaseContext as MPContext
from queue import Empty as QueueEmpty
import atexit
import os
import pickle
import traceback
import warnings
import weakref
import cloudpickle

def data_stack(
    data : Any,
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    device : Optional[BDeviceType] = None,
):
    """Recursively stack per-environment data into a batched backend representation.

    * A mapping is processed value by value.
    * A sequence of mappings that share the same keys (e.g. one ``info`` dict
      per sub-environment) becomes a mapping of stacked values.
    * A sequence of arrays or Python numbers is stacked into one array.
    * Strings, bytes and other objects are kept as a list.
    """
    if isinstance(data, Mapping):
        return {
            key: data_stack(value, backend, device)
            for key, value in data.items()
        }
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        return data
    if len(data) == 0:
        return data
    first = data[0]
    if isinstance(first, Mapping):
        keys = first.keys()
        if all(isinstance(d, Mapping) and d.keys() == keys for d in data):
            return {
                key: data_stack([d[key] for d in data], backend, device)
                for key in keys
            }
        return [data_stack(value, backend, device) for value in data]
    if backend.is_backendarray(first):
        return backend.stack(data, axis=0)
    elif isinstance(first, np.ndarray):
        return np.stack(data, axis=0)
    elif isinstance(first, (int, float, bool)):
        dtype = (
            backend.default_boolean_dtype if isinstance(first, bool) else
            backend.default_floating_dtype if isinstance(first, float) else
            backend.default_integer_dtype
        )
        return backend.asarray(data, dtype=dtype, device=device)
    elif isinstance(first, (str, bytes)) or first is None:
        return list(data)
    elif isinstance(first, SupportsFloat):
        return backend.asarray(
            [float(d) for d in data],
            dtype=backend.default_floating_dtype,
            device=device
        )
    return [
        data_stack(value, backend, device)
        for value in data
    ]

def _batch_env_spaces(
    spaces : Sequence[Optional[Space]],
    device : Optional[BDeviceType] = None,
) -> Optional[Space]:
    """Batch the per-environment spaces (``None`` if the envs have no such space)."""
    first = spaces[0]
    if first is None:
        return None
    if all(space == first for space in spaces[1:]):
        return sbu.batch_space(first, len(spaces))
    return sbu.batch_differing_spaces(spaces, device=device)

def _sub_env_seed(seed : Optional[int], index : int) -> Optional[int]:
    """Seed used for sub-environment ``index`` when the vectorized env is reset with ``seed``."""
    return None if seed is None else seed + index

class SyncVecEnv(Env[
    BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
]):
    """Run multiple non-batched environments synchronously in one process.

    Seeding: ``reset(seed=s)`` reseeds ``rng`` with ``s`` and resets
    sub-environment ``i`` with seed ``s + i`` (``i`` is the index in the
    full batch, also when only a masked subset is reset), so every
    sub-environment gets a distinct, reproducible seed.
    """
    def __init__(
        self,
        env_fn : Iterable[Callable[[], Env[
            BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
        ]]],
        seed : Optional[int] = None,
    ):
        """Instantiate every environment immediately and batch their interfaces."""
        self.envs = [fn() for fn in env_fn]
        assert len(self.envs) > 1, "env_fns should have more than 1 env function"
        assert all(env.batch_size is None for env in self.envs), "All envs must be non-batched envs"

        # check all envs have the same backend, device, action_space, observation_space, context_space
        first_env = self.envs[0]
        for env in self.envs[1:]:
            assert env.backend == first_env.backend, "All envs must have the same backend"
            assert env.device == first_env.device, "All envs must have the same device"
            # assert env.action_space == first_env.action_space, "All envs must have the same action_space"
            # assert env.observation_space == first_env.observation_space, "All envs must have the same observation_space"
            # assert env.context_space == first_env.context_space, "All envs must have the same context_space"

        self.action_space = _batch_env_spaces(
            [env.action_space for env in self.envs],
            device=first_env.device,
        )
        self.observation_space = _batch_env_spaces(
            [env.observation_space for env in self.envs],
            device=first_env.device,
        )
        self.context_space = _batch_env_spaces(
            [env.context_space for env in self.envs],
            device=first_env.device,
        )
        self.rng = self.backend.random.random_number_generator(
            seed,
            device=first_env.device,
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.envs[0].metadata

    @property
    def render_mode(self) -> Optional[str]:
        return self.envs[0].render_mode

    @property
    def render_fps(self) -> Optional[int]:
        return self.envs[0].render_fps
    
    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.envs[0].backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.envs[0].device
    
    @property
    def batch_size(self) -> Optional[int]:
        return len(self.envs)
    
    def reset(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        """Reset all selected workers and concatenate their results."""
        if seed is not None:
            self.rng = self.backend.random.random_number_generator(
                seed,
                device=self.device,
            )
        
        all_contexts = []
        all_obs = []
        all_infos = []
        for i, env in enumerate(self.envs):
            env_reset = True if mask is None else bool(mask[i])
            if env_reset:
                context, obs, info = env.reset(*args, seed=_sub_env_seed(seed, i), **kwargs)
                all_contexts.append(context)
                all_obs.append(obs)
                all_infos.append(info)

        if self.context_space is not None:
            all_contexts = sbu.concatenate(
                self.context_space,
                all_contexts,
                axis=0,
            )
        else:
            all_contexts = None

        all_obs = sbu.concatenate(
            self.observation_space,
            all_obs,
            axis=0,
        )
        all_infos = data_stack(
            all_infos,
            self.backend,
            self.device
        )
        return all_contexts, all_obs, all_infos

    def step(
        self,
        action : ActType
    ) -> Tuple[ObsType, BArrayType, BArrayType, BArrayType, Dict[str, Any]]:
        """Step every worker environment once and concatenate the outputs."""
        actions = sbu.iterate(
            self.action_space,
            action
        )
        all_obs = []
        all_rewards = []
        all_terminated = []
        all_truncated = []
        all_infos = []
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(next(actions))
            all_obs.append(obs)
            all_rewards.append(reward)
            all_terminated.append(terminated)
            all_truncated.append(truncated)
            all_infos.append(info)
        all_obs = sbu.concatenate(
            self.observation_space,
            all_obs,
            axis=0,
        )
        all_rewards = self.backend.asarray(
            [float(r) for r in all_rewards],
            dtype=self.backend.default_floating_dtype,
            device=self.device,
        )
        all_terminated = self.backend.asarray(
            [bool(t) for t in all_terminated],
            dtype=self.backend.default_boolean_dtype,
            device=self.device,
        )
        all_truncated = self.backend.asarray(
            [bool(t) for t in all_truncated],
            dtype=self.backend.default_boolean_dtype,
            device=self.device,
        )
        all_infos = data_stack(
            all_infos,
            self.backend,
            self.device
        )
        return all_obs, all_rewards, all_terminated, all_truncated, all_infos
    
    def render(self) -> Sequence[RenderFrame] | None:
        """Collect render outputs from every worker that returns a frame."""
        frames = []
        for env in self.envs:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        return frames if len(frames) > 0 else None

    def close(self):
        for env in self.envs:
            env.close()
        self.envs = []

def _serialize_device(
    backend : ComputeBackend,
    device : Optional[BDeviceType],
) -> Optional[Tuple[str, Any]]:
    """Picklable description of ``device`` (``None`` if it cannot be described).

    Picklable devices (e.g. ``torch.device``) are sent as-is so that they compare
    equal in the parent; others (JAX devices) go through ``backend.serialize_device``.
    """
    if device is None:
        return None
    if backend.simplified_name != "jax":
        try:
            pickle.dumps(device)
            return ("object", device)
        except Exception:
            pass
    try:
        return ("str", backend.serialize_device(device))
    except Exception:
        return None

def _deserialize_device(
    backend : ComputeBackend,
    device : Optional[Tuple[str, Any]],
) -> Optional[BDeviceType]:
    """Inverse of :func:`_serialize_device`; falls back to the backend's default device
    (``None``) if this process cannot see the worker's device."""
    if device is None:
        return None
    kind, value = device
    if kind == "object":
        return value
    try:
        return backend.deserialize_device(value)
    except Exception as e:
        warnings.warn(
            f"AsyncVecEnv: the worker device {value!r} is not available in the parent process "
            f"({type(e).__name__}: {e}); using the backend's default device instead."
        )
        return None

def _serialize_space(space : Optional[Space]) -> Optional[Tuple[str, Any, Any]]:
    """Describe a space so that it can be sent to the parent process.

    Spaces are serialized with :func:`space_to_json` (plus their device) rather
    than pickled, so that no backend array (e.g. a CUDA tensor, which would be
    shared through CUDA IPC) crosses the pipe. Spaces without a JSON
    representation fall back to cloudpickle.
    """
    if space is None:
        return None
    try:
        return ("json", space_to_json(space), _serialize_device(space.backend, space.device))
    except Exception:
        return ("pickle", cloudpickle.dumps(space), None)

def _deserialize_space(
    payload : Optional[Tuple[str, Any, Any]],
    backend : ComputeBackend,
) -> Optional[Space]:
    if payload is None:
        return None
    kind, data, device = payload
    if kind == "json":
        return json_to_space(data, backend, _deserialize_device(backend, device))
    return cloudpickle.loads(data)

def _describe_env(env : Env) -> Dict[str, Any]:
    """Everything the parent process needs to know about a worker environment."""
    return {
        "backend": env.backend,
        "device": _serialize_device(env.backend, env.device),
        "action_space": _serialize_space(env.action_space),
        "observation_space": _serialize_space(env.observation_space),
        "context_space": _serialize_space(env.context_space),
        "metadata": env.metadata,
        "render_mode": env.render_mode,
        "render_fps": env.render_fps,
    }

# Shared-memory envs that have not been closed yet. They are closed at interpreter
# exit (this hook runs before multiprocessing's own exit handler, which would
# terminate the workers), so that the workers release their CUDA IPC references
# before the parent frees the buffers and host segments are unlinked.
_LIVE_SHARED_ENVS : "weakref.WeakSet[AsyncVecEnv]" = weakref.WeakSet()

@atexit.register
def _close_live_shared_envs() -> None:
    for env in list(_LIVE_SHARED_ENVS):
        try:
            env.close()
        except Exception:
            pass

class _LazyEnvFn:
    """Callable that is cloudpickled when sent to a worker and only unpickled
    when called, i.e. inside the worker after its environment variables are set
    (unpickling an env factory may already import and configure JAX)."""
    def __init__(self, fn : Callable):
        self._fn = fn
        self._payload = None

    def __getstate__(self):
        return cloudpickle.dumps(self._fn)

    def __setstate__(self, payload):
        self._fn = None
        self._payload = payload

    def __call__(self, *args, **kwargs):
        if self._fn is None:
            self._fn = cloudpickle.loads(self._payload)
            self._payload = None
        return self._fn(*args, **kwargs)

def _async_worker_fn(
    index : int,
    env_fn : Callable[[], Env[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]],
    pipe : MPConnection,
    parent_pipe : MPConnection,
    error_queue : mp.Queue,
    env_var_defaults : Optional[Dict[str, str]] = None,
) -> None:
    parent_pipe.close()
    del parent_pipe
    for key, value in (env_var_defaults or {}).items():
        os.environ.setdefault(key, value)
    env = None
    shared : Optional[WorkerSharedBuffers] = None
    try:
        env = env_fn()
        while True:
            cmd, args, kwargs = pipe.recv()
            if cmd == "reset":
                context, observation, info = env.reset(*args, **kwargs)
                if shared is None:
                    pipe.send(((context, observation, info), True))
                else:
                    if shared.has_group("context"):
                        shared.write("context", context)
                    shared.write("obs", observation)
                    shared.finish()
                    pipe.send(((None, None, info), True))
            elif cmd == "step":
                observation, reward, terminated, truncated, info = env.step(*args, **kwargs)
                if shared is None:
                    pipe.send(((observation, reward, terminated, truncated, info), True))
                else:
                    shared.write("obs", observation)
                    shared.finish()
                    pipe.send(((None, float(reward), bool(terminated), bool(truncated), info), True))
            elif cmd == "render":
                frame = env.render(*args, **kwargs)
                pipe.send((frame, True))
            elif cmd == "_describe":
                pipe.send((_describe_env(env), True))
            elif cmd == "_shared_setup":
                shared = WorkerSharedBuffers(args[0], index, env.backend)
                pipe.send((None, True))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, Exception) as e:
        # Queue the exception before answering so the parent can pick it up,
        # and send a textual description in case the exception is not picklable.
        error_queue.put((index, e))
        try:
            pipe.send(((type(e).__name__, str(e), traceback.format_exc()), False))
        except Exception:
            pass
    finally:
        # Drop the shared buffers first: CUDA IPC buffers must be released by
        # every consumer before the parent (their producer) frees them.
        if shared is not None:
            shared.close()
            shared = None
        if env is not None:
            env.close()
        pipe.close()

class AsyncVecEnv(Env[
    BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
]):
    """Run multiple non-batched environments in separate worker processes.

    Seeding follows :class:`SyncVecEnv`: ``reset(seed=s)`` resets worker ``i``
    with seed ``s + i``. If a worker raises, the exception is re-raised in the
    parent as a ``RuntimeError`` (chained to the worker's exception) and that
    worker process exits.

    No environment is built in the parent process: the spaces, backend, device,
    metadata and render settings are reported by the workers (spaces travel as
    :func:`space_to_json` payloads, so this works with every start method), and
    the constructor raises if a worker fails to build its environment or if the
    workers disagree on backend or device. Differing sub-environment spaces are
    batched like in :class:`SyncVecEnv`.

    **Shared-memory observations** (``shared_memory=True``, opt-in): the parent
    allocates one buffer of shape ``(num_envs, *leaf_shape)`` per leaf of the
    single-env observation space (and of the context space), worker ``i`` writes
    its observation / context into row ``i``, and only reward, termination flags
    and ``info`` travel through the pipe, so observations are neither pickled
    nor concatenated. Where the buffers live follows the worker env:

    ================  ===============================  =================================
    worker env        buffer                           parent read
    ================  ===============================  =================================
    numpy             host shared memory               zero-copy numpy view
    torch CPU         host shared memory               zero-copy torch view
    torch CUDA        CUDA buffer owned by the         zero-copy on the GPU
                      parent, shared via CUDA IPC
    jax CPU           host shared memory               ``copy=True``: one copy into a
                                                       fresh jax array; ``copy=False``:
                                                       zero-copy via DLPack (falls back
                                                       to a copy if XLA refuses)
    jax GPU           the same CUDA IPC buffer, with   zero-copy via DLPack
                      torch as carrier (DLPack in/out)
    jax TPU / other   host shared memory (the worker   one host-to-device transfer per
    devices           copies device -> host)           read
    ================  ===============================  =================================

    Requirements and caveats of shared mode:

    * Only fixed-size spaces are supported: ``BoxSpace`` / ``BinarySpace`` leaves
      nested in ``DictSpace`` / ``TupleSpace`` (anything else raises a
      ``ValueError`` from the constructor). Every observation / context must
      match its declared space exactly (shape and dtype per leaf); a mismatch
      raises in the worker and surfaces as the usual chained ``RuntimeError``,
      nothing is cast silently. ``info`` always travels through the pipe.
    * ``copy=True`` (default) returns data that never changes afterwards.
      ``copy=False`` returns the shared buffers themselves (or zero-copy views
      of them): they are only valid until the next ``step`` / ``reset`` (which
      overwrite them in place) and must not be written to; copy anything you
      want to keep, and make sure asynchronous consumers (e.g. JAX
      computations) have finished before the next call. CUDA buffers are
      synchronised by the parent before every ``step`` / ``reset``. Masked
      resets always return freshly gathered rows.
    * CUDA sharing needs the ``spawn`` or ``forkserver`` start method; an
      explicit ``fork`` context raises. When ``ctx`` is not given, shared mode
      uses ``spawn`` (the env's device is only known once a worker built it).
      CUDA IPC is Linux-only; elsewhere CUDA observations are staged through
      host shared memory (with a warning). JAX GPU envs need PyTorch as the
      IPC carrier and a CUDA-enabled jaxlib in the parent. A TPU belongs to a
      single process, so TPU envs cannot be shared: run the workers on the
      host (or let them copy to the host) and the parent performs one
      host-to-TPU transfer per read.
    * Workers set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` unless the variable
      is already set, so that N JAX workers do not each reserve 75% of the GPU.
    * The parent owns the buffers until :meth:`close`, which first shuts the
      workers down; host shared memory is unlinked on ``close`` (also after a
      worker crash) and when the env is garbage collected, and shared-memory
      envs that are still open at interpreter exit are closed then. Views
      returned with ``copy=False`` stay readable after ``close`` (the mapping
      is released with the last view). A worker that is killed hard (e.g.
      ``SIGKILL``) cannot release its CUDA IPC references, in which case torch
      may warn about it when the parent exits.
    """
    def __init__(
        self,
        env_fn : Iterable[Callable[[], Env[
            BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
        ]]],
        seed : Optional[int] = None,
        ctx : Optional[MPContext] = None,
        daemon : bool = True,
        shared_memory : bool = False,
        copy : bool = True,
    ):
        """Spawn worker processes and batch the spaces they report.

        Args:
            env_fn: One factory per sub-environment; each builds an unbatched env.
            seed: Seed of the vector env's ``rng``.
            ctx: Multiprocessing context. Defaults to the platform default, or to
                ``spawn`` when ``shared_memory=True``.
            daemon: Whether the worker processes are daemonic.
            shared_memory: Transfer observations / contexts through shared
                buffers instead of pickling them (see the class docstring).
            copy: Only used with ``shared_memory=True``: return copies (``True``)
                or views that are valid until the next ``step`` / ``reset``.
        """
        if ctx is None:
            ctx = mp.get_context("spawn") if shared_memory else mp.get_context()
        self.command_pipes : List[MPConnection] = []
        self.processes : List[mp.Process] = []
        self.error_queue : mp.Queue = ctx.Queue()
        self.shared_memory = shared_memory
        self.copy = copy
        self._shared : Optional[ParentSharedBuffers] = None
        self._reset_mask : Optional[BArrayType] = None
        env_var_defaults = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"} if shared_memory else None

        for i, fn in enumerate(env_fn):
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(
                target=_async_worker_fn,
                name="AsyncWorker-{}".format(i),
                args=(i, _LazyEnvFn(fn), child_pipe, parent_pipe, self.error_queue, env_var_defaults),
                daemon=daemon,
            )
            process.start()
            child_pipe.close()

            self.command_pipes.append(parent_pipe)
            self.processes.append(process)

        try:
            assert len(self.processes) > 1, "env_fns should have more than 1 env function"
            device_description = self._init_from_workers(seed)
            if shared_memory:
                self._init_shared_memory(device_description, ctx.get_start_method())
        except BaseException:
            self.close()
            raise

    def _init_from_workers(self, seed : Optional[int]) -> Any:
        """Query every worker for its environment description and batch the spaces.

        Returns the worker-reported (serialized) device.
        """
        for i in range(len(self.processes)):
            self.send_command(i, "_describe")
        descriptions = self._get_command_results(range(len(self.processes)))
        first = descriptions[0]
        for i, description in enumerate(descriptions[1:], start=1):
            if description["backend"] != first["backend"]:
                raise ValueError(
                    f"All envs must have the same backend, worker 0 uses {first['backend']} "
                    f"but worker {i} uses {description['backend']}"
                )
            if description["device"] != first["device"]:
                raise ValueError(
                    f"All envs must have the same device, worker 0 uses {first['device']} "
                    f"but worker {i} uses {description['device']}"
                )

        self.backend = first["backend"]
        self.device = _deserialize_device(self.backend, first["device"])
        self.metadata = first["metadata"]
        self.render_mode = first["render_mode"]
        self.render_fps = first["render_fps"]

        def single_spaces(key : str) -> List[Optional[Space]]:
            first_space = _deserialize_space(first[key], self.backend)
            return [
                first_space if description[key] == first[key]
                else _deserialize_space(description[key], self.backend)
                for description in descriptions
            ]

        self._single_observation_spaces = single_spaces("observation_space")
        self._single_context_spaces = single_spaces("context_space")
        self.action_space = _batch_env_spaces(single_spaces("action_space"), device=self.device)
        self.observation_space = _batch_env_spaces(self._single_observation_spaces, device=self.device)
        self.context_space = _batch_env_spaces(self._single_context_spaces, device=self.device)
        self.rng = self.backend.random.random_number_generator(
            seed,
            device=self.device,
        )
        return first["device"]

    def _init_shared_memory(self, device_description : Any, start_method : str) -> None:
        """Allocate the shared buffers and hand them to the workers."""
        groups = {"obs": self._single_observation_spaces[0]}
        space_leaves(groups["obs"], "observation")
        check_leaf_layouts_match(self.backend, self._single_observation_spaces, "observation")
        if self.context_space is not None:
            groups["context"] = self._single_context_spaces[0]
            space_leaves(groups["context"], "context")
            check_leaf_layouts_match(self.backend, self._single_context_spaces, "context")
        mode = select_mode(self.backend, device_description, start_method)
        self._shared = ParentSharedBuffers(
            self.backend,
            self.device,
            len(self.processes),
            groups,
            mode,
            self.copy,
        )
        _LIVE_SHARED_ENVS.add(self)
        spec = self._shared.worker_spec()
        for i in range(len(self.processes)):
            self.send_command(i, "_shared_setup", spec)
        self._get_command_results(range(len(self.processes)))

    @property
    def batch_size(self) -> Optional[int]:
        return len(self.processes)

    def send_command(self, index: int, cmd: Literal["reset", "step", "render", "close", "_describe", "_shared_setup"], *args, **kwargs):
        """Send a command to one worker process."""
        self.command_pipes[index].send((cmd, args, kwargs))

    def get_command_result(self, index: int):
        """Receive one worker result and surface worker-side exceptions."""
        data, success = self._recv(index)
        if not success:
            self._raise_if_error(index, data)
        return data

    def _recv(self, index: int) -> Tuple[Any, bool]:
        try:
            return self.command_pipes[index].recv()
        except EOFError:
            return ("EOFError", "worker process exited unexpectedly", ""), False

    def _get_command_results(self, indices: Iterable[int]) -> List[Any]:
        """Receive the results of several workers, raising the first worker error
        only after every pending result was received (so the pipes stay in sync)."""
        results = [(i, *self._recv(i)) for i in indices]
        for i, data, success in results:
            if not success:
                self._raise_if_error(i, data)
        return [data for _, data, _ in results]

    def reset(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        """Convenience wrapper around ``reset_async`` plus ``reset_wait``."""
        self.reset_async(*args, mask=mask, seed=seed, **kwargs)
        return self.reset_wait()

    def reset_async(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> None:
        """Dispatch reset commands to all selected workers."""
        if seed is not None:
            self.rng = self.backend.random.random_number_generator(
                seed,
                device=self.device,
            )
        
        if self._shared is not None:
            self._shared.before_dispatch()
        self._reset_mask = mask
        for i in range(len(self.processes)):
            env_reset = True if mask is None else bool(mask[i])
            if env_reset:
                self.send_command(i, "reset", *args, seed=_sub_env_seed(seed, i), **kwargs)
    
    def reset_wait(
        self,
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        """Collect the pending reset results and batch them."""
        all_contexts = []
        all_obs = []
        all_infos = []
        indices = [
            i for i in range(len(self.processes))
            if self._reset_mask is None or bool(self._reset_mask[i])
        ]
        self._reset_mask = None
        results = self._get_command_results(indices)
        if self._shared is not None:
            rows = None if len(indices) == len(self.processes) else indices
            all_contexts = None if self.context_space is None else self._shared.read("context", rows)
            all_obs = self._shared.read("obs", rows)
            all_infos = data_stack(
                [info for _, _, info in results],
                self.backend,
                self.device
            )
            return all_contexts, all_obs, all_infos

        for context, obs, info in results:
            all_contexts.append(context)
            all_obs.append(obs)
            all_infos.append(info)

        if self.context_space is not None:
            all_contexts = sbu.concatenate(
                self.context_space,
                all_contexts,
                axis=0,
            )
        else:
            all_contexts = None

        all_obs = sbu.concatenate(
            self.observation_space,
            all_obs,
            axis=0,
        )
        all_infos = data_stack(
            all_infos,
            self.backend,
            self.device
        )

        return all_contexts, all_obs, all_infos
    
    def step(
        self,
        action : ActType
    ) -> Tuple[ObsType, BArrayType, BArrayType, BArrayType, Dict[str, Any]]:
        """Convenience wrapper around ``step_async`` plus ``step_wait``."""
        self.step_async(action)
        return self.step_wait()
    
    def step_async(
        self,
        action : ActType
    ) -> None:
        """Dispatch one action batch to all worker processes."""
        actions = sbu.iterate(
            self.action_space,
            action
        )
        if self._shared is not None:
            self._shared.before_dispatch()
        for i in range(len(self.processes)):
            self.send_command(i, "step", next(actions))
    
    def step_wait(
        self,
    ) -> Tuple[ObsType, BArrayType, BArrayType, BArrayType, Dict[str, Any]]:
        """Collect the pending step results and batch them."""
        all_obs = []
        all_rewards = []
        all_terminated = []
        all_truncated = []
        all_infos = []
        for obs, reward, terminated, truncated, info in self._get_command_results(range(len(self.processes))):
            all_obs.append(obs)
            all_rewards.append(reward)
            all_terminated.append(terminated)
            all_truncated.append(truncated)
            all_infos.append(info)
        if self._shared is not None:
            all_obs = self._shared.read("obs")
        else:
            all_obs = sbu.concatenate(
                self.observation_space,
                all_obs,
                axis=0,
            )
        all_rewards = self.backend.asarray(
            [float(r) for r in all_rewards],
            dtype=self.backend.default_floating_dtype,
            device=self.device,
        )
        all_terminated = self.backend.asarray(
            [bool(t) for t in all_terminated],
            dtype=self.backend.default_boolean_dtype,
            device=self.device,
        )
        all_truncated = self.backend.asarray(
            [bool(t) for t in all_truncated],
            dtype=self.backend.default_boolean_dtype,
            device=self.device,
        )
        all_infos = data_stack(
            all_infos,
            self.backend,
            self.device
        )
        return all_obs, all_rewards, all_terminated, all_truncated, all_infos

    def render(self) -> Sequence[RenderFrame] | None:
        """Collect render outputs from every worker that returns a frame."""
        for i in range(len(self.processes)):
            self.send_command(i, "render")
        frames = [
            frame for frame in self._get_command_results(range(len(self.processes)))
            if frame is not None
        ]
        return frames if len(frames) > 0 else None

    def close(self):
        """Shut down all worker processes and communication pipes, then release
        the shared observation buffers (if any). Safe to call more than once."""
        for i in range(len(self.processes)):
            try:
                self.send_command(i, "close")
            except Exception:
                pass
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()
        for pipe in self.command_pipes:
            pipe.close()
        self.command_pipes = []
        self.processes = []
        # Only release the shared buffers once every worker (consumer) has exited.
        if self._shared is not None:
            self._shared.close()
            self._shared = None
        _LIVE_SHARED_ENVS.discard(self)
    
    def _raise_if_error(self, index: Optional[int] = None, error_data: Any = None):
        """Raise the worker exception captured in the error queue.

        ``error_data`` is the ``(type name, message, traceback)`` description sent
        by the worker, used when the exception itself cannot be retrieved.
        """
        description = ""
        if error_data is not None:
            name, message, tb = error_data
            description = f": {name}: {message}\n{tb}"
        try:
            # The worker queues the exception before reporting failure, but the
            # queue's feeder thread may still be flushing it.
            error_index, e = self.error_queue.get(timeout=5.0)
        except QueueEmpty:
            if index is None and error_data is None:
                raise RuntimeError("Unknown error in AsyncVecEnv worker without any error message.")
            raise RuntimeError("Error in AsyncVecEnv worker {}{}".format(index, description))
        raise RuntimeError("Error in AsyncVecEnv worker {}{}".format(error_index, description)) from e
