from .env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, batch_utils as sbu
from unienv_interface.utils.vec_util import MultiProcessFn
from typing import Any, Dict, Generic, Literal, Optional, SupportsFloat, Tuple, TypeVar, Callable, Iterable, Mapping, Sequence, List
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection as MPConnection
from multiprocessing.context import BaseContext as MPContext
from queue import Empty as QueueEmpty

def data_stack(
    data : Any,
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    device : Optional[BDeviceType] = None,
):
    if isinstance(data, Mapping):
        data = {
            key: data_stack(value, backend)
            for key, value in data.items()
        }
    if isinstance(data, Sequence):
        if len(data) == 0:
            return data
        if backend.is_backendarray(data[0]):
            return backend.stack(data, axis=0)
        elif isinstance(data[0], np.ndarray):
            return np.stack(data, axis=0)
        elif isinstance(data[0], (int, float, bool)):
            dtype = (
                backend.default_boolean_dtype if isinstance(data[0], bool) else
                backend.default_floating_dtype if isinstance(data[0], float) else
                backend.default_integer_dtype
            )
            return backend.asarray(data, dtype=dtype, device=device)
        elif isinstance(data[0], SupportsFloat):
            return backend.asarray(
                [float(d) for d in data],
                dtype=backend.default_floating_dtype,
                device=device
            )
        else:
            data = [
                data_stack(value, backend)
                for value in data
            ]
    return data

class SyncVecEnv(Env[
    BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
]):
    def __init__(
        self,
        env_fn : Iterable[Callable[[], Env[
            BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
        ]]],
        seed : Optional[int] = None,
    ):
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

        self.action_space = sbu.batch_differing_spaces(
            [env.action_space for env in self.envs],
            device=first_env.device,
        )
        self.observation_space = sbu.batch_differing_spaces(
            [env.observation_space for env in self.envs],
            device=first_env.device,
        )
        self.context_space = None if first_env.context_space is None else sbu.batch_differing_spaces(
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
                context, obs, info = env.reset(*args, **kwargs)
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

def _async_worker_fn(
    index : int,
    env_fn : Callable[[], Env[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]],
    pipe : MPConnection,
    parent_pipe : MPConnection,
    error_queue : mp.Queue,
) -> None:
    parent_pipe.close()
    del parent_pipe
    env = env_fn()
    try:
        while True:
            cmd, args, kwargs = pipe.recv()
            if cmd == "reset":
                context, observation, info = env.reset(*args, **kwargs)
                pipe.send(((context, observation, info), True))
            elif cmd == "step":
                observation, reward, terminated, truncated, info = env.step(*args, **kwargs)
                pipe.send(((observation, reward, terminated, truncated, info), True))
            elif cmd == "render":
                frame = env.render(*args, **kwargs)
                pipe.send((frame, True))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, Exception) as e:
        pipe.send(None, False)
        error_queue.put((index, e))
    finally:
        env.close()
        pipe.close()

class AsyncVecEnv(Env[
    BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
]):
    def __init__(
        self,
        env_fn : Iterable[Callable[[], Env[
            BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
        ]]],
        seed : Optional[int] = None,
        ctx : Optional[MPContext] = None,
        daemon : bool = True,
    ):
        ctx = ctx or mp.get_context()
        self.command_pipes : List[MPConnection] = []
        self.processes : List[mp.Process] = []
        self.error_queue : mp.Queue = ctx.Queue()

        dummy_fn = None
        for i, fn in enumerate(env_fn):
            if i == 0:
                dummy_fn = fn
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(
                target=_async_worker_fn,
                name="AsyncWorker-{}".format(i),
                args=(i, MultiProcessFn(fn), child_pipe, parent_pipe, self.error_queue),
                daemon=daemon,
            )
            process.start()
            child_pipe.close()

            self.command_pipes.append(parent_pipe)
            self.processes.append(process)

        assert len(self.processes) > 1, "env_fns should have more than 1 env function"

        # Use Dummy Environment to get spaces and metadata
        dummy_env = dummy_fn()
        self.backend = dummy_env.backend
        self.device = dummy_env.device
        self.metadata = dummy_env.metadata
        self.render_mode = dummy_env.render_mode
        self.render_fps = dummy_env.render_fps
        self.action_space = sbu.batch_space(
            dummy_env.action_space,
            len(self.processes),
        )
        self.observation_space = sbu.batch_space(
            dummy_env.observation_space,
            len(self.processes),
        )
        self.context_space = None if dummy_env.context_space is None else sbu.batch_space(
            dummy_env.context_space,
            len(self.processes),
        )
        self.rng = dummy_env.backend.random.random_number_generator(
            seed,
            device=dummy_env.device,
        )
        dummy_env.close()
        del dummy_env

        # Temporal mask storage
        self._reset_mask : Optional[BArrayType] = None

    @property
    def batch_size(self) -> Optional[int]:
        return len(self.processes)

    def send_command(self, index: int, cmd: Literal["reset", "step", "render", "close"], *args, **kwargs):
        self.command_pipes[index].send((cmd, args, kwargs))

    def get_command_result(self, index: int):
        data, success = self.command_pipes[index].recv()
        if not success:
            self._raise_if_error()
        return data

    def reset(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        self.reset_async(*args, mask=mask, seed=seed, **kwargs)
        return self.reset_wait()

    def reset_async(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> None:
        if seed is not None:
            self.rng = self.backend.random.random_number_generator(
                seed,
                device=self.device,
            )
        
        self._reset_mask = mask
        for i in range(len(self.processes)):
            env_reset = True if mask is None else bool(mask[i])
            if env_reset:
                self.send_command(i, "reset", *args, **kwargs)
    
    def reset_wait(
        self,
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        all_contexts = []
        all_obs = []
        all_infos = []
        for i in range(len(self.processes)):
            env_reset = True if self._reset_mask is None else bool(self._reset_mask[i])
            if env_reset:
                context, obs, info = self.get_command_result(i)
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

        self._reset_mask = None
        return all_contexts, all_obs, all_infos
    
    def step(
        self,
        action : ActType
    ) -> Tuple[ObsType, BArrayType, BArrayType, BArrayType, Dict[str, Any]]:
        self.step_async(action)
        return self.step_wait()
    
    def step_async(
        self,
        action : ActType
    ) -> None:
        actions = sbu.iterate(
            self.action_space,
            action
        )
        for i in range(len(self.processes)):
            self.send_command(i, "step", next(actions))
    
    def step_wait(
        self,
    ) -> Tuple[ObsType, BArrayType, BArrayType, BArrayType, Dict[str, Any]]:
        all_obs = []
        all_rewards = []
        all_terminated = []
        all_truncated = []
        all_infos = []
        for i in range(len(self.processes)):
            obs, reward, terminated, truncated, info = self.get_command_result(i)
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

    def close(self):
        for i in range(len(self.processes)):
            try:
                self.send_command(i, "close")
            except Exception:
                pass
        for process in self.processes:
            process.join()
            process.close()
        for pipe in self.command_pipes:
            pipe.close()
        self.command_pipes = []
        self.processes = []
    
    def _raise_if_error(self):
        try:
            index, e = self.error_queue.get(block=False)
        except QueueEmpty:
            raise RuntimeError("Unknown error in AsyncVecEnv worker without any error message.")
        raise RuntimeError("Error in AsyncVecEnv worker {}".format(index)) from e