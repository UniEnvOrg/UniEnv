from typing import Dict, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence
import gymnasium as gym
import numpy as np
import copy
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import Space, DictSpace
from unienv_interface.space.space_utils import gym_utils
from unienv_interface.utils import seed_util

class ToGymnasiumEnv(
    gym.Env[Any, Any],
    Generic[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]
):
    def __init__(
        self,
        env: Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]
    ):
        assert env.batch_size is None, "Parallel Environments is not supported by this wrapper."
        self.env = env

        self._metadata : Optional[Dict[str, Any]] = None
        self.action_space = gym_utils.to_gym_space(env.action_space)
        self._unienv_combined_space = __class__.combine_context_obs_space(
            env.context_space, env.observation_space
        )
        self.observation_space = gym_utils.to_gym_space(self._unienv_combined_space)
        self._episode_context : Optional[ContextType] = None
        self._np_rng : Optional[np.random.Generator] = None

    @staticmethod
    def combine_context_obs_space(
        context_space: Space[ContextType, BDeviceType, BDeviceType, BRNGType],
        observation_space: Space[ObsType, BDeviceType, BDtypeType, BRNGType]
    ) -> Space[Union[ContextType, ObsType], BDeviceType, BDtypeType, BRNGType]:
        if isinstance(context_space, DictSpace) and isinstance(observation_space, DictSpace):
            spaces = context_space.spaces.copy()
            spaces.update(observation_space.spaces)
        elif isinstance(context_space, DictSpace):
            spaces = context_space.spaces.copy()
            spaces["observation"] = observation_space
        elif isinstance(observation_space, DictSpace):
            spaces = observation_space.spaces.copy()
            spaces["context"] = context_space
        else:
            spaces = {
                "context": context_space,
                "observation": observation_space
            }

        return DictSpace(
            backend=context_space.backend,
            spaces=spaces,
            device=context_space.device,
        )
    
    @staticmethod
    def combine_context_obs(
        context: ContextType,
        observation: ObsType
    ) -> Dict[str, Any]:
        if isinstance(context, Dict) and isinstance(observation, Dict):
            new_obs = context.copy()
            new_obs.update(observation)
        elif isinstance(context, Dict):
            new_obs = context.copy()
            new_obs["observation"] = observation
        elif isinstance(observation, Dict):
            new_obs = observation.copy()
            new_obs["context"] = context
        else:
            new_obs = {
                "context": context,
                "observation": observation
            }
        return new_obs

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            if self.env.render_fps is not None:
                metadata = copy.copy(self.env.metadata)
                metadata["render_fps"] = self.env.render_fps
                self._metadata = metadata
                return metadata
            else:
                return self.env.metadata
        else:
            return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value
    
    @property
    def render_mode(self) -> Optional[str]:
        return self.env.render_mode

    @render_mode.setter
    def render_mode(self, value: Optional[str]):
        self.env.render_mode = value
    
    def _init_np_rng(self):
        if self._np_rng is None:
            self._np_rng = np.random.default_rng(
                seed_util.next_seed_rng(self.env.rng, self.env.backend)
            )

    @property
    def np_random(self) -> np.random.Generator:
        self._init_np_rng()
        return self._np_rng

    def step(self, action: ActType) -> Tuple[
        ObsType, 
        SupportsFloat, 
        bool, 
        bool,
        Dict[str, Any]
    ]:
        c_action = gym_utils.from_gym_data(
            self.env.action_space, action
        )
        obs, rew, terminated, truncated, info = self.env.step(c_action)
        c_obs = gym_utils.to_gym_data(
            self._unienv_combined_space, 
            self.combine_context_obs(self._episode_context, obs
        ))
        c_rew = float(rew)
        c_terminated = bool(terminated)
        c_truncated = bool(truncated)
        return c_obs, c_rew, c_terminated, c_truncated, info

    def reset(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        options : Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        assert mask is None, "Batched environment param `mask` is not supported by this wrapper."
        kwargs = kwargs.update(options) if options is not None else kwargs
        context, obs, info = self.env.reset(
            *args, 
            mask=None, 
            seed=seed, 
            **kwargs
        )
        self._episode_context = context
        c_obs = gym_utils.to_gym_data(
            self._unienv_combined_space,
            self.combine_context_obs(context, obs)
        )
        return c_obs, info

    def render(self) -> RenderFrame | None:
        return self.env.render()

    def close(self):
        self.env.close()
    
    def __str__(self):
        return f'{type(self).__name__}<{self.env}>'

class FromGymnasiumEnv(
    Env[np.ndarray, None, ObsType, ActType, RenderFrame, Any, np.dtype, np.random.Generator],
    Generic[ObsType, ActType, RenderFrame]
):
    def __init__(
        self,
        env: gym.Env[Any, Any]
    ):
        self.env = env

        self._metadata : Optional[Dict[str, Any]] = None
        self._render_fps : Optional[int] = None

        self.backend = NumpyComputeBackend
        self.device = None

        self.batch_size = None

        self.action_space = gym_utils.from_gym_space(
            env.action_space,
            self.backend,
        )
        self.observation_space = gym_utils.from_gym_space(
            env.observation_space,
            self.backend,
        )
        self.context_space = None

        self._rng = env.np_random
    
    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            return self.env.metadata
        else:
            return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value
    
    @property
    def render_mode(self) -> Optional[str]:
        return self.env.render_mode
    
    @render_mode.setter
    def render_mode(self, value: Optional[str]):
        self.env.render_mode = value

    @property
    def render_fps(self) -> Optional[int]:
        if self._render_fps is None:
            return self.env.metadata.get("render_fps", None)
        else:
            return self._render_fps

    @render_fps.setter
    def render_fps(self, value: Optional[int]):
        self._render_fps = value
    
    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def step(
        self, action: ActType
    ) -> Tuple[
        ObsType, 
        SupportsFloat, 
        bool, 
        bool, 
        Dict[str, Any]
    ]:
        c_action = gym_utils.to_gym_data(
            self.action_space, action
        )
        obs, rew, terminated, truncated, info = self.env.step(c_action)
        c_obs = gym_utils.from_gym_data(self.observation_space, obs)
        c_rew = rew
        c_terminated = terminated
        c_truncated = truncated
        return c_obs, c_rew, c_terminated, c_truncated, info

    def reset(
        self,
        *args,
        mask : Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[None, ObsType, Dict[str, Any]]:
        assert mask is None, "Batched environment param `mask` is not supported by this wrapper."
        obs, info = self.env.reset(
            *args, 
            seed=seed, 
            **kwargs
        )
        c_obs = gym_utils.from_gym_data(self.observation_space, obs)
        return None, c_obs, info

    def render(self) -> RenderFrame | Sequence[RenderFrame] | None:
        return self.env.render()

    def close(self):
        self.env.close()
    
    def __str__(self):
        return f'{type(self).__name__}<{self.env}>'

    # =========== Wrapper Methods ===========

    # We'll not change unwrapped property as the inside env is not a standard UniEnv Environment

    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or (
            hasattr(self.env, name) or
            (hasattr(self.env, 'has_wrapper_attr' and self.env.has_wrapper_attr(name)))
        )

    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        elif hasattr(self.env, 'get_wrapper_attr'):
            return self.env.get_wrapper_attr(name)
        else:
            return getattr(self.env, name)
    
    def set_wrapper_attr(self, name: str, value: Any):
        if hasattr(self, name):
            setattr(self, name, value)
        elif hasattr(self.env, 'set_wrapper_attr'):
            self.env.set_wrapper_attr(name, value)
        else:
            setattr(self.env, name, value)