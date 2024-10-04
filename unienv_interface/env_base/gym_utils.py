from typing import Dict, Any, Optional, Tuple, Union, Generic, SupportsFloat
import gymnasium as gym
import numpy as np
import copy
from .env import Env, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT
from ..space import Space
from ..space import gym_utils

class ToGymnasiumEnv(
    gym.Env[Any, Any],
    Generic[ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT]
):
    def __init__(
        self,
        env: Env[ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT]
    ):
        self.env = env

        self._metadata : Optional[Dict[str, Any]] = None
        self.action_space = gym_utils.to_gym_space(env.action_space)
        self.observation_space = gym_utils.to_gym_space(env.observation_space)

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
    
    @property
    def np_random(self) -> np.random.Generator:
        return self.env.np_rng

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
        c_obs = gym_utils.to_gym_data(self.env.observation_space, obs)
        c_rew = float(rew)
        c_terminated = bool(terminated)
        c_truncated = bool(truncated)
        return c_obs, c_rew, c_terminated, c_truncated, info

    def reset(
        self,
        *args,
        seed : Optional[int] = None,
        options : Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        kwargs = kwargs.update(options) if options is not None else kwargs
        obs, info = self.env.reset(
            *args, seed=seed, **kwargs
        )
        c_obs = gym_utils.to_gym_data(self.env.observation_space, obs)
        return c_obs, info

    def render(self) -> RenderFrame | None:
        return self.env.render()

    def close(self):
        self.env.close()
    
    def __str__(self):
        return f'{type(self).__name__}<{self.env}>'

