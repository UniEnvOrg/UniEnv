from typing import Dict as DictT, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence
import gymnasium as gym
import numpy as np
import copy
from unienv_interface.utils import seed_util
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.wrapper import ContextObservationWrapper, ActionWrapper, WrapperContextT, WrapperObsT, WrapperActT
from unienv_interface.backends import ComputeBackend
from unienv_interface.space import Space, Dict, batch_utils as sbu
from collections import deque

class FrameStackWrapper(
    ContextObservationWrapper[
        ContextType, Union[DictT[str, Any], Any],
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self, 
        env: Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        obs_stack_size: int = 0,
        action_stack_size: int = 0,
    ):
        assert obs_stack_size >= 0, "Observation stack size must be greater than 0"
        assert action_stack_size >= 0, "Action stack size must be greater than 0"
        assert obs_stack_size > 0 or action_stack_size > 0, "At least one of observation stack size or action stack size must be greater than 0"
        super().__init__(env)
        obs_is_dict = isinstance(env.observation_space, Dict)
        assert obs_is_dict or action_stack_size == 0, "Action stack size must be 0 if observation space is not a Dict"
        
        self.action_stack_size = action_stack_size
        self.obs_stack_size = obs_stack_size

        if action_stack_size > 0:
            self.action_deque = deque(maxlen=action_stack_size)
            self.action_space_stacked = sbu.batch_space(env.action_space, action_stack_size)
            if env.batch_size is not None:
                self.action_space_stacked = sbu.swap_batch_dims(
                    self.action_space_stacked,
                    0,
                    1
                )
        else:
            self.action_deque = None
            self.action_space_stacked = None
        
        self.obs_deque = None
        if obs_stack_size > 0:
            self.obs_deque = deque(maxlen=obs_stack_size)
            self.obs_space_stacked = sbu.batch_space(env.observation_space, obs_stack_size + 1)
            if env.batch_size is not None:
                self.obs_space_stacked = sbu.swap_batch_dims(
                    self.obs_space_stacked,
                    0,
                    1
                )
            if action_stack_size > 0:
                new_obs_spaces = self.obs_space_stacked.spaces.copy()
                new_obs_spaces['past_actions'] = self.action_space_stacked
                self.observation_space = Dict(
                    env.backend,
                    new_obs_spaces,
                    device=env.observation_space.device
                )
            else:
                self.observation_space = self.obs_space_stacked
        else:
            self.obs_space_stacked = None
            if action_stack_size > 0:
                spaces = env.observation_space.spaces.copy()
                spaces['past_actions'] = self.action_space_stacked
                self.observation_space = Dict(
                    env.backend,
                    spaces,
                    device=env.observation_space.device
                )
            self.obs_deque = None
            

    def reverse_map_context(self, context: ContextType) -> ContextType:
        return context

    def stack_deque(self, space : Space[Any, Any, BDeviceType, BDtypeType, BRNGType], hist_store: deque, data: Optional[Any], num_stack : int) -> Any:
        remaining_num = num_stack - len(hist_store)
        if remaining_num > 0:
            to_stack = [data] * remaining_num if len(hist_store) == 0 else [hist_store[0]] * remaining_num
        else:
            to_stack = []

        to_stack.extend(hist_store)
        if data is not None:
            to_stack.append(data)
        return sbu.concatenate(
            space,
            to_stack
        )

    def map_observation(self, observation: ObsType) -> Union[DictT[str, Any], Any]:
        if self.obs_deque is not None:
            stacked_obs = self.stack_deque(
                self.obs_space_stacked,
                self.obs_deque,
                observation,
                self.obs_stack_size
            )
            if self.env.batch_size is not None:
                stacked_obs = sbu.swap_batch_dims_in_data(
                    self.obs_space_stacked,
                    stacked_obs,
                    0,
                    1
                ) # (obs_stack_size, B, ...) => (B, obs_stack_size, ...)
        else:
            stacked_obs = observation
        if self.action_deque is not None:
            stacked_action = self.stack_deque(
                self.action_space_stacked,
                self.action_deque,
                None,
                self.action_stack_size
            )
            if self.env.batch_size is not None:
                stacked_action = sbu.swap_batch_dims_in_data(
                    self.action_space_stacked,
                    stacked_action,
                    0,
                    1
                ) # (action_stack_size, B, ...) => (B, action_stack_size, ...)
            stacked_obs['past_actions'] = stacked_action
        return stacked_obs
    
    def reverse_map_observation(self, observation: Union[DictT[str, Any], Any]) -> ObsType:
        if isinstance(observation, dict):
            stacked_obs = observation.copy()
            stacked_obs.pop('past_actions', None)
        else:
            stacked_obs = observation
        if self.obs_space_stacked is None:
            return stacked_obs
        
        if self.env.batch_size is not None:
            stacked_obs = sbu.swap_batch_dims_in_data(
                self.obs_space_stacked,
                stacked_obs,
                0,
                1
            ) # (B, obs_stack_size, ...) => (obs_stack_size, B, ...)
        
        mask = self.backend.array_api_namespace.zeros(
            self.obs_stack_size + 1,
            dtype=self.backend.default_boolean_dtype,
            device=self.obs_space_stacked.device
        )
        mask[-1] = True
        obs_last_stacked = sbu.read_batched_data_with_mask(
            self.obs_space_stacked,
            stacked_obs,
            mask
        )
        obs_iter = sbu.iterate(
            self.obs_space_stacked,
            obs_last_stacked
        )
        return next(obs_iter)
