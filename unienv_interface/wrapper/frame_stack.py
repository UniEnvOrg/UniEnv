from typing import Dict, Any, Optional, Tuple, Union, SupportsFloat
import numpy as np
import copy

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.wrapper import ContextObservationWrapper, ActionWrapper, WrapperContextT, WrapperObsT, WrapperActT
from unienv_interface.space import Space, DictSpace
from unienv_interface.utils.data_queue import SpaceDataQueue

class FrameStackWrapper(
    ContextObservationWrapper[
        ContextType, Union[Dict[str, Any], Any],
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self, 
        env: Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        obs_stack_size: int = 0,
        action_stack_size: int = 0,
        action_default_value: Optional[ActType] = None,
    ):
        assert obs_stack_size >= 0, "Observation stack size must be greater than 0"
        assert action_stack_size >= 0, "Action stack size must be greater than 0"
        assert action_stack_size == 0 or action_default_value is not None, "Action default value must be provided if action stack size is greater than 0"
        assert obs_stack_size > 0 or action_stack_size > 0, "At least one of observation stack size or action stack size must be greater than 0"
        super().__init__(env)
        obs_is_dict = isinstance(env.observation_space, DictSpace)
        assert obs_is_dict or action_stack_size == 0, "Action stack size must be 0 if observation space is not a DictSpace"
        
        self.action_stack_size = action_stack_size
        self.obs_stack_size = obs_stack_size

        if action_stack_size > 0:
            self.action_deque = SpaceDataQueue(
                env.action_space,
                env.batch_size,
                action_stack_size,
            )
            self.action_default_value = action_default_value
        else:
            self.action_deque = None
        
        self.obs_deque = None
        if obs_stack_size > 0:
            self.obs_deque = SpaceDataQueue(
                env.observation_space,
                env.batch_size,
                obs_stack_size + 1
            )
            if action_stack_size > 0:
                new_obs_space = copy.copy(self.obs_deque.output_space)
                new_obs_space['past_actions'] = self.action_deque.output_space
                self.observation_space = new_obs_space
            else:
                self.observation_space = self.obs_deque.output_space
        else:
            if action_stack_size > 0:
                new_obs_space = copy.copy(env.observation_space)
                new_obs_space['past_actions'] = self.action_deque.output_space
                self.observation_space = new_obs_space
            else:
                raise ValueError("At least one of observation stack size or action stack size must be greater than 0")

    def reverse_map_context(self, context: ContextType) -> ContextType:
        return context

    def map_observation(self, observation: ObsType) -> Union[Dict[str, Any], Any]:
        if self.obs_deque is not None:
            observation = self.obs_deque.get_output_data()
        
        if self.action_deque is not None:
            stacked_action = self.action_deque.get_output_data()
            observation['past_actions'] = stacked_action
        return observation
    
    def reverse_map_observation(self, observation: Union[Dict[str, Any], Any]) -> ObsType:
        if isinstance(observation, dict):
            stacked_obs = observation.copy()
            stacked_obs.pop('past_actions', None)
        else:
            stacked_obs = observation
        
        if self.obs_deque is not None:
            obs_last = sbu.get_at(
                self.obs_deque.output_space,
                stacked_obs,
                -1
            ) if self.env.batch_size is None else sbu.get_at(
                self.obs_deque.output_space,
                stacked_obs,
                (slice(None), -1)
            )
            return obs_last
        else:
            return stacked_obs

    def reset(
        self,
        *args,
        mask: Optional[BArrayType] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, Union[Dict[str, Any], Any], Dict[str, Any]]:
        # TODO: If a mask is provided, we should only reset the stack for the masked indices
        context, obs, info = self.env.reset(
            *args,
            mask=mask,
            seed=seed,
            **kwargs
        )

        if self.action_deque is not None:
            self.action_deque.reset(
                initial_data=sbu.get_at( # Add a singleton batch dimension to the action
                    self.env.action_space,
                    self.action_default_value,
                    None
                ) if self.env.batch_size is not None else self.action_default_value,
                mask=mask
            )
        if self.obs_deque is not None:
            self.obs_deque.reset(
                initial_data=obs,
                mask=mask
            )
        
        return context, self.map_observation(obs), info
    
    def step(
        self,
        action: ActType
    ) -> Tuple[
        Union[Dict[str, Any], Any],
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
        Dict[str, Any]
    ]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        if self.action_deque is not None:
            self.action_deque.add(action)
        if self.obs_deque is not None:
            self.obs_deque.add(obs)
        return self.map_observation(obs), rew, terminated, truncated, info