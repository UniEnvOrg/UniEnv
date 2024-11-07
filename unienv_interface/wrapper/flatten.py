from typing import Dict, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence
import gymnasium as gym
import numpy as np
import copy
from unienv_interface.utils import seed_util
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT
from unienv_interface.env_base.wrapper import ContextObservationWrapper, ActionWrapper, WrapperContextType, WrapperObsType, WrapperActType, WrapperRewardType, WrapperTerminationType, WrapperRenderFrame, WrapperBDeviceT, WrapperBRngT
from unienv_interface.backends import ComputeBackend
from unienv_interface.space import Space, flatten_utils as space_flatten_utils
import array_api_compat

class FlattenActionWrapper(
    ActionWrapper[
        Any,
        ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT
    ]
):
    def __init__(
        self, 
        env: Env[ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT]
    ):
        super().__init__(env)
        assert space_flatten_utils.is_flattenable(env.action_space)
        self.action_space = space_flatten_utils.flatten_space(env.action_space)

    def map_action(self, action: Any) -> ActType:
        return space_flatten_utils.unflatten_data(
            self.env.action_space,
            action
        )
    
    def reverse_map_action(self, action: ActType) -> Any:
        return space_flatten_utils.flatten_data(
            self.env.action_space,
            action
        )
    
class FlattenContextObservationWrapper(
    ContextObservationWrapper[
        Any, Any,
        ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT
    ]
):
    def __init__(
        self, 
        env: Env[ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT],
        flatten_context: bool = False,
        flatten_observation: bool = True
    ):
        super().__init__(env)
        if flatten_context:
            assert space_flatten_utils.is_flattenable(env.context_space)
            self.context_space = space_flatten_utils.flatten_space(env.context_space)
        if flatten_observation:
            assert space_flatten_utils.is_flattenable(env.observation_space)
            self.observation_space = space_flatten_utils.flatten_space(env.observation_space)
        self.flatten_context = flatten_context
        self.flatten_observation = flatten_observation

    def map_context(self, context: Any) -> ContextType:
        if self.flatten_context:
            return space_flatten_utils.unflatten_data(
                self.env.context_space,
                context
            )
        return context
    
    def reverse_map_context(self, context: ContextType) -> Any:
        if self.flatten_context:
            return space_flatten_utils.flatten_data(
                self.env.context_space,
                context
            )
        return context

    def map_observation(self, observation: Any) -> ObsType:
        if self.flatten_observation:
            return space_flatten_utils.unflatten_data(
                self.env.observation_space,
                observation
            )
        return observation
    
    def reverse_map_observation(self, observation: ObsType) -> Any:
        if self.flatten_observation:
            return space_flatten_utils.flatten_data(
                self.env.observation_space,
                observation
            )
        return observation