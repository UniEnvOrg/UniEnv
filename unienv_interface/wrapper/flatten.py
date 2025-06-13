from typing import Dict, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence
import gymnasium as gym
import numpy as np
import copy
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space.space_utils import flatten_utils as space_flatten_utils
from unienv_interface.utils import seed_util
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.env_base.wrapper import ContextObservationWrapper, ActionWrapper, WrapperContextT, WrapperObsT, WrapperActT
from unienv_interface.space import Space

class FlattenActionWrapper(
    ActionWrapper[
        BArrayType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self, 
        env: Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]
    ):
        super().__init__(env)
        if env.batch_size is None:
            assert space_flatten_utils.is_flattenable(env.action_space)
            self.action_space = space_flatten_utils.flatten_space(env.action_space)
        else:
            assert space_flatten_utils.is_flattenable(env.action_space, start_dim=1)
            self.action_space = space_flatten_utils.flatten_space(env.action_space, start_dim=1)

    def map_action(self, action: Any) -> ActType:
        if self.env.batch_size is None:
            return space_flatten_utils.unflatten_data(
                self.env.action_space,
                action
            )
        else:
            return space_flatten_utils.unflatten_data(
                self.env.action_space,
                action,
                start_dim=1
            )
    
    def reverse_map_action(self, action: ActType) -> BArrayType:
        if self.env.batch_size is None:
            return space_flatten_utils.flatten_data(
                self.env.action_space,
                action
            )
        else:
            return space_flatten_utils.flatten_data(
                self.env.action_space,
                action,
                start_dim=1
            )
    
class FlattenContextObservationWrapper(
    ContextObservationWrapper[
        Union[Any, BArrayType], Union[Any, BArrayType],
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self, 
        env: Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        flatten_context: bool = False,
        flatten_observation: bool = True
    ):
        super().__init__(env)
        if flatten_context:
            if env.batch_size is None:
                assert space_flatten_utils.is_flattenable(env.context_space)
                self.context_space = space_flatten_utils.flatten_space(env.context_space)
            else:
                assert space_flatten_utils.is_flattenable(env.context_space, start_dim=1)
                self.context_space = space_flatten_utils.flatten_space(env.context_space, start_dim=1)
        if flatten_observation:
            if env.batch_size is None:
                assert space_flatten_utils.is_flattenable(env.observation_space)
                self.observation_space = space_flatten_utils.flatten_space(env.observation_space)
            else:
                assert space_flatten_utils.is_flattenable(env.observation_space, start_dim=1)
                self.observation_space = space_flatten_utils.flatten_space(env.observation_space, start_dim=1)
        self.flatten_context = flatten_context
        self.flatten_observation = flatten_observation

    def map_context(self, context: Any) -> ContextType:
        if self.flatten_context:
            if self.env.batch_size is None:
                return space_flatten_utils.unflatten_data(
                    self.env.context_space,
                    context
                )
            else:
                return space_flatten_utils.unflatten_data(
                    self.env.context_space,
                    context,
                    start_dim=1
                )
        return context
    
    def reverse_map_context(self, context: ContextType) -> Union[Any, BArrayType]:
        if self.flatten_context:
            if self.env.batch_size is None:
                return space_flatten_utils.flatten_data(
                    self.env.context_space,
                    context
                )
            else:
                return space_flatten_utils.flatten_data(
                    self.env.context_space,
                    context,
                    start_dim=1
                )
        return context

    def map_observation(self, observation: Any) -> ObsType:
        if self.flatten_observation:
            if self.env.batch_size is None:
                return space_flatten_utils.unflatten_data(
                    self.env.observation_space,
                    observation
                )
            else:
                return space_flatten_utils.unflatten_data(
                    self.env.observation_space,
                    observation,
                    start_dim=1
                )
        return observation
    
    def reverse_map_observation(self, observation: ObsType) -> Union[Any, BArrayType]:
        if self.flatten_observation:
            if self.env.batch_size is None:
                return space_flatten_utils.flatten_data(
                    self.env.observation_space,
                    observation
                )
            else:
                return space_flatten_utils.flatten_data(
                    self.env.observation_space,
                    observation,
                    start_dim=1
                )
        return observation