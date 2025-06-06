from typing import Dict as DictT, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence, TypeVar
import numpy as np
import copy
from xbarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.utils import seed_util
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.env_base.wrapper import *
from unienv_interface.space import Space, Dict
from unienv_interface.transformations.transformation import DataTransformation
from collections import deque

class TransformWrapper(
    Wrapper[
        WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        context_transformation : Optional[DataTransformation[
            WrapperContextT, WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
            ContextType, BArrayType, BDeviceType, BDtypeType, BRNGType
        ]] = None,
        observation_transformation : Optional[DataTransformation[
            WrapperObsT, WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
            ObsType, BArrayType, BDeviceType, BDtypeType, BRNGType
        ]] = None,
        action_transformation : Optional[DataTransformation[
            WrapperActT, WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
            ActType, BArrayType, BDeviceType, BDtypeType, BRNGType
        ]] = None,
    ):
        super().__init__(env)
        assert context_transformation is not None or observation_transformation is not None or action_transformation is not None, "At least one of context_transformation, observation_transformation or action_transformation must be provided"
        self.context_transformation = context_transformation
        self.observation_transformation = observation_transformation
        self.action_transformation = action_transformation
        self._context_space = env.context_space if context_transformation is None else context_transformation.target_space
        self._observation_space = env.observation_space if observation_transformation is None else observation_transformation.target_space
        self._action_space = env.action_space if action_transformation is None else action_transformation.source_space

    def step(
        self,
        action: ActType
    ) -> Tuple[
        ObsType,
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
        DictT[str, Any]
    ]:
        action = self.action_transformation.transform(action) if self.action_transformation is not None else action
        obs, rew, termination, truncation, info = self.env.step(action)
        transformed_obs = self.observation_transformation.transform(obs) if self.observation_transformation is not None else obs
        return transformed_obs, rew, termination, truncation, info
    
    def reset(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, ObsType, DictT[str, Any]]:
        context, obs, info = self.env.reset(*args, mask=mask, seed=seed, **kwargs)
        transformed_context = self.context_transformation.transform(context) if self.context_transformation is not None else context
        transformed_obs = self.observation_transformation.transform(obs) if self.observation_transformation is not None else obs
        return transformed_context, transformed_obs, info
    

class ContextObservationTransformWrapper(
    ContextObservationWrapper[
        WrapperContextT, WrapperObsT,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        context_transformation : Optional[DataTransformation[
            WrapperContextT, BArrayType, BDeviceType, BDtypeType, BRNGType,
            ContextType, BArrayType, BDeviceType, BDtypeType, BRNGType
        ]] = None,
        observation_transformation : Optional[DataTransformation[
            WrapperObsT, BArrayType, BDeviceType, BDtypeType, BRNGType,
            ObsType, BArrayType, BDeviceType, BDtypeType, BRNGType
        ]] = None,
    ):
        super().__init__(env)
        assert context_transformation is not None or observation_transformation is not None, "At least one of context_transformation or observation_transformation must be provided"
        self.context_transformation = context_transformation
        self.observation_transformation = observation_transformation
    
    def map_context(self, context : ContextType) -> WrapperContextT:
        return context if self.context_transformation is None else self.context_transformation.transform(context)
    
    def reverse_map_context(self, context : WrapperContextT) -> ContextType:
        return context if self.context_transformation is None else self.context_transformation.inverse_transform(context)
    
    def map_observation(self, observation : ObsType) -> WrapperObsT:
        return observation if self.observation_transformation is None else self.observation_transformation.transform(observation)
    
    def reverse_map_observation(self, observation : WrapperObsT) -> ObsType:
        return observation if self.observation_transformation is None else self.observation_transformation.inverse_transform(observation)

class ActionTransformWrapper(
    ActionWrapper[
        WrapperActT,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        action_transformation : DataTransformation[
            WrapperActT, BArrayType, BDeviceType, BDtypeType, BRNGType,
            ActType, BArrayType, BDeviceType, BDtypeType, BRNGType
        ],
    ):
        super().__init__(env)
        self.action_transformation = action_transformation
    
    def map_action(self, action : ActType) -> WrapperActT:
        return self.action_transformation.transform(action)
    
    def reverse_map_action(self, action : WrapperActT) -> ActType:
        return self.action_transformation.inverse_transform(action)