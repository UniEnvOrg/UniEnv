from typing import Dict as DictT, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence, TypeVar
import numpy as np
import copy
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.utils import seed_util
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.env_base.wrapper import *
from unienv_interface.space import Space
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
        context_transformation : Optional[DataTransformation] = None,
        observation_transformation : Optional[DataTransformation] = None,
        action_transformation : Optional[DataTransformation] = None,
        target_action_space : Optional[Space[BArrayType, BDeviceType, BDtypeType, BRNGType]] = None,
    ):
        super().__init__(env)
        self.context_transformation = context_transformation
        self.observation_transformation = observation_transformation
        self.set_action_transformation(action_transformation, target_action_space)
        
    @property
    def context_transformation(self) -> Optional[DataTransformation]:
        return self._context_transformation
    
    @context_transformation.setter
    def context_transformation(self, value: Optional[DataTransformation]):
        if value is not None:
            assert self.env.context_space is not None, "env.context_space must be defined if context_transformation is provided"
        self._context_space = None if value is None else value.get_target_space_from_source(self.env.context_space)
        self._context_transformation = value
    
    @property
    def observation_transformation(self) -> Optional[DataTransformation]:
        return self._observation_transformation

    @observation_transformation.setter
    def observation_transformation(self, value: Optional[DataTransformation]):
        self._observation_space = None if value is None else value.get_target_space_from_source(self.env.observation_space)
        self._observation_transformation = value
    
    def set_action_transformation(
        self,
        action_transformation: Optional[DataTransformation] = None,
        target_action_space: Optional[Space[BArrayType, BDeviceType, BDtypeType, BRNGType]] = None
    ) -> None:
        assert (action_transformation is None) == (target_action_space is None), "action_transformation or target_action_space must be None or both must be provided"
        if action_transformation is not None:
            origin_space = action_transformation.get_target_space_from_source(target_action_space)
            assert origin_space == self.env.action_space, "action_transformation must be compatible with the env.action_space"
        self._action_transformation = action_transformation
        self._action_space = target_action_space

    @property
    def action_transformation(self) -> Optional[DataTransformation]:
        return self._action_transformation

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
        action = self.action_transformation.transform(self.action_space, action) if self.action_transformation is not None else action
        obs, rew, termination, truncation, info = self.env.step(action)
        transformed_obs = self.observation_transformation.transform(self.env.observation_space, obs) if self.observation_transformation is not None else obs
        return transformed_obs, rew, termination, truncation, info
    
    def reset(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, ObsType, DictT[str, Any]]:
        context, obs, info = self.env.reset(*args, mask=mask, seed=seed, **kwargs)
        transformed_context = self.context_transformation.transform(self.env.context_space, context) if self.context_transformation is not None else context
        transformed_obs = self.observation_transformation.transform(self.env.observation_space, obs) if self.observation_transformation is not None else obs
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
        context_transformation : Optional[DataTransformation] = None,
        observation_transformation : Optional[DataTransformation] = None,
    ):
        super().__init__(env)
        assert context_transformation is not None or observation_transformation is not None, "At least one of context_transformation or observation_transformation must be provided"
        self.context_transformation = context_transformation
        self.observation_transformation = observation_transformation
    
    @property
    def context_transformation(self) -> Optional[DataTransformation]:
        return self._context_transformation
    
    @context_transformation.setter
    def context_transformation(self, value: Optional[DataTransformation]):
        if value is not None:
            assert self.env.context_space is not None, "env.context_space must be defined if context_transformation is provided"
        self._context_space = None if value is None else value.get_target_space_from_source(self.env.context_space)
        self._context_transformation = value
        self._context_transformation_inv = None if value is None or not value.has_inverse else value.direction_inverse(self.env.context_space)
    
    @property
    def observation_transformation(self) -> Optional[DataTransformation]:
        return self._observation_transformation

    @observation_transformation.setter
    def observation_transformation(self, value: Optional[DataTransformation]):
        self._observation_space = None if value is None else value.get_target_space_from_source(self.env.observation_space)
        self._observation_transformation = value
        self._observation_transformation_inv = None if value is None or not value.has_inverse else value.direction_inverse(self.env.observation_space)

    def map_context(self, context : ContextType) -> WrapperContextT:
        return context if self.context_transformation is None else self.context_transformation.transform(self.env.context_space, context)
    
    def reverse_map_context(self, context : WrapperContextT) -> ContextType:
        if self._context_transformation is not None and self._context_transformation_inv is None:
            raise NotImplementedError
        return context if self._context_transformation_inv is None else self._context_transformation_inv.transform(self.context_space, context)
    
    def map_observation(self, observation : ObsType) -> WrapperObsT:
        return observation if self.observation_transformation is None else self.observation_transformation.transform(self.env.observation_space, observation)
    
    def reverse_map_observation(self, observation : WrapperObsT) -> ObsType:
        if self._observation_transformation is not None and self._observation_transformation_inv is None:
            raise NotImplementedError
        return observation if self._observation_transformation_inv is None else self._observation_transformation_inv.transform(self.observation_space, observation)

class ActionTransformWrapper(
    ActionWrapper[
        WrapperActT,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        action_transformation : DataTransformation,
        target_action_space : Space[BArrayType, BDeviceType, BDtypeType, BRNGType]
    ):
        super().__init__(env)
        self._action_transformation = action_transformation
    
    @property
    def action_transformation(self) -> DataTransformation:
        return self._action_transformation
    
    def set_action_transformation(
        self,
        action_transformation: DataTransformation,
        target_action_space: Space[BArrayType, BDeviceType, BDtypeType, BRNGType] = None
    ) -> None:
        origin_space = action_transformation.get_target_space_from_source(target_action_space)
        assert origin_space == self.env.action_space, "action_transformation must be compatible with the env.action_space"
        self._action_transformation = action_transformation
        self._action_space = target_action_space
        self._action_transformation_inv = None if not action_transformation.has_inverse else action_transformation.direction_inverse(target_action_space)

    def map_action(self, action : WrapperActT) -> ActType:
        return self.action_transformation.transform(self.action_space, action)
    
    def reverse_map_action(self, action : ActType) -> WrapperActT:
        if self._action_transformation_inv is None:
            raise NotImplementedError("Inverse transformation is not implemented for the action transformation")
        return self._action_transformation_inv.transform(self.env.action_space, action)