from typing import Dict as DictT, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence, TypeVar
import numpy as np
import copy
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.utils import seed_util
from unienv_interface.env_base.funcenv import FuncEnv, ContextType, ObsType, ActType, RenderFrame, StateType, RenderStateType
from unienv_interface.env_base.funcenv_wrapper import *
from unienv_interface.space import Space
from unienv_interface.transformations.transformation import DataTransformation

class FuncTransformWrapper(
    FuncEnvWrapper[
        StateType, RenderStateType,
        BArrayType, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
        StateType, RenderStateType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        func_env : FuncEnv[
            StateType, RenderStateType,
            BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
        ],
        context_transformation : Optional[DataTransformation] = None,
        observation_transformation : Optional[DataTransformation] = None,
        action_transformation : Optional[DataTransformation] = None,
        target_action_space : Optional[Space[BArrayType, BDeviceType, BDtypeType, BRNGType]] = None
    ):
        super().__init__(func_env)
        self.context_transformation = context_transformation
        self.observation_transformation = observation_transformation
        self.set_action_transformation(action_transformation, target_action_space)

    @property
    def context_transformation(self) -> Optional[DataTransformation]:
        return self._context_transformation
    
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
            assert origin_space == self.func_env.action_space, "action_transformation must be compatible with the env.action_space"
        self._action_transformation = action_transformation
        self._action_space = target_action_space
    
    @property
    def action_transformation(self) -> Optional[DataTransformation]:
        return self._action_transformation
    
    def initial(
        self, 
        *, 
        seed : Optional[int] = None, 
        **kwargs
    ) -> Tuple[
        StateType,
        WrapperContextT,
        WrapperObsT,
        Dict[str, Any]
    ]:
        state, context, obs, info = self.func_env.initial(seed=seed, **kwargs)
        context = self.context_transformation.transform(self.func_env.context_space, context) if self.context_transformation is not None else context
        obs = self.observation_transformation.transform(self.func_env.observation_space, obs) if self.observation_transformation is not None else obs
        return state, context, obs, info

    def reset(
        self,
        state : StateType,
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> Tuple[
        StateType,
        WrapperContextT,
        WrapperObsT,
        Dict[str, Any]
    ]:
        state, context, obs, info = self.func_env.reset(state, seed=seed, mask=mask, **kwargs)
        context = self.context_transformation.transform(self.func_env.context_space, context) if self.context_transformation is not None else context
        obs = self.observation_transformation.transform(self.func_env.observation_space, obs) if self.observation_transformation is not None else obs
        return state, context, obs, info

    def step(
        self,
        state : StateType,
        action : WrapperActT
    ) -> Tuple[
        StateType,
        WrapperObsT,
        Union[SupportsFloat, WrapperBArrayT],
        Union[bool, WrapperBArrayT],
        Union[bool, WrapperBArrayT],
        Dict[str, Any]
    ]:
        if self.action_transformation is not None:
            action = self.action_transformation.transform(self.action_space, action)
        state, obs, rew, terminated, truncated, info = self.func_env.step(state, action)
        obs = self.observation_transformation.transform(self.func_env.observation_space, obs) if self.observation_transformation is not None else obs
        return state, obs, rew, terminated, truncated, info