from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal
import numpy as np
from xbarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.space.space_utils import batch_utils as sbu

from unienv_interface.transformations.batch_and_unbatch import BatchifyTransformation, UnBatchifyTransformation
from .transformation import TransformWrapper

class BatchifyWrapper(
    TransformWrapper[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    """
    This wrapper will rescale the action space to a new range.
    """

    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
    ):
        assert env.batch_size is None
        super().__init__(
            env,
            context_transformation=None if env.context_space is None else BatchifyTransformation(env.context_space),
            observation_transformation=BatchifyTransformation(env.observation_space),
            action_transformation=UnBatchifyTransformation(env.action_space),
        )
        print(env.action_space, self.action_space)
    
    @property
    def batch_size(self) -> int:
        return 1
    
    def step(self, action):
        obs, rewards, terminated, truncated, info = super().step(action)
        terminated = self.backend.asarray(
            [terminated], dtype=self.backend.default_boolean_dtype, device=self.device
        )
        truncated = self.backend.asarray(
            [truncated], dtype=self.backend.default_boolean_dtype, device=self.device
        )
        return obs, rewards, terminated, truncated, info

class UnBatchifyWrapper(
    TransformWrapper[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    """
    This wrapper will rescale the action space to a new range.
    """

    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
    ):
        assert env.batch_size == 1, "UnBatchifyWrapper can only be used with envs that have batch_size == 1"
        super().__init__(
            env,
            context_transformation=None if env.context_space is None else UnBatchifyTransformation(env.context_space),
            observation_transformation=UnBatchifyTransformation(env.observation_space),
            action_transformation=UnBatchifyTransformation(env.action_space),
        )

    @property
    def batch_size(self) -> Optional[int]:
        return None
    
    def step(self, action):
        obs, rewards, terminated, truncated, info = super().step(action)
        terminated = bool(terminated)
        truncated = bool(truncated)
        return obs, rewards, terminated, truncated, info