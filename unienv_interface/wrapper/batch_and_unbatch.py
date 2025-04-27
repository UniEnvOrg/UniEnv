from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal
import numpy as np
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame

from unienv_interface.transformations.batch_and_unbatch import BachifyTransformation, UnBachifyTransformation
from .transformation import TransformWrapper


class BachifyWrapper(
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
        super().__init__(
            env,
            context_transformation=None if env.context_space is None else BachifyTransformation(env.context_space),
            observation_transformation=BachifyTransformation(env.observation_space),
            action_transformation=BachifyTransformation(env.action_space),
        )
    
class UnBachifyWrapper(
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
        super().__init__(
            env,
            context_transformation=None if env.context_space is None else UnBachifyTransformation(env.context_space),
            observation_transformation=UnBachifyTransformation(env.observation_space),
            action_transformation=UnBachifyTransformation(env.action_space),
        )

