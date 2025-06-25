from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal
import numpy as np
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame

from unienv_interface.transformations.rescale import RescaleTransformation
from .transformation import ActionTransformWrapper

class ActionRescaleWrapper(
    ActionTransformWrapper[
        ActType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    """
    This wrapper will rescale the action space to a new range.
    """

    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        new_low : Union[BArrayType, float] = -1.0,
        new_high : Union[BArrayType, float] = 1.0
    ):
        action_transformation = RescaleTransformation(
            new_low,
            new_high
        )
        target_space = action_transformation.get_target_space_from_source(env.action_space)
        super().__init__(env, action_transformation.direction_inverse(env.action_space), target_space)

