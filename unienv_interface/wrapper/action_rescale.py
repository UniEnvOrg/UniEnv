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
        new_low : Union[Any, float] = -1.0,
        new_high : Union[Any, float] = 1.0,
    ):
        action_transformation = RescaleTransformation(
            env.action_space,
            new_low,
            new_high
        )
        super().__init__(env, action_transformation)

