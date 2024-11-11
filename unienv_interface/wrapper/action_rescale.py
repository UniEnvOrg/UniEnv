from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal
from unienv_interface.env_base.wrapper import ActionWrapper
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.space import batch_utils, Box
from unienv_interface.utils import seed_util
import os
import numpy as np
import array_api_compat

"""
This wrapper will rescale the action space to a new range.
"""
class ActionRescaleWrapper(
    ActionWrapper[
        ActType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        new_low : Union[Any, float] = -1.0,
        new_high : Union[Any, float] = 1.0,
    ):
        super().__init__(env)
        assert isinstance(env.action_space, Box), "ActionRescaleWrapper only supports Box action spaces"
        assert env.backend.dtype_is_real_floating(env.action_space.dtype), "ActionRescaleWrapper only supports real-valued floating action spaces"
        assert env.action_space.is_bounded('both'), "ActionRescaleWrapper only supports bounded action spaces"
        # new_low = new_low if isinstance(new_low, float) else array_api_compat.to_device(new_low, env.device)
        # new_high = new_high if isinstance(new_high, float) else array_api_compat.to_device(new_high, env.device)
        self.action_space = Box(
            backend=env.backend,
            low=new_low,
            high=new_high,
            dtype=env.action_space.dtype,
            device=env.device,
            shape=env.action_space.shape
        )

    def map_action(self, action: ActType) -> ActType:
        normalized_action = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        rescaled_action = normalized_action * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low
        return rescaled_action
    
    def reverse_map_action(self, action: ActType) -> ActType:
        normalized_action = (action - self.env.action_space.low) / (self.env.action_space.high - self.env.action_space.low)
        rescaled_action = normalized_action * (self.action_space.high - self.action_space.low) + self.action_space.low
        return rescaled_action

