from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal, SupportsFloat
from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.space.space_utils import batch_utils
import os
import numpy as np
import time

"""
This wrapper will limit the control frequency of the environment.
"""
class ControlFrequencyLimitWrapper(
    Wrapper[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        control_dt : Optional[int] = None
    ):
        super().__init__(env)
        if control_dt is not None:
            self.control_dt = control_dt
        else:
            assert env.render_fps is not None, "Control dt is not provided and render fps is not set. Cannot set control dt."
            self.control_dt = 1 / env.render_fps
        self._last_control_time = time.monotonic()

    def step(
        self, action: ActType
    ) -> Tuple[
        ObsType, 
        Union[SupportsFloat, BArrayType], 
        Union[bool, BArrayType], 
        Union[bool, BArrayType], 
        Dict[str, Any]
    ]:
        ctime = time.monotonic()
        dt_remaining = self.control_dt - (ctime - self._last_control_time)
        if dt_remaining > 0:
            time.sleep(dt_remaining)
        else:
            print("Action is late by {:.3f}s".format(-dt_remaining))
        self._last_control_time = ctime
        
        obs, rew, termination, truncation, info = self.env.step(action)
        return obs, rew, termination, truncation, info

    def reset(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        ctime = time.monotonic()
        ret = self.env.reset(*args, mask=mask, seed=seed, **kwargs)
        self._last_control_time = ctime
        return ret
    
