from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal, SupportsFloat
from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.space.space_utils import batch_utils
import os
import numpy as np

"""
This wrapper will truncate the episode after a certain number of steps.
"""
class TimeLimitWrapper(
    Wrapper[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        time_limit : int
    ):
        super().__init__(env)
        if self.env.batch_size is None:
            self._episode_time = 0
        else:
            self._episode_time = self.backend.zeros(
                self.env.batch_size, 
                dtype=self.backend.default_integer_dtype
            )
            if self.device is not None:
                self._episode_time = self.backend.to_device(self._episode_time, self.device)

        self.time_limit = time_limit

    def step(
        self, action: ActType
    ) -> Tuple[
        ObsType, 
        Union[SupportsFloat, BArrayType], 
        Union[bool, BArrayType], 
        Union[bool, BArrayType], 
        Dict[str, Any]
    ]:
        obs, rew, termination, truncation, info = self.env.step(action)
        self._episode_time += 1

        if self.env.batch_size is None:
            if not termination and self._episode_time >= self.time_limit:
                truncation = True
        else:
            exceeds_time_mask = self._episode_time >= self.time_limit
            truncation = self.backend.logical_or(truncation, exceeds_time_mask)
        return obs, rew, termination, truncation, info

    def reset(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        ret = self.env.reset(*args, mask=mask, seed=seed, **kwargs)
        if self.env.batch_size is not None:
            if mask is None:
                self._episode_time = self.backend.zeros(
                    self.env.batch_size, 
                    dtype=self.backend.default_integer_dtype
                )
                if self.device is not None:
                    self._episode_time = self.backend.to_device(self._episode_time, self.device)
            else:
                self._episode_time = self.backend.at(self._episode_time)[mask].set(0)
        else:
            self._episode_time = 0
        return ret
    
