from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal
from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.backends.base import ComputeBackend, BDtypeType, BRNGType, BDeviceType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame
from unienv_interface.space import batch_utils
import os
import numpy as np

"""
This wrapper will truncate the episode after a certain number of steps.
"""
class TimeLimitWrapper(
    Wrapper[
        ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceType, BRNGType,
        ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceType, BRNGType],
        time_limit : int,
        truncation_value : Any = True,
    ):
        super().__init__(env)
        assert env.batch_size is None, "TimeLimitWrapper does not support batched environments"
        self._episode_time = 0
        self.time_limit = time_limit
        self.truncation_value = truncation_value

    def step(
        self, action: ActType
    ) -> Tuple[
        ObsType, 
        RewardType, 
        TerminationType, 
        TerminationType, 
        Dict[str, Any]
    ]:
        obs, rew, termination, truncation, info = self.env.step(action)
        self._episode_time += 1
        if self._episode_time >= self.time_limit:
            truncation = self.truncation_value
        return obs, rew, termination, truncation, info

    def reset(
        self,
        *args,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        ret = self.env.reset(*args, seed=seed, **kwargs)
        self._episode_time = 0
        return ret
    
