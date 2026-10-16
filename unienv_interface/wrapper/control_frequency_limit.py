from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal, SupportsFloat
from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.space.space_utils import batch_utils
import logging
import os
import numpy as np
import time

logger = logging.getLogger(__name__)

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
        control_dt : Optional[float] = None,
        late_warning_interval : Optional[float] = 5.0,
    ):
        """
        Args:
            control_dt: Minimum wall-clock period between actions; defaults to ``1 / env.render_fps``.
            late_warning_interval: Late actions are logged (``logging.WARNING``) at most once per this
                many seconds, with a count of the late actions not logged in between.
                ``None`` disables the messages.
        """
        super().__init__(env)
        self.late_warning_interval = late_warning_interval
        self._last_late_warning_time : Optional[float] = None
        self._suppressed_late_count = 0
        self._suppressed_max_lateness = 0.0
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
            self._report_late(-dt_remaining, ctime)
        # The control period starts when the action is actually sent (after sleeping).
        self._last_control_time = time.monotonic()
        
        obs, rew, termination, truncation, info = self.env.step(action)
        return obs, rew, termination, truncation, info

    def _report_late(self, lateness : float, now : float) -> None:
        if self.late_warning_interval is None:
            return
        if (
            self._last_late_warning_time is not None
            and now - self._last_late_warning_time < self.late_warning_interval
        ):
            self._suppressed_late_count += 1
            self._suppressed_max_lateness = max(self._suppressed_max_lateness, lateness)
            return
        if self._suppressed_late_count > 0:
            logger.warning(
                "Action is late by %.3fs (control_dt=%.3fs); %d more late action(s), up to %.3fs late, "
                "in the last %.1fs were not logged",
                lateness, self.control_dt, self._suppressed_late_count, self._suppressed_max_lateness,
                now - self._last_late_warning_time,
            )
        else:
            logger.warning("Action is late by %.3fs (control_dt=%.3fs)", lateness, self.control_dt)
        self._last_late_warning_time = now
        self._suppressed_late_count = 0
        self._suppressed_max_lateness = 0.0

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
    
