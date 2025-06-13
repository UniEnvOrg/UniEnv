from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable, Literal
from abc import ABC, abstractmethod

import mujoco.viewer

from unienv_interface.world.sensors.windowed_view import FuncWindowedViewSensor
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import TupleSpace, BoxSpace
from unienv_interface.utils import seed_util
import mujoco
from dm_control import mjcf
from dataclasses import dataclass, replace as dataclass_replace
import numpy as np
from unienv_mujoco import mjcf_util
from unienv_mujoco.base import *
import time

@dataclass(frozen=True)
class MujocoFuncWindowedViewSensorState:
    viewer : mujoco.viewer.Handle

    # Temporarily stored parameters
    render_kwargs : Dict[str, Any]
    scene_option : Optional[mujoco.MjvOption] = None

class MujocoFuncWindowedViewSensor(FuncWindowedViewSensor[MujocoFuncWorldState, MujocoFuncWindowedViewSensorState, Any, np.dtype, np.random.Generator]):
    observation_space = TupleSpace(
        backend=NumpyComputeBackend,
        spaces=(),
        device=None
    )

    def __init__(
        self,
        control_timestep : float,

        # If True, the world is dirty (task or world will re-compile MJCF at reset) 
        # and the window should be re-opened each episode
        is_dirty_world : bool = False
    ):
        assert control_timestep > 0.0
        self.control_timestep = control_timestep
        self.is_dirty_world = is_dirty_world
    
    def initial(
        self,
        world : MujocoFuncWorld,
        state : MujocoFuncWorldState,
        rng : np.random.Generator,
        render_kwargs : Dict[str, Any] = {},
        scene_option : Optional[mujoco.MjvOption] = None
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        MujocoFuncWindowedViewSensorState
    ]:
        passive_viewer = mujoco.viewer.launch_passive(
            model=state.mj_model,
            data=state.data,
            show_left_ui=render_kwargs.pop('show_left_ui', True),
            show_right_ui=render_kwargs.pop('show_right_ui', True),
            **render_kwargs
        )
        mujoco.mjv_defaultFreeCamera(state.mj_model, passive_viewer.cam)
        if scene_option is not None:
            with passive_viewer.lock():
                passive_viewer.opt.flags |= scene_option.flags
                passive_viewer.opt.geomgroup = scene_option.geomgroup
                passive_viewer.opt.sitegroup = scene_option.sitegroup
                passive_viewer.opt.frame = scene_option.frame
        return state, rng, MujocoFuncWindowedViewSensorState(
            viewer=passive_viewer,
            render_kwargs=render_kwargs,
            scene_option=scene_option
        )

    def reset(
        self,
        world : MujocoFuncWorld,
        state : MujocoFuncWorldState,
        rng : np.random.Generator,
        sensor_state : MujocoFuncWindowedViewSensorState
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        MujocoFuncWindowedViewSensorState
    ]:
        if self.is_dirty_world:
            # We may need to close and re-open the viewer here, since mjdata handle might have been re-created
            state, rng = self.close(state, rng, sensor_state)
            time.sleep(0.1)
            state, rng, sensor_state = self.initial(
                world,
                state,
                rng,
                render_kwargs=sensor_state.render_kwargs,
                scene_option=sensor_state.scene_option
            )
        else:
            sensor_state.viewer.sync()
        
        return state, rng, sensor_state

    def step(
        self,
        state : MujocoFuncWorldState,
        rng : np.random.Generator,
        sensor_state : MujocoFuncWindowedViewSensorState,
        last_step_elapsed : float
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        MujocoFuncWindowedViewSensorState
    ]:
        sensor_state.viewer.sync()
        return state, rng, sensor_state
    
    def get_data(
        self, 
        state: MujocoFuncWorldState, 
        rng: np.random.Generator, 
        sensor_state: MujocoFuncWindowedViewSensorState,
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState, 
        np.random.Generator, 
        MujocoFuncWindowedViewSensorState, 
        Tuple
    ]:
        return state, rng, sensor_state, ()
    
    def close(
        self, 
        state: MujocoFuncWorldState, 
        rng: np.random.Generator, 
        sensor_state: MujocoFuncWindowedViewSensorState
    ) -> Tuple[
        MujocoFuncWorldState, 
        np.random.Generator
    ]:
        sensor_state.viewer.close()
        return state, rng
