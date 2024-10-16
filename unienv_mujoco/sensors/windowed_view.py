from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable, Literal
from abc import ABC, abstractmethod

import mujoco.viewer

from unienv_interface.env_base.funcenv import FuncEnvCommonState
from unienv_interface.world.sensors.windowed_view import FuncWindowedViewSensor
from unienv_interface.world.sensor import FuncSensorSingleState
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import Tuple as TupleSpace, Box
import mujoco
from dm_control import mjcf
from dataclasses import dataclass, replace as dataclass_replace
import numpy as np
from .. import mjcf_util
from ..base.world import MujocoFuncWorldState, MujocoFuncWorld

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
        seed : Optional[int] = None,
    ):
        assert control_timestep > 0.0
        self.control_timestep = control_timestep
    
    def initial(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        render_kwargs : Dict[str, Any],
        seed : int,
        scene_option : Optional[mujoco.MjvOption] = None
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        FuncSensorSingleState[MujocoFuncWindowedViewSensorState]
    ]:
        passive_viewer = mujoco.viewer.launch_passive(
            model=state.mj_model,
            data=state.data,
            show_left_ui=render_kwargs.pop('show_left_ui', True),
            show_right_ui=render_kwargs.pop('show_right_ui', True),
            **render_kwargs
        )
        mujoco.mjv_defaultFreeCamera(state.mj_model, passive_viewer.cam)
        with passive_viewer.lock():
            passive_viewer.opt.flags |= scene_option.flags
            passive_viewer.opt.geomgroup = scene_option.geomgroup
            passive_viewer.opt.sitegroup = scene_option.sitegroup
            passive_viewer.opt.frame = scene_option.frame
        return state, common_state, FuncSensorSingleState(
            sensor_state=MujocoFuncWindowedViewSensorState(
                viewer=passive_viewer,
                render_kwargs=render_kwargs,
                scene_option=scene_option
            ),
            remaining_time_until_read=0.0,
        )

    def reset(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        sensor_single_state : FuncSensorSingleState[MujocoFuncWindowedViewSensorState]
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        FuncSensorSingleState[MujocoFuncWindowedViewSensorState]
    ]:
        # We may need to close and re-open the viewer here, since mjdata handle might have been re-created
        state, common_state = self.close(state, common_state, sensor_single_state)
        state, common_state, sensor_single_state = self.initial(
            state=state,
            common_state=common_state,
            render_kwargs=sensor_single_state.sensor_state.render_kwargs,
            seed=common_state.np_rng.integers(0),
            scene_option=sensor_single_state.sensor_state.scene_option
        )
        return state, common_state, sensor_single_state

    def step(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        sensor_single_state : FuncSensorSingleState[MujocoFuncWindowedViewSensorState],
        last_step_elapsed : float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        FuncSensorSingleState[MujocoFuncWindowedViewSensorState]
    ]:
        sensor_single_state.sensor_state.viewer.sync()
        return state, common_state, dataclass_replace(
            sensor_single_state,
            remaining_time_until_read=sensor_single_state.remaining_time_until_read - last_step_elapsed
        )
    
    def get_data(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        sensor_single_state: FuncSensorSingleState[MujocoFuncWindowedViewSensorState]
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator], 
        FuncSensorSingleState[MujocoFuncWindowedViewSensorState], 
        Tuple
    ]:
        return state, common_state, sensor_single_state, ()
    
    def close(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        sensor_single_state: FuncSensorSingleState[MujocoFuncWindowedViewSensorState]
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator]
    ]:
        sensor_single_state.sensor_state.viewer.close()
        return state, common_state
