from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable, Literal
from abc import ABC, abstractmethod

import PIL.Image
from unienv_interface.env_base.funcenv import FuncEnvCommonState
from unienv_interface.world.sensors.camera import FuncCameraSensor
from unienv_interface.world.sensor import FuncSensorSingleState
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import Dict as DictSpace, Box
import mujoco
from dm_control import mjcf
from dataclasses import dataclass
import numpy as np
import PIL
from .. import mjcf_util
from ..base.world import MujocoFuncWorldState, MujocoFuncWorld

MujocoFuncEnvSceneCallback = Callable[[mujoco.MjvScene], None]

@dataclass
class MujocoFuncCameraSensorState:
    renderer : mujoco.Renderer
    camera : Union[mujoco.MjvCamera, int, str]
    render_kwargs : Dict[str, Any]
    scene_option : Optional[mujoco.MjvOption] = None
    scene_callback : Optional[MujocoFuncEnvSceneCallback] = None

class MujocoFuncCameraSensor(
    FuncCameraSensor[MujocoFuncWorldState, MujocoFuncCameraSensorState, np.ndarray, Any, np.dtype, np.random.Generator],
):
    def __init__(
        self,
        camera : Union[int, str],
        width : int,
        height : int,
        render_mode : Literal['rgb_array', 'depth_array', 'segmentation_array'],
        control_timestep : float,
        seed : Optional[int] = None,
    ):
        assert control_timestep > 0.0
        self.control_timestep = control_timestep
        
        assert width > 0 and height > 0
        self._render_mode = render_mode
        channels = 3 if render_mode == 'rgb_array' else 1
        if render_mode == 'depth_array':
            observation_dtype = np.float32
            observation_min = 0.0
            observation_max = np.inf
        elif render_mode == 'segmentation_array':
            observation_dtype = int
            observation_min = -1
            observation_max = np.iinfo(int).max
        else:
            observation_dtype = np.uint8
            observation_min = 0
            observation_max = 255
        self.observation_space = Box(
            backend=NumpyComputeBackend,
            low=observation_min,
            high=observation_max,
            dtype=observation_dtype,
            device=None,
            shape=(height, width, channels),
            seed=seed
        )

        self.camera = camera
    
    @property
    def height(self) -> int:
        return self.observation_space.shape[0]

    @property
    def width(self) -> int:
        return self.observation_space.shape[1]

    @property
    def channels(self) -> int:
        return self.observation_space.shape[2]

    @property
    def render_mode(self) -> Literal['rgb_array', 'depth_array', 'segmentation_array']:
        return self._render_mode

    def to_image(self, data : np.ndarray) -> PIL.Image:
        if self.render_mode == 'depth_array':
            min_depth = np.min(data)
            max_depth = np.max(data)
            data = np.clip((data - min_depth) / (max_depth - min_depth), 0.0, 1.0)
            image = PIL.Image.fromarray(
                (data * 255).astype(np.uint8),
                mode='L'
            )
        elif self.render_mode == 'segmentation_array':
            # Since Infinity is mapped to -1, we need to map it to 0
            data = data.astype(np.float64) + 1

            # Normalize the data to [0, 1]
            data = data / np.max(data)
            pixels = (data * 255).astype(np.uint8)
            image = PIL.Image.fromarray(
                pixels, 
                mode='L'
            )
        else:
            image = PIL.Image.fromarray(data, mode='RGB')
        return image

    def initial(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        render_kwargs : Dict[str, Any],
        seed : int,
        max_geom : int = 10_000,
        scene_option : Optional[mujoco.MjvOption] = None,
        scene_callback : Optional[MujocoFuncEnvSceneCallback] = None,
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        FuncSensorSingleState[MujocoFuncCameraSensorState]
    ]:
        renderer = mujoco.Renderer(
            model=state.mj_model,
            height=self.height,
            width=self.width,
            max_geom=max_geom,
        )
        if self.render_mode == 'depth_array':
            renderer.enable_depth_rendering()
        elif self.render_mode == 'segmentation_array':
            renderer.enable_segmentation_rendering()
        else:
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
        
        if not isinstance(self.camera, str) and self.camera < 0:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(state.mj_model, camera)
        elif isinstance(self.camera, int):
            assert self.camera >= 0 and self.camera < state.mj_model.ncam
            camera = self.camera
        else:
            camera = self.camera
        camera_state = MujocoFuncCameraSensorState(
            renderer=renderer,
            camera=camera,
            render_kwargs=render_kwargs,
            scene_option=scene_option,
            scene_callback=scene_callback
        )
        return state, common_state, FuncSensorSingleState(
            sensor_state=camera_state,
            remaining_time_until_read=0.0,
        )

    def reset(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        sensor_single_state : FuncSensorSingleState[MujocoFuncCameraSensorState]
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        FuncSensorSingleState[MujocoFuncCameraSensorState]
    ]:
        return state, common_state, FuncSensorSingleState(
            sensor_state=sensor_single_state.sensor_state,
            remaining_time_until_read=0.0,
        )

    def step(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator],
        sensor_single_state : FuncSensorSingleState[MujocoFuncCameraSensorState],
        last_step_elapsed : float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        FuncSensorSingleState[MujocoFuncCameraSensorState]
    ]:
        return state, common_state, FuncSensorSingleState(
            sensor_state=sensor_single_state.sensor_state,
            remaining_time_until_read=sensor_single_state.remaining_time_until_read - last_step_elapsed,
        )
    
    def get_data(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        sensor_single_state: FuncSensorSingleState[MujocoFuncCameraSensorState]
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator], 
        FuncSensorSingleState[MujocoFuncCameraSensorState], 
        np.ndarray
    ]:
        render_state = sensor_single_state.sensor_state
        render_state.renderer.update_scene(
            data=state.data,
            camera=render_state.camera,
            scene_option=render_state.scene_option
        )
        if render_state.scene_callback is not None:
            render_state.scene_callback(render_state.renderer.scene)
        rendered_image = render_state.renderer.render(**render_state.render_kwargs)
        if self.render_mode == 'segmentation_array':
            # For segmentation, channel 0 is the geom id, 1 is the object type (site, geom, etc.)
            rendered_image = rendered_image[:, :, 0:1]
        elif self.render_mode == 'depth_array':
            rendered_image = rendered_image[:, :, np.newaxis]
        return state, common_state, FuncSensorSingleState(
            sensor_state=render_state,
            remaining_time_until_read=self.control_timestep,
        ), rendered_image
    
    def close(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        sensor_single_state: FuncSensorSingleState[MujocoFuncCameraSensorState]
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator]
    ]:
        sensor_single_state.sensor_state.renderer.close()
        return state, common_state
