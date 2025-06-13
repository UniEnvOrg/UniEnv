from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable, Literal
from abc import ABC, abstractmethod

from unienv_interface.world.sensors.camera import FuncCameraSensor
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import DictSpace, BoxSpace
import mujoco
from dm_control import mjcf
from dataclasses import dataclass
import numpy as np
import PIL
import PIL.Image
from unienv_mujoco import mjcf_util
from unienv_mujoco.base import *
import transforms3d as tr3d

MujocoFuncEnvSceneCallback = Callable[[mujoco.MjvScene], None]

@dataclass
class MujocoFuncCameraSensorState:
    renderer : mujoco.Renderer
    camera : Union[mujoco.MjvCamera, int, str]
    render_kwargs : Dict[str, Any]
    scene_option : Optional[mujoco.MjvOption] = None
    scene_callback : Optional[MujocoFuncEnvSceneCallback] = None

"""
Camera sensor for Mujoco environment.
Some useful information for segmentation:
mujoco.mju_type2Str and mujoco.mju_str2Type can be used to convert between object type id and name
"""
class MujocoFuncCameraSensor(
    FuncCameraSensor[MujocoFuncWorldState, MujocoFuncCameraSensorState, np.ndarray, Any, np.dtype, np.random.Generator],
):
    PINHOLE_ROTATION_MATRIX = tr3d.euler.euler2mat(np.pi, 0, 0)

    def __init__(
        self,
        camera : Union[int, str],
        width : int,
        height : int,
        camera_mode : Literal['rgb_array', 'depth_array', 'segmentation_array', 'segmentation_type_array'],
        control_timestep : float,
    ):
        assert control_timestep > 0.0
        self.control_timestep = control_timestep
        
        assert width > 0 and height > 0
        self._camera_mode = camera_mode
        channels = 3 if camera_mode == 'rgb_array' else 1
        if camera_mode == 'depth_array':
            observation_dtype = np.float32
            observation_min = 0.0
            observation_max = np.inf
        elif camera_mode in ['segmentation_array', 'segmentation_type_array']:
            observation_dtype = int
            observation_min = -1
            observation_max = 1e16
        else:
            observation_dtype = np.uint8
            observation_min = 0
            observation_max = 255
        self.observation_space = BoxSpace(
            backend=NumpyComputeBackend,
            low=observation_min,
            high=observation_max,
            dtype=observation_dtype,
            device=None,
            shape=(height, width, channels),
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
    def camera_mode(self) -> Literal['rgb_array', 'depth_array', 'segmentation_array']:
        return self._camera_mode

    def to_image(self, data : np.ndarray) -> PIL.Image:
        if self.camera_mode == 'depth_array':
            min_depth = np.min(data)
            max_depth = np.max(data)
            delta_depth = max_depth - min_depth
            if delta_depth <= 1e-3:
                data = np.clip(data / max_depth, 0.0, 1.0)
            else:
                data = np.clip((data - min_depth) / delta_depth, 0.0, 1.0)
            data = (data * 255)[:, :, 0].astype(np.uint8),
            image = PIL.Image.fromarray(
                data,
                mode='L'
            )
        elif self.camera_mode in ['segmentation_array', 'segmentation_type_array']:
            # Since Infinity is mapped to -1, we need to map it to 0
            data = data.astype(np.float64) + 1

            # Normalize the data to [0, 1]
            data = data / np.max(data)
            pixels = (data * 255).astype(np.uint8)
            pixels = pixels[:, :, 0]
            image = PIL.Image.fromarray(
                pixels[:, :], 
                mode='L'
            )
        else:
            image = PIL.Image.fromarray(data, mode='RGB')
        return image
    
    def get_intrinsic_matrix(
        self,
        state: MujocoFuncWorldState,
        rng: np.random.Generator,
    ) -> np.ndarray:
        cam_id = state.mj_model.camera_name2id(self.camera) if isinstance(self.camera, str) else self.camera
        fovy = state.mj_model.cam_fovy[cam_id]
        f = self.height / (2 * np.tan(fovy / 2))
        return np.array([
            [f, 0, self.width / 2],
            [0, f, self.height / 2],
            [0, 0, 1]
        ])

    def get_extrinsic_matrix(
        self,
        state: MujocoFuncWorldState,
        rng: np.random.Generator,
    ) -> np.ndarray:
        cam_id = state.mj_model.camera_name2id(self.camera) if isinstance(self.camera, str) else self.camera
        cam_pos = state.data.cam_xpos[cam_id*3:cam_id*3+3]
        cam_xmat = state.data.cam_xmat[cam_id*9:cam_id*9+9].reshape(3, 3)
        cam_xmat_inv = cam_xmat.T

        """
        Because for our produced image (rendered), 
        we have
        x axis pointing to the right, 
        y axis pointing downwards, 
        and z axis pointing into the plane of the image

        We rotate the default mujoco camera coordinate system (x right, y up, z out of the screen)
        """
        cam_extrinsic_rotation = self.PINHOLE_ROTATION_MATRIX @ cam_xmat_inv
        cam_extrinsic_offset = -cam_extrinsic_rotation @ cam_pos
        cam_extrinsic_matrix = np.eye(4)
        cam_extrinsic_matrix[:3, :3] = cam_extrinsic_rotation
        cam_extrinsic_matrix[:3, 3] = cam_extrinsic_offset
        return cam_extrinsic_matrix

    def initial(
        self,
        world : MujocoFuncWorld,
        state : MujocoFuncWorldState,
        rng : np.random.Generator,
        # ---- Additional parameters ----
        render_kwargs : Dict[str, Any] = {},
        max_geom : int = 10_000,
        scene_option : Optional[mujoco.MjvOption] = None,
        scene_callback : Optional[MujocoFuncEnvSceneCallback] = None,
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        MujocoFuncCameraSensorState
    ]:
        renderer = mujoco.Renderer(
            model=state.mj_model,
            height=self.height,
            width=self.width,
            max_geom=max_geom,
        )
        if self.camera_mode == 'depth_array':
            renderer.enable_depth_rendering()
        elif self.camera_mode in ['segmentation_array', 'segmentation_type_array']:
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
        return state, rng, camera_state

    def reset(
        self,
        world : MujocoFuncWorld,
        state : MujocoFuncWorldState,
        rng : np.random.Generator,
        sensor_state : MujocoFuncCameraSensorState
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        MujocoFuncCameraSensorState
    ]:
        return state, rng, sensor_state

    def step(
        self,
        state : MujocoFuncWorldState,
        rng : np.random.Generator,
        sensor_state : MujocoFuncCameraSensorState,
        last_step_elapsed : float
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        MujocoFuncCameraSensorState
    ]:
        return state, rng, sensor_state
    
    def get_data(
        self, 
        state: MujocoFuncWorldState, 
        rng: np.random.Generator, 
        sensor_state: MujocoFuncCameraSensorState,
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState, 
        np.random.Generator, 
        MujocoFuncCameraSensorState, 
        np.ndarray
    ]:
        sensor_state.renderer.update_scene(
            data=state.data,
            camera=sensor_state.camera,
            scene_option=sensor_state.scene_option
        )
        if sensor_state.scene_callback is not None:
            sensor_state.scene_callback(sensor_state.renderer.scene)
        rendered_image = sensor_state.renderer.render(**sensor_state.render_kwargs)
        if self.camera_mode == 'segmentation_array':
            # For segmentation, channel 0 is the geom id, 1 is the object type (site, geom, etc.)
            rendered_image = rendered_image[:, :, 0:1]
        elif self.camera_mode == 'segmentation_type_array':
            # For segmentation, channel 0 is the geom id, 1 is the object type (site, geom, etc.)
            rendered_image = rendered_image[:, :, 1:2]
        elif self.camera_mode == 'depth_array':
            rendered_image = rendered_image[:, :, np.newaxis]
        return state, rng, sensor_state, rendered_image
    
    def close(
        self, 
        state: MujocoFuncWorldState, 
        rng: np.random.Generator, 
        sensor_state: MujocoFuncCameraSensorState
    ) -> Tuple[
        MujocoFuncWorldState, 
        np.random.Generator
    ]:
        sensor_state.renderer.close()
        return state, rng
