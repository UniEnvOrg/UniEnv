from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union
from abc import ABC, abstractmethod
from unienv_interface.world.world import FuncWorld
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.utils import seed_util
import mujoco
import os.path
from dm_control import mjcf
from .. import mjcf_util
from dataclasses import dataclass, replace as dataclass_replace
import numpy as np

@dataclass(frozen=True)
class MujocoFuncWorldState:
    mjcf_model : mjcf.RootElement
    mj_model : mujoco.MjModel
    data : mujoco.MjData
    home_qpos : np.ndarray
    home_ctrl : np.ndarray

    def replace(self, **kwargs) -> 'MujocoFuncWorldState':
        return dataclass_replace(self, **kwargs)

class MujocoFuncWorld(FuncWorld[MujocoFuncWorldState, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE]):
    backend = NumpyComputeBackend
    device = None

    is_real = False

    def __init__(
        self,
        world_timestep : float,
        world_subtimestep : Optional[float] = None,
        xml_path : Optional[Union[str, Any]] = None
    ):
        self._xml_path = xml_path
        self._mjcf_model = self.build_mjcf_model()
        self._mjmodel = mjcf_util.compile_mjcf(self._mjcf_model)
        self.set_timestep(world_timestep, world_subtimestep)

    @property
    def world_timestep(self) -> float:
        return self._world_timestep
    
    @world_timestep.setter
    def world_timestep(self, value : float) -> None:
        self.set_timestep(value, min(self._world_subtimestep, value))
    
    @property
    def world_subtimestep(self) -> Optional[float]:
        return self._world_subtimestep
    
    @world_subtimestep.setter
    def world_subtimestep(self, value : Optional[float]) -> None:
        if value is None:
            value = self._world_timestep
        self.set_timestep(max(self.world_timestep, value), value)

    def recompile_mjcf_state(self, mjcf_root : mjcf.RootElement) -> MujocoFuncWorldState:
        """
        This is used to re-compile the world state if the mjcf model is changed (by the Task / Sensor / Actor)
        When making changes to the mjcf model, it is recommended to first call `copy.deepcopy` on the mjcf contained in the world state
            to avoid contaminating the original mjcf model in the World object
        """
        mj_model = mjcf_util.compile_mjcf(mjcf_root)
        mj_model.opt.timestep = self._world_subtimestep
        mj_data = mujoco.MjData(mj_model)
        
        try:
            home_keypose = mj_model.key("home")
            mj_data.qpos[:] = home_keypose.qpos.copy()
            mj_data.ctrl[:] = home_keypose.ctrl.copy()
        except:
            pass

        home_qpos = mj_data.qpos.copy()
        home_ctrl = mj_data.ctrl.copy()

        mujoco.mj_forward(mj_model, mj_data)

        return MujocoFuncWorldState(
            mjcf_model=mjcf_root,
            mj_model=mj_model,
            data=mj_data,
            home_qpos=home_qpos,
            home_ctrl=home_ctrl
        )

    def set_timestep(self, world_timestep : float, world_subtimestep : Optional[float] = None) -> None:
        if world_subtimestep is None:
            world_subtimestep = world_timestep
        assert world_timestep > 0 and world_subtimestep > 0
        n_step = int(world_timestep / world_subtimestep)
        assert n_step > 0
        self._world_timestep = world_timestep
        self._world_subtimestep = self._world_timestep / n_step
        self._nstep = n_step

        # Set the sub-timestep
        self._mjmodel.opt.timestep = self._world_subtimestep

    def build_mjcf_model(self) -> mjcf.RootElement:
        assert self._xml_path is not None
        return mjcf.from_path(self._xml_path)

    def initial(
        self,
        rng : np.random.Generator,
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator
    ]:
        mj_data = mujoco.MjData(self._mjmodel)
        
        try:
            home_keypose = self._mjmodel.key("home")
            mj_data.qpos[:] = home_keypose.qpos.copy()
            mj_data.ctrl[:] = home_keypose.ctrl.copy()
        except:
            pass
        
        home_qpos = mj_data.qpos.copy()
        home_ctrl = mj_data.ctrl.copy()

        mujoco.mj_forward(self._mjmodel, mj_data)

        state = MujocoFuncWorldState(
            mjcf_model=self._mjcf_model,
            mj_model=self._mjmodel,
            data=mj_data,
            home_qpos=home_qpos,
            home_ctrl=home_ctrl
        )
        return state, rng
    
    def reset(
        self,
        state : MujocoFuncWorldState,
        rng : np.random.Generator
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator
    ]:
        mj_data = state.data
        mj_data.qpos[:] = state.home_qpos.copy()
        mj_data.qvel[:] = 0
        mj_data.ctrl[:] = state.home_ctrl.copy()
        mujoco.mj_forward(state.mj_model, mj_data)
        return state, rng
    
    def step(
        self,
        state : MujocoFuncWorldState,
        rng : np.random.Generator
    ) -> Tuple[
        float, # elapsed time
        MujocoFuncWorldState,
        np.random.Generator
    ]:
        mujoco.mj_step(state.mj_model, state.data, nstep=self._nstep)
        return self._world_timestep, state, rng
    
    def close(
        self,
        state : MujocoFuncWorldState,
        rng : np.random.Generator
    ) -> None:
        pass
