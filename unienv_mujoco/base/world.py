from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union
from abc import ABC, abstractmethod
from unienv_interface.world.world import FuncWorld, FuncEnvCommonState
from unienv_interface.backends.numpy import NumpyComputeBackend
import mujoco
import os.path
from dm_control import mjcf
from .. import mjcf_util
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MujocoFuncWorldState:
    mjcf_model : mjcf.RootElement
    mj_model : mujoco.MjModel
    data : mujoco.MjData

class MujocoFuncWorld(FuncWorld[MujocoFuncWorldState, Any, np.random.Generator]):
    is_real = False
    backend = NumpyComputeBackend

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
        self.set_timestep(max(self._world_timestep, value), value)

    def recompile_mjcf(self) -> None:
        self._mjmodel = mjcf_util.compile_mjcf(self._mjcf_model)
        self._mjmodel.opt.timestep = self._world_subtimestep

    def recompile_mjcf_state(self, state : MujocoFuncWorldState) -> MujocoFuncWorldState:
        mj_model = mjcf_util.compile_mjcf(state.mjcf_model)
        mj_model.opt.timestep = self._world_subtimestep
        mj_data = mujoco.MjData(mj_model)
        
        try:
            home_keypose = self._mjmodel.key("home")
            mj_data.qpos[:] = home_keypose.qpos.copy()
            mj_data.ctrl[:] = home_keypose.ctrl.copy()
        except:
            pass

        mujoco.mj_forward(mj_model, mj_data)

        return MujocoFuncWorldState(
            mjcf_model=state.mjcf_model,
            mj_model=mj_model,
            data=mj_data,
        )

    def set_timestep(self, world_timestep : float, world_subtimestep : Optional[float] = None) -> None:
        assert world_timestep > 0 and (world_subtimestep is None or world_subtimestep > 0)
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
        *,
        seed : int
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator]
    ]:
        np_random = np.random.default_rng(seed)
        common_state = FuncEnvCommonState(
            np_rng=np_random,
            rng=np_random,
            device=None
        )
        mj_data = mujoco.MjData(self._mjmodel)
        
        try:
            home_keypose = self._mjmodel.key("home")
            mj_data.qpos[:] = home_keypose.qpos.copy()
            mj_data.ctrl[:] = home_keypose.ctrl.copy()
        except:
            pass

        mujoco.mj_forward(self._mjmodel, mj_data)

        world_state = MujocoFuncWorldState(
            mjcf_model=self._mjcf_model,
            mj_model=self._mjmodel,
            data=mj_data
        )
        return world_state, common_state
    
    def reset(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator]
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator]
    ]:
        return self.initial(
            seed=common_state.np_rng.integers(0)
        )
    
    def step(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator]
    ) -> Tuple[
        float, # elapsed time
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator]
    ]:
        mj_data = state.data
        mujoco.mj_step(state.mj_model, mj_data, nstep=self._nstep)
        return (
            state.mj_model.opt.timestep,
            MujocoFuncWorldState(
                mjcf_model=state.mjcf_model,
                mj_model=state.mj_model,
                data=mj_data
            ),
            common_state
        )
    
    def close(
        self,
        state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[Any, np.random.Generator]
    ) -> None:
        pass
