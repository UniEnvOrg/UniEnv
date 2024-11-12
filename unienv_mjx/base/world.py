from typing import Optional, Any, Dict, Generic, TypeVar, Type, Tuple
from unienv_interface.backends.base import ComputeBackend, BDeviceType, BDtypeType, BRNGType
from unienv_interface.world.world import FuncWorld, FuncEnvCommonState
from unienv_interface.backends.jax import JaxComputeBackend, JaxDevice, JaxRNG
from unienv_mujoco import mjcf_util
import jax
import jax.numpy as jnp
import numpy as np


from mujoco import mjx
from dm_control import mjcf
import mujoco
from flax import struct

@struct.dataclass
class MJXFuncWorldState:
    data : mjx.Data
    mjcf_model : mjcf.RootElement = struct.field(pytree_node=False)
    mj_model : mujoco.MjModel = struct.field(pytree_node=False)

class MJXFuncWorld(FuncWorld[MJXFuncWorldState, JaxDevice, jnp.dtype, JaxRNG]):
    is_real = False
    backend = JaxComputeBackend

    def __init__(
        self,
        world_timestep : float,
        world_subtimestep : Optional[float] = None,
        xml_path : Optional[str] = None
    ):
        self._xml_path = xml_path
        self._mjcf_model = self.build_mjcf_model()
        self._mjmodel = mjcf_util.compile_mjcf(self._mjcf_model)

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

    def recompile_mjcf(self) -> None:
        self._mjmodel = mjcf_util.compile_mjcf(self._mjcf_model)
        self._mjmodel.opt.timestep = self._world_subtimestep

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
    
    