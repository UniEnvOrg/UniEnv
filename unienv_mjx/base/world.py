from typing import Optional, Any, Dict, Generic, TypeVar, Type, Tuple, Union, List
from unienv_interface.backends.base import ComputeBackend, BDeviceType, BDtypeType, BRNGType
from unienv_interface.world.world import FuncWorld, FuncEnvCommonState
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_mujoco import mjcf_util
import jax
import jax.numpy as jnp
import numpy as np

from mujoco import mjx
from dm_control import mjcf
import mujoco
from flax import struct

class MJXFuncWorldState(struct.PyTreeNode):
    mjx_model : mjx.Model
    mjx_data : mjx.Data
    home_qpos : JaxComputeBackend.ARRAY_TYPE
    home_ctrl : JaxComputeBackend.ARRAY_TYPE
    
    # Stored here as a reference for recompilation
    mjcf_model : mjcf.RootElement = struct.field(pytree_node=False)
    mj_model : mujoco.MjModel = struct.field(pytree_node=False)
    
    """ 
    This is used if something isn't supported by MJX yet (like camera rendering)
    Everything that uses this should first check if `mj_data_updated` is True, and use `mjx.get_data_into` if it is not updated yet.
    """
    mj_data : Union[mujoco.MjData, List[mujoco.MjData]] = struct.field(pytree_node=False)
    mj_data_updated : bool = struct.field(pytree_node=False, default=False)

class MJXFuncWorld(
    FuncWorld[MJXFuncWorldState, JaxComputeBackend.DEVICE_TYPE, JaxComputeBackend.DTYPE_TYPE, JaxComputeBackend.RNG_TYPE
]):
    backend = JaxComputeBackend
    is_real = False

    def __init__(
        self,
        world_timestep : float,
        world_subtimestep : Optional[float] = None,
        xml_path : Optional[str] = None,
        device : Optional[JaxComputeBackend.DEVICE_TYPE] = None,
    ):
        self.device = device
        
        # Mujoco Models
        self._xml_path = xml_path
        self._mjcf_model = self.build_mjcf_model()
        self._mjmodel = mjcf_util.compile_mjcf(self._mjcf_model)
        self._mjx_model = mjx.put_model(self._mjmodel, device=self.device)

        # Set the timestep
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

    def recompile_mjcf_state(self, mjcf : mjcf.RootElement) -> MJXFuncWorldState:
        """
        This is used to re-compile the world state if the mjcf model is changed (by the Task / Sensor / Actor)
        When making changes to the mjcf model, it is recommended to first call `copy.deepcopy` on the mjcf contained in the world state
            to avoid contaminating the original mjcf model in the World object
        """

        mj_model = mjcf_util.compile_mjcf(mjcf)
        mj_model.opt.timestep = self._world_subtimestep
        mj_data = mujoco.MjData(mj_model)
        
        mjx_model = mjx.put_model(mj_model, device=self.device)
        try:
            home_keypose = mj_model.key("home")
            mj_data.qpos[:] = home_keypose.qpos.copy()
            mj_data.ctrl[:] = home_keypose.ctrl.copy()
        except:
            pass
        
        mujoco.mj_forward(mj_model, mj_data)

        # Cache the home qpos and ctrl for reset
        home_qpos = jnp.asarray(mj_data.qpos, device=self.device)
        home_ctrl = jnp.asarray(mj_data.ctrl, device=self.device)

        mjx_data = mjx.put_data(
            mj_model,
            mj_data,
            device=self.device
        )
        return MJXFuncWorldState(
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            home_qpos=home_qpos,
            home_ctrl=home_ctrl,
            mjcf_model=mjcf,
            mj_model=mj_model,
            mj_data=mj_data,
            mj_data_updated=True
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
        self._mjx_model = self._mjx_model.tree_replace(
            "opt.timestep",
            self._mjx_model.opt.timestep.at[:].set(self._world_subtimestep)
        )

    def build_mjcf_model(self) -> mjcf.RootElement:
        assert self._xml_path is not None
        return mjcf.from_path(self._xml_path)
    
    def initial(
        self,
        rng : JaxComputeBackend.RNG_TYPE,
    ) -> Tuple[
        MJXFuncWorldState,
        JaxComputeBackend.RNG_TYPE
    ]:
        mjcf_model = self._mjcf_model
        mj_model = self._mjmodel
        mjx_model = self._mjx_model
        mj_data = mujoco.MjData(mj_model)
        
        try:
            home_keypose = mj_model.key("home")
            mj_data.qpos[:] = home_keypose.qpos.copy()
            mj_data.ctrl[:] = home_keypose.ctrl.copy()
        except:
            pass
        
        mujoco.mj_forward(mj_model, mj_data)

        # Cache the home qpos and ctrl for reset
        home_qpos = jnp.asarray(mj_data.qpos, device=self.device)
        home_ctrl = jnp.asarray(mj_data.ctrl, device=self.device)

        mjx_data = mjx.put_data(
            mj_model,
            mj_data,
            device=self.device
        )
        return MJXFuncWorldState(
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            home_qpos=home_qpos,
            home_ctrl=home_ctrl,
            mjcf_model=mjcf_model,
            mj_model=mj_model,
            mj_data=mj_data,
            mj_data_updated=True
        ), rng
    
    def reset(
        self,
        state : MJXFuncWorldState,
        rng : JaxComputeBackend.RNG_TYPE
    ) -> Tuple[
        MJXFuncWorldState,
        JaxComputeBackend.RNG_TYPE
    ]:
        mjx_data = state.mjx_data.replace(
            qpos=state.home_qpos,
            ctrl=state.home_ctrl,
            qvel=jnp.zeros_like(state.mjx_data.qvel),
            act=jnp.zeros_like(state.mjx_data.act)
        )
        mjx_data = mjx.forward(state.mjx_model, mjx_data)
        state = state.replace(
            mjx_data=mjx_data,
            mj_data_updated=False
        )
        return state, rng
    
    def step(
        self,
        state : MJXFuncWorldState,
        rng : JaxComputeBackend.RNG_TYPE
    ) -> Tuple[
        float, # elapsed time
        MJXFuncWorldState,
        JaxComputeBackend.RNG_TYPE
    ]:
        def single_step(
            _, # unused idx
            state : MJXFuncWorldState
        ) -> MJXFuncWorldState:
            mjx_data = state.mjx_data
            mjx_data = mjx.step(state.mjx_model, mjx_data)
            return state.replace(
                mjx_data=mjx_data
            )
        state = jax.lax.fori_loop(
            0,
            self._nstep,
            single_step,
            state
        )
        return (
            self._world_timestep,
            state,
            rng
        )
    
    def close(
        self,
        state : MJXFuncWorldState,
        rng : JaxComputeBackend.RNG_TYPE
    ) -> None:
        pass