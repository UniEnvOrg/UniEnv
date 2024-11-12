from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable, Literal
import mujoco
from mujoco import mjx
from unienv_interface.world.sensor import FuncSensor
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.space import Space, Box
import numpy as np
import jax
import jax.numpy as jnp
from ..base.world import MJXFuncWorld, MJXFuncWorldState
import flax.struct

class MJXFuncGenericSensorState(flax.struct.PyTreeNode):
    sensor_id : int
    sensor_adr : int
    sensor_dim : int

class MJXFuncGenericSensor(
    FuncSensor[MJXFuncWorldState, MJXFuncGenericSensorState, JaxComputeBackend.ARRAY_TYPE, JaxComputeBackend.DEVICE_TYPE, JaxComputeBackend.DTYPE_TYPE, JaxComputeBackend.RNG_TYPE]
):
    def __init__(
        self,
        world : MJXFuncWorld,
        sensor_name : str,
        control_timestep : float,
    ):
        assert control_timestep > 0.0
        self.control_timestep = control_timestep

        ndim = world._mjmodel.sensor(sensor_name).dim[0]
        self.observation_space = Box(
            backend=JaxComputeBackend,
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            device=world.device,
            shape=(ndim,)
        )

        # Cache sensor name and ids needed for data fetching
        self._sensor_name = sensor_name
    
    @property
    def sensor_name(self) -> str:
        return self._sensor_name
    
    def initial(
        self,
        world : MJXFuncWorld,
        state : MJXFuncWorldState,
        rng : JaxComputeBackend.RNG_TYPE,
    ) -> Tuple[
        MJXFuncWorldState,
        JaxComputeBackend.RNG_TYPE,
        MJXFuncGenericSensorState
    ]:
        sensor_id = state.mj_model.sensor(self.sensor_name).id
        sensor_adr = state.mjx_model.sensor_adr[sensor_id]
        sensor_dim = state.mjx_model.sensor_dim[sensor_id]
        return state, rng, MJXFuncGenericSensorState(
            sensor_id=sensor_id, 
            sensor_adr=sensor_adr, 
            sensor_dim=sensor_dim
        )

    def reset(
        self,
        world : MJXFuncWorld,
        state : MJXFuncWorldState,
        rng : JaxComputeBackend.RNG_TYPE,
        sensor_state : MJXFuncGenericSensorState
    ) -> Tuple[
        MJXFuncWorldState,
        JaxComputeBackend.RNG_TYPE,
        MJXFuncGenericSensorState
    ]:
        return state, rng, sensor_state

    def step(
        self,
        state : MJXFuncWorldState,
        rng : JaxComputeBackend.RNG_TYPE,
        sensor_state : MJXFuncGenericSensorState,
        last_step_elapsed : float
    ) -> Tuple[
        MJXFuncWorldState,
        JaxComputeBackend.RNG_TYPE,
        MJXFuncGenericSensorState
    ]:
        return state, rng, None
    
    def get_data(
        self, 
        state: MJXFuncWorldState, 
        rng: JaxComputeBackend.RNG_TYPE, 
        sensor_state: MJXFuncGenericSensorState,
        last_control_step_elapsed: float
    ) -> Tuple[
        MJXFuncWorldState, 
        JaxComputeBackend.RNG_TYPE, 
        MJXFuncGenericSensorState, 
        jax.Array
    ]:
        sensor_data = state.mjx_data.sensordata[
            sensor_state.sensor_adr:sensor_state.sensor_adr+sensor_state.sensor_dim
        ]
        return state, rng, sensor_state, sensor_data
