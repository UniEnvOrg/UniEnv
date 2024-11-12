from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type
from abc import ABC, abstractmethod
from unienv_interface.world.actor import Actor, FuncActor
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.space import Dict as DictSpace, Box
import mujoco
from mujoco import mjx
from dm_control import mjcf
from dataclasses import dataclass, replace as dataclass_replace
import numpy as np
import jax
import jax.numpy as jnp
from ..base.world import MJXFuncWorldState, MJXFuncWorld

class MJXDefaultFuncActor(
    FuncActor[MJXFuncWorldState, None, JaxComputeBackend.DEVICE_TYPE, JaxComputeBackend.DTYPE_TYPE, JaxComputeBackend.RNG_TYPE]
):
    backend : Type[JaxComputeBackend] = JaxComputeBackend
    is_real = False

    def __init__(
        self,
        world : MJXFuncWorld,
        control_timestep : float,
    ):
        self.device = world.device # Copy the device from the world

        # Compute Observation Space
        self.extra_observation_space = None

        # Compute Action Space
        is_limited = world._mjx_model.actuator_ctrllimited.ravel().astype(bool)
        ctrlrange = world._mjx_model.actuator_ctrlrange
        self.extra_action_space = Box(
            backend=JaxComputeBackend,
            low=jnp.asarray(np.where(is_limited, ctrlrange[:, 0], -mujoco.mjMAXVAL)),
            high=jnp.asarray(np.where(is_limited, ctrlrange[:, 1], mujoco.mjMAXVAL)),
            dtype=np.float32,
            device=self.device
        )

        # Set the control timestep
        assert control_timestep > 0
        self.control_timestep = control_timestep

        super().__init__()

    def onboard_initial(
        self,
        world : MJXFuncWorld,
        state : MJXFuncWorldState,
        rng : JaxComputeBackend.RNG_TYPE,
    ) -> Tuple[
        MJXFuncWorldState,
        JaxComputeBackend.RNG_TYPE,
        None
    ]:
        return state, rng, None
    
    def onboard_reset(
        self,
        world : MJXFuncWorld,
        state : MJXFuncWorldState,
        rng : JaxComputeBackend.RNG_TYPE,
        actor_state : None
    ) -> Tuple[
        MJXFuncWorldState,
        JaxComputeBackend.RNG_TYPE,
        None
    ]:
        return state, rng, None
    
    def onboard_step(
        self,
        state : MJXFuncWorldState,
        rng : JaxComputeBackend.RNG_TYPE,
        actor_state : None,
        last_step_elapsed : float
    ) -> Tuple[
        MJXFuncWorldState,
        JaxComputeBackend.RNG_TYPE,
        None
    ]:
        return state, rng, None
    
    def set_next_extra_action(
        self, 
        state: MJXFuncWorldState, 
        rng: JaxComputeBackend.RNG_TYPE,
        actor_state: None, 
        action: jax.Array, 
        last_control_step_elapsed: float
    ) -> Tuple[
        MJXFuncWorldState, 
        JaxComputeBackend.RNG_TYPE,
        None
    ]:
        state = state.replace(
            mjx_data=state.mjx_data.replace(
                ctrl=action
            )
        )
        return state, rng, None
    