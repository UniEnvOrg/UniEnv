from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type
from abc import ABC, abstractmethod
from unienv_interface.world.actor import Actor, FuncActor
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import Dict as DictSpace, Box
import mujoco
from dm_control import mjcf
from dataclasses import dataclass, replace as dataclass_replace
import numpy as np
from .. import mjcf_util
from .world import MujocoFuncWorldState, MujocoFuncWorld

class MujocoDefaultFuncActor(
    FuncActor[MujocoFuncWorldState, None, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE]
):
    is_real = False

    def __init__(
        self,
        world : MujocoFuncWorld,
        control_timestep : float,
    ):
        self._world = world

        # Compute Observation Space
        self.extra_observation_space = DictSpace(
            backend=NumpyComputeBackend,
            spaces={},
            device=None
        )

        # Compute Action Space
        is_limited = world._mjmodel.actuator_ctrllimited.ravel().astype(bool)
        ctrlrange = world._mjmodel.actuator_ctrlrange
        self.extra_action_space = Box(
            backend=NumpyComputeBackend,
            low=np.where(is_limited, ctrlrange[:, 0], -mujoco.mjMAXVAL),
            high=np.where(is_limited, ctrlrange[:, 1], mujoco.mjMAXVAL),
            dtype=np.float32,
            device=None
        )

        # Set the control timestep
        assert control_timestep > 0
        self.control_timestep = control_timestep

        super().__init__()

    def onboard_initial(
        self, 
        world: MujocoFuncWorld,
        state: MujocoFuncWorldState, 
        rng : np.random.Generator,
    ) -> Tuple[
        MujocoFuncWorldState, 
        np.random.Generator,
        None
    ]:
        return state, rng, None
    
    def onboard_reset(
        self,
        world: MujocoFuncWorld,
        state: MujocoFuncWorldState,
        rng: np.random.Generator,
        actor_state: None
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        None
    ]:
        return state, rng, None

    def onboard_step(
        self,
        state: MujocoFuncWorldState,
        rng: np.random.Generator,
        actor_state: None,
        last_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        None
    ]:
        return state, rng, None
    
    def get_data_extra(
        self,
        state: MujocoFuncWorldState,
        rng: np.random.Generator,
        actor_state: None,
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        np.random.Generator,
        None,
        Dict[str, Any]
    ]:
        return state, rng, None , {}
    
    def set_next_extra_action(
        self, 
        state: MujocoFuncWorldState, 
        rng: np.random.Generator, 
        actor_state: None,
        action: np.ndarray,
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState, 
        np.random.Generator, 
        None
    ]:
        mjdata = state.data
        mjdata.ctrl[:] = action
        new_state = MujocoFuncWorldState(
            mjcf_model=state.mjcf_model,
            mj_model=state.mj_model,
            data=mjdata
        )
        return new_state, rng, None

    def onboard_close(
        self, 
        state: MujocoFuncWorldState, 
        rng: np.random.Generator,
        actor_state: None
    ) -> Tuple[
        MujocoFuncWorldState, 
        np.random.Generator
    ]:
        return state, rng

