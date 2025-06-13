from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type
from abc import ABC, abstractmethod
from unienv_interface.world.actor import Actor, FuncActor
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import DictSpace, BoxSpace
import mujoco
import numpy as np
from ..base.world import MujocoFuncWorldState, MujocoFuncWorld

class MujocoDefaultFuncActor(
    FuncActor[MujocoFuncWorldState, None, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE]
):
    backend = NumpyComputeBackend
    device = None

    is_real = False

    def __init__(
        self,
        world : MujocoFuncWorld,
        control_timestep : float,
    ):
        # Compute Observation Space
        self.extra_observation_space = None

        # Compute Action Space
        is_limited = world._mjmodel.actuator_ctrllimited.ravel().astype(bool)
        ctrlrange = world._mjmodel.actuator_ctrlrange
        self.extra_action_space = BoxSpace(
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
        state.data.ctrl[:] = action
        return state, rng, None
