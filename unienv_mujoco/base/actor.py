from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type
from abc import ABC, abstractmethod
from unienv_interface.env_base.funcenv import FuncEnvCommonState
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
    FuncActor[MujocoFuncWorldState, None, np.ndarray, Any, np.dtype, np.random.Generator]
):
    is_real = False

    def __init__(
        self,
        world : MujocoFuncWorld,
        control_timestep : float,
        *,
        seed : Optional[int] = None
    ):
        self._world = world

        # Compute Observation Space
        self.onboard_observation_space = DictSpace(
            backend=NumpyComputeBackend,
            spaces={},
            device=None,
            seed=seed
        )

        # Compute Action Space
        is_limited = world._mjmodel.actuator_ctrllimited.ravel().astype(bool)
        ctrlrange = world._mjmodel.actuator_ctrlrange
        self.action_space = Box(
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
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        *, 
        seed: int
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator],
        None
    ]:
        return state, common_state, None
    
    def onboard_reset(
        self,
        state: MujocoFuncWorldState,
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: None
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        None
    ]:
        return state, common_state, None

    def onboard_step(
        self,
        state: MujocoFuncWorldState,
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: None,
        last_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        None
    ]:
        return state, common_state, None
    
    def get_data_onboard(
        self,
        state: MujocoFuncWorldState,
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: None,
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState,
        FuncEnvCommonState[Any, np.random.Generator],
        None,
        Dict[str, Any]
    ]:
        return state, common_state, None , {}

    def set_next_action(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator], 
        actor_state: None,
        action: np.ndarray,
        last_control_step_elapsed: float
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator], 
        None
    ]:
        mjdata = state.data
        mjdata.ctrl[:] = action
        new_state = MujocoFuncWorldState(
            mjcf_model=state.mjcf_model,
            mj_model=state.mj_model,
            data=mjdata
        )
        return new_state, common_state, None

    def onboard_close(
        self, 
        state: MujocoFuncWorldState, 
        common_state: FuncEnvCommonState[Any, np.random.Generator],
        actor_state: None
    ) -> Tuple[
        MujocoFuncWorldState, 
        FuncEnvCommonState[Any, np.random.Generator]
    ]:
        return state, common_state

