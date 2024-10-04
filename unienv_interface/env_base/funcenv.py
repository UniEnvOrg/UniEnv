from typing import Any, Callable, Generic, TypeVar, Tuple, Dict, Optional, SupportsFloat, Type, Sequence
import abc
import numpy as np
from unienv_interface.backends import ComputeBackend
import gymnasium as gym
from ..space import Space
from dataclasses import dataclass
from .env import Env, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT
StateType = TypeVar("StateType")
RenderStateType = TypeVar("RenderStateType")

# ObsType = TypeVar("ObsType")
# ActType = TypeVar("ActType")
# RewardType = TypeVar("RewardType")
# TerminationType = TypeVar("TerminalType")
# RenderFrame = TypeVar("RenderFrame")
# BDeviceT = TypeVar("BDeviceT")
# BRngT = TypeVar("BRngT")

@dataclass
class FuncEnvCommonState(Generic[BDeviceT, BRngT]):
    np_rng : np.random.Generator
    rng : BRngT
    device : Optional[BDeviceT] = None

@dataclass
class FuncEnvCommonRenderState:
    render_mode : Optional[str] = None
    render_fps : Optional[int] = None

class FuncEnv(
    abc.ABC,
    Generic[StateType, RenderStateType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT]
):
    metadata : Dict[str, Any] = {
        "render_modes": []
    }

    is_batched : bool = False

    backend : Type[ComputeBackend[Any, BDeviceT, Any, BRngT]]

    observation_space: Space[Any, Any, BDeviceT, Any, BRngT]
    action_space: Space[Any, Any, BDeviceT, Any, BRngT]

    @abc.abstractmethod
    def initial(self, *, seed : int, device : Optional[BDeviceT] = None) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceT, BRngT],
        ObsType,
        Dict[str, Any] | Sequence[Dict[str, Any]]
    ]:
        """Initial state."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def reset(
        self, 
        state : StateType, 
        common_state : FuncEnvCommonState[BDeviceT, BRngT],
        *args,
        **kwargs
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceT, BRngT],
        ObsType,
        Dict[str, Any] | Sequence[Dict[str, Any]]
    ]:
        """Reset the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, state: StateType, common_state : FuncEnvCommonState[BDeviceT, BRngT], action : ActType) -> Tuple[
        StateType, 
        FuncEnvCommonState[BDeviceT, BRngT],
        ObsType, 
        RewardType, 
        TerminationType, 
        TerminationType, 
        Dict[str, Any] | Sequence[Dict[str, Any]]
    ]:
        """Transition."""
        raise NotImplementedError
    
    def close(self, state: StateType, common_state : FuncEnvCommonState[BDeviceT, BRngT]) -> None:
        """Close the environment."""
        return

    def render_image(
        self, 
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceT, BRngT],
        render_state : RenderStateType,
        render_common_state : FuncEnvCommonRenderState
    ) -> Tuple[
        RenderFrame | Sequence[RenderFrame] | None, 
        FuncEnvCommonState[BDeviceT, BRngT],
        RenderStateType,
        FuncEnvCommonRenderState
    ]:
        """Show the state."""
        raise NotImplementedError

    def render_init(
        self, 
        state : StateType, 
        common_state : FuncEnvCommonState[BDeviceT, BRngT], 
        *,
        render_mode : Optional[str] = None, 
    ) -> Tuple[
        FuncEnvCommonState[BDeviceT, BRngT],
        RenderStateType,
        FuncEnvCommonRenderState
    ]:
        """Initialize the render state."""
        raise NotImplementedError

    def render_close(
        self, 
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceT, BRngT],
        render_state : RenderStateType,
        render_common_state : FuncEnvCommonRenderState
    ) -> None:
        """Close the render state."""
        raise NotImplementedError

class StatefulSingleFuncEnv(Env[
    ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT
],Generic[
    StateType, RenderStateType, 
    ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT
]):
    def __init__(
        self,
        func_env : FuncEnv[StateType, RenderStateType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT],
        *instance_args,
        instance_kwargs : Dict[str, Any] = {},
        render_mode : Optional[str] = None,
        render_kwargs : Dict[str, Any] = {},
        device : Optional[BDeviceT] = None,
        seed : int = 0,
    ):
        assert not self.func_env.is_batched, "StatefulSingleFuncEnv only supports single instance environments."
        self.func_env = func_env
        self.device = device
        
        
        self.state, self.common_state, obs, info = self.func_env.initial(
            *instance_args, seed=seed, device=device, **instance_kwargs
        )
        self._first_instance_obs_info : Optional[Tuple[ObsType, Dict[str, Any]]] = (obs, info)

        self.common_state : Optional[FuncEnvCommonState[BDeviceT, BRngT]] = None,
        self.state : Optional[StateType] = None
        self.render_state : Optional[RenderStateType] = None
        self.render_common_state : Optional[FuncEnvCommonRenderState] = None

        self._metadata : Optional[Dict[str, Any]] = None

        # Construction Metadata
        self._render_mode = render_mode
        self._render_kwargs = render_kwargs

    def _init_render(self) -> None:
        if self.render_state is not None:
            return
        
        (
            self.common_state,
            self.render_state,
            self.render_common_state
        ) = self.func_env.render_init(
            self.state,
            self.common_state,
            render_mode=self._render_mode,
            **self._render_kwargs
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is not None:
            return self._metadata
        else:
            return self.func_env.metadata
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value

    @property
    def render_mode(self) -> Optional[str]:
        return self._render_mode if self.render_common_state is None else self.render_common_state.render_mode

    @property
    def render_fps(self) -> Optional[int]:
        self._init_render()
        return self.render_common_state.render_fps
    
    @property
    def backend(self) -> Type[ComputeBackend[Any, BDeviceT, Any, BRngT]]:
        return self.func_env.backend

    @property
    def action_space(self) -> Space[ActType, Any, BDeviceT, Any, BRngT]:
        return self.func_env.action_space

    @property
    def observation_space(self) -> Space[ObsType, Any, BDeviceT, Any, BRngT]:
        return self.func_env.observation_space

    @property
    def np_rng(self) -> np.random.Generator:
        return self.common_state.np_rng
    
    @property
    def rng(self) -> BRngT:
        return self.common_state.rng
    
    def reset(
        self,
        *args,
        seed : Optional[int] = None,
        **kwargs,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if (
            self._first_instance_obs_info is not None and
            seed is None and
            len(args) == 0 and len(kwargs) == 0
        ):
            obs, info = self._first_instance_obs_info
            self._first_instance_obs_info = None
            return obs, info
        else:
            self.state, self.common_state, obs, info = self.func_env.reset(
                self.state, self.common_state, seed=seed
            )
            return obs, info
    
    def step(
        self,
        action : ActType
    ) -> Tuple[ObsType, RewardType, TerminationType, TerminationType, Dict[str, Any]]:
        self.state, self.common_state, obs, rew, terminated, truncated, info = self.func_env.step(
            self.state, self.common_state, action
        )
        return obs, rew, terminated, truncated, info

    def render(self) -> RenderFrame | Sequence[RenderFrame] | None:
        self._init_render()
        return self.func_env.render_image(
            self.state, self.common_state, self.render_state, self.render_common_state
        )
    
    def close(self) -> None:
        if self.render_state is not None:
            self.func_env.render_close(
                self.state, self.common_state, self.render_state, self.render_common_state
            )
        self.func_env.close(self.state, self.common_state)
    
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or hasattr(self.func_env, name)

    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return getattr(self.func_env, name)
    
    def set_wrapper_attr(self, name: str, value: Any):
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            setattr(self.func_env, name, value)