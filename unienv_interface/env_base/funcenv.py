from typing import Any, Callable, Generic, TypeVar, Tuple, Dict, Optional, SupportsFloat, Type, Sequence, Union
import abc
import numpy as np
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from dataclasses import dataclass, replace as dataclass_replace
from .env import Env, ContextType, ObsType, ActType, RenderFrame

StateType = TypeVar("StateType", covariant=True)
RenderStateType = TypeVar("RenderStateType", covariant=True)

@dataclass(frozen=True)
class FuncEnvCommonRenderState:
    render_mode : Optional[str] = None
    render_fps : Optional[int] = None

class FuncEnv(
    abc.ABC,
    Generic[
        StateType, RenderStateType, 
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    metadata : Dict[str, Any] = {
        "render_modes": []
    }

    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType] = None

    batch_size : Optional[int] = None

    observation_space: Space[Any, BDeviceType, BDtypeType, BRNGType]
    action_space: Space[Any, BDeviceType, BDtypeType, BRNGType]
    context_space: Optional[Space[ContextType, BDeviceType, BDtypeType, BRNGType]] = None

    @abc.abstractmethod
    def initial(self, rng : BRNGType) -> Tuple[
        StateType,
        BRNGType,
        ContextType,
        ObsType,
        Dict[str, Any]
    ]:
        """Initial state."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def reset(
        self, 
        state : StateType, 
        rng : BRNGType,
        mask : Optional[BArrayType] = None,
    ) -> Tuple[
        StateType,
        BRNGType,
        ContextType,
        ObsType,
        Dict[str, Any]
    ]:
        """
        Resets the environment to its initial state and returns the initial context and observation.
        If mask is provided, it will only return the masked context and observation, so the batch dimension in the output will not be same as the batch dimension in the context and observation spaces.
        Note that state and rng should be with their full batch dimensions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, state: StateType, rng : BRNGType, action : ActType) -> Tuple[
        StateType, 
        BRNGType,
        ObsType, 
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
        Dict[str, Any]
    ]:
        """Transition."""
        raise NotImplementedError
    
    def close(self, state: StateType, rng : BRNGType) -> None:
        """Close the environment."""
        return

    def render_image(
        self, 
        state : StateType,
        rng : BRNGType,
        render_state : RenderStateType,
        render_common_state : FuncEnvCommonRenderState
    ) -> Tuple[
        RenderFrame | Sequence[RenderFrame] | None, 
        StateType,
        BRNGType,
        RenderStateType,
        FuncEnvCommonRenderState
    ]:
        """Show the state."""
        raise NotImplementedError

    def render_init(
        self, 
        state : StateType, 
        rng : BRNGType,
        *,
        render_mode : Optional[str] = None, 
    ) -> Tuple[
        StateType,
        BRNGType,
        RenderStateType,
        FuncEnvCommonRenderState
    ]:
        """Initialize the render state."""
        raise NotImplementedError

    def render_close(
        self, 
        state : StateType,
        rng : BRNGType,
        render_state : RenderStateType,
        render_common_state : FuncEnvCommonRenderState
    ) -> Tuple[
        StateType,
        BRNGType
    ]:
        """Close the render state."""
        raise NotImplementedError

class FuncEnvBasedEnv(Env[
    BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
],Generic[
    StateType, RenderStateType, 
    BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
]):
    def __init__(
        self,
        func_env : FuncEnv[StateType, RenderStateType, BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        *,
        rng : BRNGType,
        instance_kwargs : Dict[str, Any] = {},
        render_mode : Optional[str] = None,
        render_kwargs : Dict[str, Any] = {},
    ):
        self.func_env = func_env

        self.state, self.rng, _, _, _ = self.func_env.initial(
            rng, **instance_kwargs
        )

        self.render_state : Optional[RenderStateType] = None
        self.render_common_state : Optional[FuncEnvCommonRenderState] = None

        self._metadata : Optional[Dict[str, Any]] = None

        # Construction Metadata
        self._render_mode = render_mode
        if self._render_mode is None and hasattr(self.func_env, "render_mode"):
            self._render_mode = self.func_env.render_mode
        
        self._render_kwargs = render_kwargs

    def _init_render(self) -> None:
        if self.render_common_state is not None:
            return
        
        (
            self.state,
            self.rng,
            self.render_state,
            self.render_common_state
        ) = self.func_env.render_init(
            self.state,
            self.rng,
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
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.func_env.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.func_env.device

    @property
    def batch_size(self) -> Optional[int]:
        return self.func_env.batch_size

    @property
    def action_space(self) -> Space[ActType, BDeviceType, BDtypeType, BRNGType]:
        return self.func_env.action_space

    @property
    def observation_space(self) -> Space[ObsType, BDeviceType, BDtypeType, BRNGType]:
        return self.func_env.observation_space

    @property
    def context_space(self) -> Optional[Space[ContextType, BDeviceType, BDtypeType, BRNGType]]:
        return self.func_env.context_space
    
    def step(
        self,
        action : ActType
    ) -> Tuple[
        ObsType,
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
        Dict[str, Any]
    ]:
        self.state, self.rng, obs, rew, terminated, truncated, info = self.func_env.step(
            self.state, self.rng, action
        )
        return obs, rew, terminated, truncated, info

    def reset(
        self,
        *,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        self.state, self.rng, context, obs, info = self.func_env.reset(
            self.state, self.rng, mask=mask
        )
        return context, obs, info

    def render(self) -> RenderFrame | Sequence[RenderFrame] | None:
        self._init_render()
        image, self.state, self.rng, self.render_state, self.render_common_state = self.func_env.render_image(
            self.state, self.rng, self.render_state, self.render_common_state
        )
        return image
    
    def close(self) -> None:
        if self.render_state is not None:
            self.state, self.rng = self.func_env.render_close(
                self.state, self.rng, self.render_state, self.render_common_state
            )
        self.func_env.close(self.state, self.rng)
    
    # ========== Wrapper methods ==========

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