from typing import Any, Callable, Generic, TypeVar, Tuple, Dict, Optional, SupportsFloat, Type, Sequence, Union
import abc
import numpy as np
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, batch_utils as sbu
from dataclasses import dataclass, replace as dataclass_replace
from .env import Env, ContextType, ObsType, ActType, RenderFrame

StateType = TypeVar("StateType", covariant=True)
RenderStateType = TypeVar("RenderStateType", covariant=True)

@dataclass(frozen=True)
class FuncEnvCommonRenderInfo:
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
    def initial(
        self, 
        *,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[
        StateType,
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
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> Tuple[
        StateType,
        ContextType,
        ObsType,
        Dict[str, Any]
    ]:
        """
        Resets the environment to its initial state and returns the initial context and observation.
        If mask is provided, it will only return the masked context and observation, so the batch dimension in the output will not be same as the batch dimension in the context and observation spaces.
        Note that state input and output should be with full batch dimensions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, state: StateType, action : ActType) -> Tuple[
        StateType,
        ObsType, 
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
        Dict[str, Any]
    ]:
        """Transition."""
        raise NotImplementedError

    def close(self, state: StateType) -> None:
        """Close the environment."""
        return

    def render_init(
        self, 
        state : StateType, 
        *,
        seed : Optional[int] = None,
        render_mode : Optional[str] = None, 
        **kwargs
    ) -> Tuple[
        StateType,
        RenderStateType,
        FuncEnvCommonRenderInfo
    ]:
        """Initialize the render state."""
        raise NotImplementedError

    def render_image(
        self, 
        state : StateType,
        render_state : RenderStateType,
    ) -> Tuple[
        RenderFrame | Sequence[RenderFrame] | None, 
        StateType,
        RenderStateType,
    ]:
        """Render the environment."""
        raise NotImplementedError

    def render_close(
        self, 
        state : StateType,
        render_state : RenderStateType
    ) -> StateType:
        """Close the render state."""
        raise NotImplementedError

    # ========== Convenience methods ==========
    def update_observation_post_reset(
        self,
        old_obs: ObsType,
        newobs_masked: ObsType,
        mask: BArrayType
    ) -> ObsType:
        assert self.batch_size is not None, "This method is used by batched environment after reset"
        return sbu.set_at(
            self.observation_space,
            old_obs,
            mask,
            newobs_masked
        )
    
    def update_context_post_reset(
        self,
        old_context: ContextType,
        new_context: ContextType,
        mask: BArrayType
    ) -> ContextType:
        assert self.batch_size is not None, "This method is used by batched environment after reset"
        if self.context_space is None:
            return None
        return sbu.set_at(
            self.context_space,
            old_context,
            mask,
            new_context
        )

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncEnv":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["FuncEnv"]:
        return None

    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the attribute `name` exists in the environment."""
        return hasattr(self, name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Gets the attribute `name` from the environment."""
        return getattr(self, name)

    def set_wrapper_attr(self, name: str, value: Any):
        """Sets the attribute `name` on the environment with `value`."""
        setattr(self, name, value)

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
        render_mode : Optional[str] = None,
        render_kwargs : Dict[str, Any] = {},
    ):
        self.func_env = func_env

        # Environment state
        self.state : Optional[StateType] = None
        self._inited = False

        # Render related attributes
        self.render_state : Optional[RenderStateType] = None
        self._render_inited = False
        self._render_mode = render_mode
        self._render_fps = None
        if self._render_mode is None and hasattr(self.func_env, "render_mode"):
            self._render_mode = self.func_env.render_mode
        if hasattr(self.func_env, "render_fps"):
            self._render_fps = self.func_env.render_fps
        self._render_kwargs = render_kwargs

        # Env attribute overwrite
        self._metadata : Optional[Dict[str, Any]] = None
        self.rng = self.backend.random.random_number_generator(device=self.device)

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
        return self._render_mode

    @property
    def render_fps(self) -> Optional[int]:
        self._init_render()
        return self._render_fps
    
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

    def _init_render(self) -> None:
        if self._render_inited:
            return
        
        (
            self.state,
            self.render_state,
            render_info
        ) = self.func_env.render_init(
            self.state,
            render_mode=self._render_mode,
            **self._render_kwargs
        )
        self._render_mode = render_info.render_mode or self._render_mode
        self._render_fps = render_info.render_fps or self._render_fps
        self._render_inited = True
    
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
        self.state, obs, rew, terminated, truncated, info = self.func_env.step(
            self.state, action
        )
        return obs, rew, terminated, truncated, info

    def reset(
        self,
        *,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        if not self._inited:
            assert mask is None or bool(self.backend.all(mask)), "For the initial reset mask must not be provided or be all True"
            self.state, context, obs, info = self.func_env.initial(
                seed=seed,
                **kwargs
            )
            self._inited = True
        else:
            self.state, context, obs, info = self.func_env.reset(
                self.state, seed=seed, mask=mask, **kwargs
            )
        return context, obs, info

    def render(self) -> RenderFrame | Sequence[RenderFrame] | None:
        self._init_render()
        image, self.state, self.render_state = self.func_env.render_image(
            self.state, self.render_state
        )
        return image
    
    def close(self) -> None:
        if self.render_state is not None:
            self.state = self.func_env.render_close(
                self.state, self.render_state
            )
            self._render_inited = False
            self.render_state = None
        self.func_env.close(self.state)
        self.state = None
        self._inited = False
    
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