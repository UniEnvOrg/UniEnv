from .funcenv import FuncEnv, ContextType, ObsType, ActType, RenderFrame, StateType, RenderStateType, BArrayType, BDeviceType, BDtypeType, BRNGType
from .wrapper import WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT, WrapperRenderFrame
from typing import Any, Generic, SupportsFloat, TypeVar, Optional, Union, Dict, Tuple, Sequence, Type
import abc
from unienv_interface.space import Space
from unienv_interface.backends import ComputeBackend
import numpy as np

WrapperStateT = TypeVar("WrapperStateT")
WrapperRenderStateT = TypeVar("WrapperRenderStateT")

class FuncEnvWrapper(
    Generic[
        WrapperStateT, WrapperRenderStateT, 
        WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
        StateType, RenderStateType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ],
    FuncEnv[
        WrapperStateT, WrapperRenderStateT, 
        WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
    ]
):
    # ========== Public Attribute Getter / Setters ===========
    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns the :attr:`Env` :attr:`metadata`."""
        if self._metadata is None:
            return self.func_env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value
    
    @property
    def backend(self) -> ComputeBackend[Any, WrapperBDeviceT, Any, WrapperBRngT]:
        return self.func_env.backend

    @property
    def device(self) -> Optional[WrapperBDeviceT]:
        return self.func_env.device

    @property
    def batch_size(self) -> Optional[int]:
        return self.func_env.batch_size

    @property
    def action_space(
        self,
    ) -> Space[WrapperActT, WrapperBDeviceT, Any, WrapperBRngT]:
        if self._action_space is None:
            return self.func_env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: Space[WrapperActT, WrapperBDeviceT, Any, WrapperBRngT]):
        self._action_space = space

    @property
    def observation_space(
        self,
    ) -> Space[WrapperObsT, WrapperBDeviceT, Any, WrapperBRngT]:
        if self._observation_space is None:
            return self.func_env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: Space[WrapperObsT, WrapperBDeviceT, Any, WrapperBRngT]):
        self._observation_space = space

    @property
    def context_space(
        self,
    ) -> Optional[Space[WrapperContextT, WrapperBDeviceT, Any, WrapperBRngT]]:
        return self._context_space
    
    @context_space.setter
    def context_space(self, space: Optional[Space[WrapperContextT, WrapperBDeviceT, Any, WrapperBRngT]]):
        self._context_space = space
    
    def __init__(
        self,
        func_env : FuncEnv[
            StateType, RenderStateType,
            BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
        ]
    ):
        self.func_env = func_env

        # Initialize optional replacement attributes
        self._action_space: Optional[Space[WrapperActT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]] = None
        self._observation_space: Optional[Space[WrapperObsT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]] = None
        self._context_space: Optional[Space[WrapperContextT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]] = self.func_env.context_space
        self._metadata: Optional[Dict[str, Any]] = None
    
    def initial(self, rng):
        return self.func_env.initial(rng)

    def reset(self, state, rng, mask):
        return self.func_env.reset(state, rng, mask)

    def step(self, state, rng, action):
        return self.func_env.step(state, rng, action)

    def close(self, state, rng) -> None:
        self.func_env.close(state, rng)
    
    def render_image(self, state, render_state, rng):
        return self.func_env.render_image(state, render_state, rng)

    def render_close(self, state, render_state, rng):
        return self.func_env.render_close(state, render_state, rng)

    # ========== Wrapper Methods ==========
    @property
    def unwrapped(self) -> "FuncEnv":
        return self.func_env.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> FuncEnv[
        StateType, RenderStateType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]:
        return self.func_env

    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the attribute `name` exists in the environment."""
        if hasattr(self, name):
            return True
        else:
            return self.func_env.has_wrapper_attr(name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Gets the attribute `name` from the environment."""
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.func_env.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        """Sets the attribute `name` on the environment with `value`."""
        sub_env = self
        attr_set = False

        while attr_set is False and sub_env is not None:
            if hasattr(sub_env, name):
                setattr(sub_env, name, value)
                attr_set = True
            else:
                sub_env = sub_env.prev_wrapper_layer

        if attr_set is False and sub_env is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )
    
    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.func_env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)