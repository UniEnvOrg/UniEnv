from .env import Env, ContextType, ObsType, ActType, RenderFrame, BArrayType, BDeviceType, BDtypeType, BRNGType
from copy import deepcopy
from typing import Any, Generic, SupportsFloat, TypeVar, Optional, Union, Dict, Tuple, Sequence, Type
import abc
from unienv_interface.space import Space
from unienv_interface.backends import ComputeBackend, ArrayAPIArray
import numpy as np

WrapperBArrayT = TypeVar("WrapperBArrayT")
WrapperContextT = TypeVar("WrapperContextT")
WrapperObsT = TypeVar("WrapperObsT")
WrapperActT = TypeVar("WrapperActT")
WrapperBDeviceT = TypeVar("WrapperBDeviceT")
WrapperBDtypeT = TypeVar("WrapperBDtypeT")
WrapperBRngT = TypeVar("WrapperBRngT")
WrapperRenderFrame = TypeVar("WrapperRenderFrame")

class Wrapper(
    Env[WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT],
    Generic[
        WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    # ========== Public Attribute Getters ==========
    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns the :attr:`Env` :attr:`metadata`."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value

    @property
    def render_mode(self) -> Optional[WrapperRenderFrame]:
        return self.env.render_mode

    @property
    def render_fps(self) -> Optional[int]:
        return self.env.render_fps

    @property
    def backend(self) -> ComputeBackend[Any, WrapperBDeviceT, Any, WrapperBRngT]:
        return self.env.backend

    @property
    def device(self) -> Optional[WrapperBDeviceT]:
        return self.env.device

    @property
    def batch_size(self) -> Optional[int]:
        return self.env.batch_size

    @property
    def action_space(
        self,
    ) -> Space[WrapperActT, WrapperBDeviceT, Any, WrapperBRngT]:
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: Space[WrapperActT, WrapperBDeviceT, Any, WrapperBRngT]):
        self._action_space = space

    @property
    def observation_space(
        self,
    ) -> Space[WrapperObsT, WrapperBDeviceT, Any, WrapperBRngT]:
        if self._observation_space is None:
            return self.env.observation_space
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

    @property
    def rng(self) -> WrapperBRngT:
        """Returns the :attr:`Env` :attr:`rng` attribute."""
        return self._rng or self.env.rng

    @rng.setter
    def rng(self, value: WrapperBRngT):
        self._rng = value

    def __init__(
        self, 
        env: Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]):
        self.env = env
        assert isinstance(env, Env)

        self._action_space: Optional[Space[WrapperActT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]] = None
        self._observation_space: Optional[Space[WrapperObsT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]] = None
        self._context_space: Optional[Space[WrapperContextT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]] = self.env.context_space
        self._metadata: Optional[Dict[str, Any]] = None
        self._rng : Optional[WrapperBRngT] = None

    def step(
        self, action: WrapperActT
    ) -> Tuple[
        WrapperObsT, 
        Union[SupportsFloat, WrapperBArrayT], 
        Union[bool, WrapperBArrayT], 
        Union[bool, WrapperBArrayT], 
        Dict[str, Any]
    ]:
        return self.env.step(action)

    def reset(
        self, 
        *args, 
        mask : Optional[WrapperBArrayT] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[WrapperContextT, WrapperObsT, Dict[str, Any]]:
        return self.env.reset(
            *args,
            mask=mask,
            seed=seed,
            **kwargs
        )

    def render(self) -> RenderFrame | Sequence[RenderFrame] | None:
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self) -> Env:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]:
        return self.env

    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the given attribute is within the wrapper or its environment."""
        if hasattr(self, name):
            return True
        else:
            return self.env.has_wrapper_attr(name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Gets an attribute from the wrapper and lower environments if `name` doesn't exist in this object.

        Args:
            name: The variable name to get

        Returns:
            The variable with name in wrapper or lower environments
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.env.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        """Sets an attribute on this wrapper or lower environment if `name` is already defined.

        Args:
            name: The variable name
            value: The new variable value
        """
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
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

class ActionWrapper(
    Wrapper[
        BArrayType, ContextType, ObsType, WrapperActT, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
    ],
    Generic[
        WrapperActT, 
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
    ]
):
    @abc.abstractmethod
    def map_action(self, action : WrapperActT) -> ActType:
        raise NotImplementedError
    
    def reverse_map_action(self, action : ActType) -> WrapperActT:
        raise NotImplementedError
    
    def step(
        self, action: WrapperActT
    ) -> Tuple[
        ObsType, 
        Union[SupportsFloat, BArrayType], 
        Union[bool, BArrayType], 
        Union[bool, BArrayType], 
        Dict[str, Any]
    ]:
        return self.env.step(self.map_action(action))
    
class ContextObservationWrapper(
    Wrapper[
        BArrayType, WrapperContextT, WrapperObsT, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
    ],
    Generic[
        WrapperContextT, WrapperObsT, 
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
    ]
):
    def map_context(self, context : WrapperContextT) -> ContextType:
        return context
    
    def reverse_map_context(self, context : ContextType) -> WrapperContextT:
        raise NotImplementedError
    
    def map_observation(self, observation : WrapperObsT) -> ObsType:
        return observation
    
    def reverse_map_observation(self, observation : ObsType) -> WrapperObsT:
        raise NotImplementedError
    
    def reset(
        self, 
        *args, 
        mask : Optional[BArrayType] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[WrapperContextT, WrapperObsT, Dict[str, Any]]:
        context, observation, info = self.env.reset(
            *args,
            mask=mask,
            seed=seed,
            **kwargs
        )
        return self.map_context(context), self.map_observation(observation), info
    
    def step(
        self, action: ActType
    ) -> Tuple[
        WrapperObsT, 
        Union[float, BArrayType], 
        Union[bool, BArrayType], 
        Union[bool, BArrayType], 
        Dict[str, Any]
    ]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.map_observation(observation), reward, terminated, truncated, info
