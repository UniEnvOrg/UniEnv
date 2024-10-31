from .env import Env, ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT
from copy import deepcopy
from typing import Any, Generic, SupportsFloat, TypeVar, Optional, Union, Dict, Tuple, Sequence, Type
import abc
from ..space import Space
from ..backends import ComputeBackend
import numpy as np

WrapperContextType = TypeVar("WrapperContextType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")
WrapperRewardType = TypeVar("WrapperRewardType", bound=SupportsFloat)
WrapperTerminationType = TypeVar("WrapperTerminationType")
WrapperBDeviceT = TypeVar("WrapperBDeviceT")
WrapperBRngT = TypeVar("WrapperBRngT")
WrapperRenderFrame = TypeVar("WrapperRenderFrame")

class Wrapper(
    Env[WrapperContextType, WrapperObsType, WrapperActType, WrapperRewardType, WrapperTerminationType, WrapperRenderFrame, WrapperBDeviceT, WrapperBRngT],
    Generic[
        WrapperContextType, WrapperObsType, WrapperActType, WrapperRewardType, WrapperTerminationType, WrapperRenderFrame, WrapperBDeviceT, WrapperBRngT,
        ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT
    ]
):
    def __init__(self, env: Env[ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT]):
        self.env = env
        assert isinstance(env, Env)

        self._action_space: Space[WrapperActType, Any, WrapperBDeviceT, Any, WrapperBRngT] | None = None
        self._observation_space: Space[WrapperObsType, Any, WrapperBDeviceT, Any, WrapperBRngT] | None = None
        self._context_space: Space[WrapperContextType, Any, WrapperBDeviceT, Any, WrapperBRngT] | None = self.env.context_space
        self._metadata: Dict[str, Any] | None = None

    def step(
        self, action: WrapperActType
    ) -> Tuple[WrapperObsType, WrapperRewardType, WrapperTerminationType, WrapperTerminationType, Dict[str, Any]]:
        return self.env.step(action)

    def reset(
        self, 
        *, 
        seed: Optional[int] = None
    ) -> Tuple[WrapperContextType, WrapperObsType, Dict[str, Any]]:
        return self.env.reset(seed=seed)

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
    def prev_wrapper_layer(self) -> Env[ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT]:
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

    # ========== Public Attribute Getters ==========

    @property
    def action_space(
        self,
    ) -> Space[WrapperActType, Any, WrapperBDeviceT, Any, WrapperBRngT]:
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: Space[WrapperActType, Any, WrapperBDeviceT, Any, WrapperBRngT]):
        self._action_space = space

    @property
    def observation_space(
        self,
    ) -> Space[WrapperObsType, Any, WrapperBDeviceT, Any, WrapperBRngT]:
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: Space[WrapperObsType, Any, WrapperBDeviceT, Any, WrapperBRngT]):
        self._observation_space = space

    @property
    def context_space(
        self,
    ) -> Optional[Space[WrapperContextType, Any, WrapperBDeviceT, Any, WrapperBRngT]]:
        return self._context_space
    
    @context_space.setter
    def context_space(self, space: Optional[Space[WrapperContextType, Any, WrapperBDeviceT, Any, WrapperBRngT]]):
        self._context_space = space

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
    def backend(self) -> Type[ComputeBackend[Any, WrapperBDeviceT, Any, WrapperBRngT]]:
        return self.env.backend

    @property
    def device(self) -> Optional[WrapperBDeviceT]:
        return self.env.device

    @property
    def batch_size(self) -> Optional[int]:
        return self.env.batch_size

    @property
    def np_rng(self) -> np.random.Generator:
        """Returns the :attr:`Env` :attr:`np_random` attribute."""
        return self.env.np_rng

    @property
    def rng(self) -> WrapperBRngT:
        """Returns the :attr:`Env` :attr:`rng` attribute."""
        return self.env.rng
