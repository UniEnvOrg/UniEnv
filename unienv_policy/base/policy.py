from typing import Any, TypeVar, Generic, Optional, Tuple, Union, Dict
import abc

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space

__all__ = [
    "Policy",
    "PolicyWrapper",
    "BArrayType",
    "ContextType",
    "ObsType",
    "ActType",
    "RenderFrame",
    "BDeviceType",
    "BDtypeType",
    "BRNGType",
    "WrapperBArrayT",
    "WrapperContextT",
    "WrapperObsT",
    "WrapperActT",
    "WrapperRenderFrame",
    "WrapperBDeviceT",
    "WrapperBDtypeT",
    "WrapperBRngT",
]

ContextType = TypeVar("ContextType")
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

class Policy(abc.ABC, Generic[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]):
    metadata : Dict[str, Any] = {}

    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType]

    batch_size : Optional[int] = None

    action_space : Space[ActType, BDeviceType, BDtypeType, BRNGType]
    observation_space : Space[ObsType, BDeviceType, BDtypeType, BRNGType]
    context_space : Optional[Space[ContextType, BDeviceType, BDtypeType, BRNGType]] = None

    rng : BRNGType = None

    def reset(
        self,
        context : Optional[ContextType],
        observation : ObsType,
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> None:
        """
        Resets the policy with the initial context and observation.
        If mask is provided, the context and observation only contains the context / observation for the masked batch indices, and the actor should only reset the corresponding batch indices.
        """
        pass
    
    def observe_transition(
        self,
        action_taken : ActType,
        new_observation : ObsType,
    ) -> None:
        """
        Informs the policy of the action taken and the new observation received after taking the action.
        This can be used for policies that require knowledge of the transition, such as RNN-based policies.
        By default, this method does nothing.
        """
        pass

    @abc.abstractmethod
    def compute_action(
        self,
        observation : ObsType,
        **kwargs
    ) -> ActType:
        """
        Samples an action given the current observation.
        """
        raise NotImplementedError
    
    def visualize_action(
        self,
        environment_render : RenderFrame,
        observation : ObsType,
        action : ActType,
        **kwargs
    ) -> RenderFrame:
        """
        Visualizes the action on the environment render frame.
        By default, it returns the environment render frame without any modification.
        """
        return environment_render
    
    def close(self):
        pass

    def __str__(self):
        return f"<{type(self).__name__} instance>"
    
    def __repr__(self):
        return str(self)

    def __enter__(self):
        return self
    
    def __exit__(self, *args: Any):
        self.close()
        
        # propagate exception
        return False
    
    # ========== Convenience methods ==========
    def sample_space(self, space: Space) -> Any:
        self.rng, sample = space.sample(self.rng)
        return sample
    
    def sample_action(self) -> ActType:
        return self.sample_space(self.action_space)
    
    def sample_observation(self) -> ObsType:
        return self.sample_space(self.observation_space)
    
    def sample_context(self) -> Optional[ContextType]:
        if self.context_space is None:
            return None
        return self.sample_space(self.context_space)
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "Policy":
        return self

    @property
    def prev_wrapper_layer(self) -> Optional["Policy"]:
        return None

    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the attribute `name` exists in the policy."""
        return hasattr(self, name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Gets the attribute `name` from the policy."""
        return getattr(self, name)

    def set_wrapper_attr(self, name: str, value: Any):
        """Sets the attribute `name` on the policy with `value`."""
        setattr(self, name, value)

WrapperBArrayT = TypeVar("WrapperBArrayT")
WrapperContextT = TypeVar("WrapperContextT")
WrapperObsT = TypeVar("WrapperObsT")
WrapperActT = TypeVar("WrapperActT")
WrapperBDeviceT = TypeVar("WrapperBDeviceT")
WrapperBDtypeT = TypeVar("WrapperBDtypeT")
WrapperBRngT = TypeVar("WrapperBRngT")
WrapperRenderFrame = TypeVar("WrapperRenderFrame")
class PolicyWrapper(
    Policy[WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT],
    Generic[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT
    ]
):
    @property
    def metadata(self) -> Dict[str, Any]:
        if hasattr(self, "_metadata") and self._metadata is not None:
            return self._metadata
        return self.policy.metadata
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value

    @property
    def backend(self) -> ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]:
        return self.policy.backend
    
    @property
    def device(self) -> Optional[WrapperBDeviceT]:
        return self.policy.device
    
    @property
    def batch_size(self) -> Optional[int]:
        return self.policy.batch_size
    
    @property
    def action_space(self) -> Space[WrapperActT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]:
        if hasattr(self, "_action_space") and self._action_space is not None:
            return self._action_space
        return self.policy.action_space
    
    @action_space.setter
    def action_space(self, value: Space[WrapperActT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]):
        self._action_space = value

    @property
    def observation_space(self) -> Space[WrapperObsT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]:
        if hasattr(self, "_observation_space") and self._observation_space is not None:
            return self._observation_space
        return self.policy.observation_space
    
    @observation_space.setter
    def observation_space(self, value: Space[WrapperObsT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]):
        self._observation_space = value

    @property
    def context_space(self) -> Optional[Space[WrapperContextT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]]:
        if hasattr(self, "_context_space"):
            return self._context_space
        return self.policy.context_space

    @context_space.setter
    def context_space(self, value: Optional[Space[WrapperContextT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]]):
        self._context_space = value

    @property
    def rng(self) -> WrapperBRngT:
        if hasattr(self, "_rng") and self._rng is not None:
            return self._rng
        
        return self.policy.rng
    
    @rng.setter
    def rng(self, value: WrapperBRngT):
        if hasattr(self, "_rng"):
            self._rng = value
        else:
            self.policy.rng = value

    def __init__(
        self,
        policy: Policy[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]
    ):
        self.policy = policy

    def reset(self, context, observation, *, seed = None, mask = None, **kwargs):
        return self.policy.reset(context, observation, seed=seed, mask=mask, **kwargs)

    def observe_transition(self, action_taken, new_observation):
        return self.policy.observe_transition(action_taken, new_observation)

    def compute_action(self, observation, **kwargs):
        return self.policy.compute_action(observation, **kwargs)

    def visualize_action(self, environment_render, observation, action, **kwargs):
        return self.policy.visualize_action(environment_render, observation, action, **kwargs)

    def close(self):
        return self.policy.close()
    
    def __str__(self):
        return f"<{type(self).__name__}{self.policy}>"
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "Policy":
        return self.policy.unwrapped

    @property
    def prev_wrapper_layer(self) -> Optional["Policy"]:
        return self.policy

    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the attribute `name` exists in the policy."""
        return hasattr(self, name) or self.policy.has_wrapper_attr(name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Gets the attribute `name` from the policy."""
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.policy.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        """Sets the attribute `name` on the policy with `value`."""
        sub_env = self
        attr_set = False
        while not attr_set and sub_env is not None:
            if hasattr(sub_env, name):
                setattr(sub_env, name, value)
                attr_set = True
            else:
                sub_env = sub_env.prev_wrapper_layer
        
        if not attr_set and sub_env is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )