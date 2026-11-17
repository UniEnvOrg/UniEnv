from typing import Any, TypeVar, Generic, Optional, Tuple, Union, Dict
import abc

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space

from .policy import Policy, ContextType, ObsType, ActType, RenderFrame, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT

__all__ = [
    "FuncPolicy",
    "FuncPolicyWrapper",
    "FuncPolicyBasedPolicy",
    "StateType",
    "BArrayType",
    "ContextType",
    "ObsType",
    "ActType",
    "RenderFrame",
    "BDeviceType",
    "BDtypeType",
    "BRNGType",
    "WrapperStateT",
    "WrapperBArrayT",
    "WrapperContextT",
    "WrapperObsT",
    "WrapperActT",
    "WrapperRenderFrame",
    "WrapperBDeviceT",
    "WrapperBDtypeT",
    "WrapperBRngT",
]

StateType = TypeVar("StateType")

class FuncPolicy(
    abc.ABC,
    Generic[StateType, BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]
):
    metadata : Dict[str, Any] = {}

    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType] = None

    action_space : Space[ActType, BDeviceType, BDtypeType, BRNGType]
    observation_space : Space[ObsType, BDeviceType, BDtypeType, BRNGType]
    context_space : Optional[Space[ContextType, BDeviceType, BDtypeType, BRNGType]] = None

    @abc.abstractmethod
    def initial(
        self, 
        context : Optional[ContextType],
        observation : ObsType,
        *,
        batch_size : Optional[int],
        seed : Optional[int] = None,
        **kwargs
    ) -> StateType:
        raise NotImplementedError
    
    def reset(
        self,
        state : StateType,
        context : Optional[ContextType],
        observation : ObsType,
        *,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> StateType:
        return state
    
    def observe_transition(
        self,
        state : StateType,
        action_taken : ActType,
        new_observation : ObsType,
    ) -> StateType:
        return state

    @abc.abstractmethod
    def compute_action(
        self,
        state : StateType,
        observation : ObsType,
        **kwargs
    ) -> Tuple[StateType, ActType]:
        raise NotImplementedError
    
    def visualize_action(
        self,
        state : StateType,
        environment_render : RenderFrame,
        observation : ObsType,
        action : ActType,
        **kwargs
    ) -> RenderFrame:
        """
        Visualizes the action on the environment render frame.
        """
        return environment_render
    
    def close(self, state : StateType) -> None:
        pass

    def __str__(self):
        return f"<{type(self).__name__} instance>"

    def __repr__(self):
        return str(self)

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncPolicy":
        return self

    @property
    def prev_wrapper_layer(self) -> Optional["FuncPolicy"]:
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

WrapperStateT = TypeVar("WrapperStateT")
class FuncPolicyWrapper(
    FuncPolicy[WrapperStateT, WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT],
    Generic[
        StateType, BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        WrapperStateT, WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT
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

    def __init__(
        self,
        policy : FuncPolicy[StateType, BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]
     ):
        self.policy = policy
 
    def initial(self, context, observation, *, batch_size, seed = None, **kwargs):
        return self.policy.initial(context, observation, batch_size=batch_size, seed=seed, **kwargs)
    
    def reset(self, state, context, observation, *, seed = None, mask = None, **kwargs):
        return self.policy.reset(state, context, observation, seed=seed, mask=mask, **kwargs)

    def observe_transition(self, state, action_taken, new_observation):
        return self.policy.observe_transition(state, action_taken, new_observation)

    def compute_action(self, state, observation, **kwargs):
        return self.policy.compute_action(state, observation, **kwargs)
    
    def visualize_action(
        self,
        state : WrapperStateT,
        environment_render : WrapperRenderFrame,
        observation : WrapperObsT,
        action : WrapperActT,
        **kwargs
    ) -> WrapperRenderFrame:
        return self.policy.visualize_action(state, environment_render, observation, action, **kwargs)
    
    def close(self, state : WrapperStateT) -> None:
        return self.policy.close(state)
    
    def __str__(self):
        return f"<{type(self).__name__}{self.policy}>"

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncPolicy":
        return self.policy.unwrapped

    @property
    def prev_wrapper_layer(self) -> Optional["FuncPolicy"]:
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
    
class FuncPolicyBasedPolicy(
    Generic[StateType, BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
    Policy[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]
):
    def __init__(
        self,
        func_policy : FuncPolicy[StateType, BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        batch_size : Optional[int] = None,
        seed : Optional[int] = None,
    ) -> None:
        self.func_policy = func_policy
        
        self._batch_size = batch_size
        self.state : Optional[StateType] = None
        self._init_seed = seed
        self._inited = False

        self._metadata : Optional[Dict[str, Any]] = None

        self.rng = func_policy.backend.random.random_number_generator(seed, device=func_policy.device)
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata if self._metadata is not None else self.func_policy.metadata
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.func_policy.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.func_policy.device
    
    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size
    
    @property
    def action_space(self) -> Space[ActType, BDeviceType, BDtypeType, BRNGType]:
        return self.func_policy.action_space
    
    @property
    def observation_space(self) -> Space[ObsType, BDeviceType, BDtypeType, BRNGType]:
        return self.func_policy.observation_space
    
    @property
    def context_space(self) -> Optional[Space[ContextType, BDeviceType, BDtypeType, BRNGType]]:
        return self.func_policy.context_space
    
    def reset(self, context, observation, *, seed = None, mask = None, **kwargs):
        if not self._inited:
            assert mask is None, "Cannot provide mask on first reset"
            self.state = self.func_policy.initial(
                context,
                observation,
                batch_size=self.batch_size,
                seed=self._init_seed if seed is None else seed,
                **kwargs
            )
            self._inited = True
        else:
            self.state = self.func_policy.reset(
                self.state,
                context,
                observation,
                seed=seed,
                mask=mask,
                **kwargs
            )
    
    def observe_transition(self, action_taken, new_observation):
        self.state = self.func_policy.observe_transition(
            self.state,
            action_taken,
            new_observation
        )

    def compute_action(self, observation, **kwargs):
        assert self._inited, "Policy must be reset before computing action"
        self.state, action = self.func_policy.compute_action(
            self.state,
            observation,
            **kwargs
        )
        return action
    
    def visualize_action(
        self,
        environment_render : RenderFrame,
        observation : ObsType,
        action : ActType,
        **kwargs
    ) -> RenderFrame:
        assert self._inited, "Policy must be reset before visualizing action"
        return self.func_policy.visualize_action(
            self.state,
            environment_render,
            observation,
            action,
            **kwargs
        )
    
    def close(self):
        if self._inited:
            self.func_policy.close(self.state)
            self.state = None
            self._inited = False
    
    def __str__(self):
        return f"<{type(self).__name__}({self.func_policy})>"
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "Policy":
        return self

    @property
    def prev_wrapper_layer(self) -> Optional["Policy"]:
        return None

    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the attribute `name` exists in the policy."""
        return hasattr(self, name) or self.func_policy.has_wrapper_attr(name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Gets the attribute `name` from the policy."""
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.func_policy.get_wrapper_attr(name)
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
                if sub_env == self:
                    sub_env = self.func_policy
                else:
                    sub_env = sub_env.prev_wrapper_layer
        
        if not attr_set and sub_env is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )