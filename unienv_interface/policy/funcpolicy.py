from typing import Any, TypeVar, Generic, Optional, Tuple, Union, Dict
import abc

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from unienv_interface.env_base._utils import set_attr_on_first_layer, settable_here

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
    """A policy with explicit state passing, the functional counterpart of :class:`Policy`.

    Every lifecycle method takes the state and returns the updated one, mirroring
    :class:`~unienv_interface.env_base.funcenv.FuncEnv`::

        state = policy.initial(context, observation)
        while running:
            state, action = policy.get_action(state)
            observation, ... = env.step(action)
            state = policy.observe_transition(state, action, observation, dt)   # dt: seconds the step took
            # some sub-environments done: state = policy.reset(state, full_context, full_observation, mask=done)

    The policy state takes in observations only through :meth:`initial`, :meth:`reset` and
    :meth:`observe_transition`. :meth:`get_action` returns the state too, but may only update
    bookkeeping in it (e.g. advance an RNG key or cache the computed action). The context is only
    given at the start of episodes, and batching follows ``FuncEnv``: when ``batch_size`` is not
    ``None`` the spaces include the leading batch dimension and all data is a full batch.
    """
    metadata : Dict[str, Any] = {}

    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType] = None

    # ``None`` → unbatched; ``int`` → batched (spaces include the leading batch axis, even for 1)
    batch_size : Optional[int] = None

    action_space : Space[ActType, BDeviceType, BDtypeType, BRNGType]
    observation_space : Space[ObsType, BDeviceType, BDtypeType, BRNGType]
    context_space : Optional[Space[ContextType, BDeviceType, BDtypeType, BRNGType]] = None

    @abc.abstractmethod
    def initial(
        self, 
        context : Optional[ContextType],
        observation : ObsType,
        *,
        seed : Optional[int] = None,
        **kwargs
    ) -> StateType:
        """The state at the start of the first episodes (all sub-environments)."""
        raise NotImplementedError

    @abc.abstractmethod
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
        """Start new episodes from ``context`` and ``observation`` (full batch, as in :meth:`Policy.reset`).

        With ``mask`` (batched only), only the selected sub-environments start over; the state of
        the others must be kept.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def observe_transition(
        self,
        state : StateType,
        action_taken : ActType,
        new_observation : ObsType,
        dt : float,
    ) -> StateType:
        """The state after the environment executed ``action_taken`` and returned ``new_observation``.

        ``dt`` is the env time in seconds between the previous observation and
        ``new_observation``, i.e. how long ``action_taken`` was applied (sim time in simulation,
        elapsed time on a robot; one value for a batch). Policies that schedule work in time
        (e.g. asynchronous action chunking) rely on it; others may ignore it.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(
        self,
        state : StateType,
        **kwargs
    ) -> Tuple[StateType, ActType]:
        """The action for ``state``, and ``state`` with updated bookkeeping (e.g. RNG key) only."""
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

    def __init__(
        self,
        policy : FuncPolicy[StateType, BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]
     ):
        self.policy = policy
 
    def initial(self, context, observation, *, seed = None, **kwargs):
        return self.policy.initial(context, observation, seed=seed, **kwargs)
    
    def reset(self, state, context, observation, *, seed = None, mask = None, **kwargs):
        return self.policy.reset(state, context, observation, seed=seed, mask=mask, **kwargs)

    def observe_transition(self, state, action_taken, new_observation, dt):
        return self.policy.observe_transition(state, action_taken, new_observation, dt)

    def get_action(self, state, **kwargs):
        return self.policy.get_action(state, **kwargs)
    
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
        # Layers that only forward ``name`` through a read-only property (e.g. ``batch_size``)
        # are skipped: the value is set on the layer that owns it.
        set_attr_on_first_layer(self, name, value, type(self).__name__)
    
class FuncPolicyBasedPolicy(
    Generic[StateType, BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
    Policy[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType]
):
    """Runs a :class:`FuncPolicy` behind the stateful :class:`Policy` interface, holding its state.

    The batch size and spaces are the functional policy's. ``seed`` is used for the first
    :meth:`reset` (unless it passes its own) and for the RNG of the ``sample_*`` helpers.
    """
    def __init__(
        self,
        func_policy : FuncPolicy[StateType, BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        seed : Optional[int] = None,
    ) -> None:
        self.func_policy = func_policy
        
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
        return self.func_policy.batch_size
    
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
    
    def observe_transition(self, action_taken, new_observation, dt):
        assert self._inited, "Policy must be reset before observing transitions"
        self.state = self.func_policy.observe_transition(
            self.state,
            action_taken,
            new_observation,
            dt,
        )

    def get_action(self, **kwargs):
        assert self._inited, "Policy must be reset before getting an action"
        self.state, action = self.func_policy.get_action(
            self.state,
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
        # Attributes this policy only forwards through a read-only property (e.g. ``batch_size``)
        # are set on the functional policy stack that owns them.
        if settable_here(self, name):
            setattr(self, name, value)
        else:
            set_attr_on_first_layer(self.func_policy, name, value, type(self).__name__)