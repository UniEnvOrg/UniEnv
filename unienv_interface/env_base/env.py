from copy import deepcopy
from typing import Any, Generic, SupportsFloat, TypeVar, Optional, Union, Dict, Tuple, Sequence, Type
import abc
from ..space import Space
from ..backends import ComputeBackend
import numpy as np

ObsType = TypeVar("ObsType")
ContextType = TypeVar("ContextType")
ActType = TypeVar("ActType")
RewardType = TypeVar("RewardType", bound=SupportsFloat)
TerminationType = TypeVar("TerminationType")
RenderFrame = TypeVar("RenderFrame")
BDeviceT = TypeVar("BDeviceT")
BRngT = TypeVar("BRngT")

class Env(abc.ABC, Generic[ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT]):
    # metadata of the environment
    metadata: dict[str, Any] = {
        "render_modes": []
    }

    # The render mode and fps of the environment
    render_mode: Optional[str] = None
    render_fps : Optional[int] = None

    backend : ComputeBackend[Any, BDeviceT, Any, BRngT]
    device : Optional[BDeviceT]

    batch_size : Optional[int] = None

    action_space: Space[ActType, Any, BDeviceT, Any, BRngT]
    observation_space: Space[ObsType, Any, BDeviceT, Any, BRngT]
    context_space: Optional[Space[ContextType, Any, BDeviceT, Any, BRngT]] = None

    np_rng : np.random.Generator = None
    rng : BRngT = None

    @abc.abstractmethod
    def step(
        self, action: ActType
    ) -> Tuple[ObsType, RewardType, TerminationType, TerminationType, Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def render(self) -> RenderFrame | Sequence[RenderFrame] | None:
        raise NotImplementedError

    def close(self):
        pass

    def __str__(self):
        return f"<{type(self).__name__} instance>"

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args: Any):
        """Support with-statement for the environment and closes the environment."""
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
    
    # ========== Wrapper methods ==========

    @property
    def unwrapped(self) -> "Env":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["Env"]:
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