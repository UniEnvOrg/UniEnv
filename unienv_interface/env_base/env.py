from copy import deepcopy
from typing import Any, Generic, SupportsFloat, TypeVar, Optional, Union, Dict, Tuple, Sequence, Type
import abc
from ..space import Space
from ..backends import ComputeBackend
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RewardType = TypeVar("RewardType", bound=SupportsFloat)
TerminationType = TypeVar("TerminationType")
RenderFrame = TypeVar("RenderFrame")
BDeviceT = TypeVar("BDeviceT")
BRngT = TypeVar("BRngT")

class Env(abc.ABC, Generic[ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT]):
    # metadata of the environment
    metadata: dict[str, Any] = {
        "render_modes": []
    }

    # The render mode and fps of the environment
    render_mode: Optional[str] = None
    render_fps : Optional[int] = None

    backend : Type[ComputeBackend[Any, BDeviceT, Any, BRngT]]
    device : Optional[BDeviceT]

    action_space: Space[ActType, Any, BDeviceT, Any, BRngT]
    observation_space: Space[ObsType, Any, BDeviceT, Any, BRngT]

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
    ) -> Tuple[ObsType, Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def render(self) -> RenderFrame | Sequence[RenderFrame] | None:
        raise NotImplementedError

    def close(self):
        pass

    @property
    @abc.abstractmethod
    def np_rng(self) -> np.random.Generator:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def rng(self) -> BRngT:
        raise NotImplementedError

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

    # ========== Wrapper methods ==========

    @property
    def unwrapped(self) -> "Env":
        return self

    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the attribute `name` exists in the environment."""
        return hasattr(self, name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Gets the attribute `name` from the environment."""
        return getattr(self, name)

    def set_wrapper_attr(self, name: str, value: Any):
        """Sets the attribute `name` on the environment with `value`."""
        setattr(self, name, value)