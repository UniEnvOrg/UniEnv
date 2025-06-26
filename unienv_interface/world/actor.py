from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Protocol, Iterable, Union
from abc import ABC, abstractmethod
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, DictSpace
from .sensor import Sensor
from dataclasses import dataclass

class Actor(ABC, Sensor[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    action_space : Space[Any, BDeviceType, BDtypeType, BRNGType] = None

    @abstractmethod
    def set_next_action(self, action: Any) -> None:
        """
        Update the next action to be taken by the actor.
        This method should be called before the actor takes a step in the environment.
        If this method is not called after a world step, the actor will compute a dummy action to try retain the same state of the robot.
        """
        raise NotImplementedError

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "Actor":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["Actor"]:
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

ActorWrapperBArrayType = TypeVar("ActorWrapperBArrayType")
ActorWrapperBDeviceType = TypeVar("ActorWrapperBDeviceType")
ActorWrapperBDtypeType = TypeVar("ActorWrapperBDtypeType")
ActorWrapperBRNGType = TypeVar("ActorWrapperBRNGType")

class ActorWrapper(
    Generic[
        ActorWrapperBArrayType, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType,
        BArrayType, BDeviceType, BDtypeType, BRNGType
    ],
    Actor[ActorWrapperBArrayType, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]
):
    def __init__(
        self,
        actor : Actor[BArrayType, BDeviceType, BDtypeType, BRNGType]
    ):
        self.actor = actor
        self._action_space : Optional[Space[Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]] = None
        self._observation_space : Optional[Space[Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]] = None
    
    @property
    def observation_space(self) -> Space[Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]:
        if self._observation_space is None:
            return self.actor.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: Space[Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]) -> None:
        self._observation_space = value
    
    @property
    def action_space(self) -> Space[Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]:
        if self._action_space is None:
            return self.actor.action_space
        return self._action_space
    
    @action_space.setter
    def action_space(self, value: Space[Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]) -> None:
        self._action_space = value

    @property
    def control_timestep(self) -> float:
        return self.actor.control_timestep
    
    @control_timestep.setter
    def control_timestep(self, value : float) -> None:
        self.actor.control_timestep = value
    
    def get_observation(self):
        return self.actor.get_observation()

    def set_next_action(self, action) -> None:
        self.actor.set_next_action(action)
    
    def pre_environment_step(self, dt) -> None:
        self.actor.pre_environment_step(dt)

    def update(self, dt) -> None:
        self.actor.update(dt)

    def reset(self) -> None:
        self.actor.reset()
    
    def close(self) -> None:
        self.actor.close()
    
    # ========== Wrapper methods ==========

    @property
    def unwrapped(self) -> "Actor":
        return self.actor.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> "Actor[BDeviceType, BDtypeType, BRNGType]":
        return self.actor
    
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or self.actor.has_wrapper_attr(name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.actor.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        sub_actor = self
        attr_set = False

        while attr_set is False and sub_actor is not None:
            if hasattr(sub_actor, name):
                setattr(sub_actor, name, value)
                attr_set = True
            else:
                sub_actor = sub_actor.prev_wrapper_layer

        if attr_set is False and sub_actor is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )
