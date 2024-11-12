from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, SupportsFloat
from abc import ABC, abstractmethod
from unienv_interface.utils import seed_util
from .world import FuncWorld, StateType
from .actor import FuncActor, ActorStateT, FuncActorCombinedState
from ..space import Space, Dict as DictSpace
from ..backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

class Task(ABC, Generic[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    observation_space : Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]
    context_space : Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]

    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType]

    def update(self, last_step_elapsed : float) -> None:
        """Update the task with a new state (e.g. from the environment)."""
        pass
    
    def control_update(self, last_control_step_elapsed : float) -> None:
        """Update the task with a new state (e.g. from the environment)."""
        pass

    @abstractmethod
    def get_data(self) -> Optional[Dict[str, Any]]:
        """Get the optional Obs from the task. This is called after control_update or reset / init."""
        raise NotImplementedError
    
    @abstractmethod
    def get_context(self) -> Optional[Dict[str, Any]]:
        """Get the context of the task."""
        raise NotImplementedError

    @abstractmethod
    def get_reward(self) -> Union[SupportsFloat, BArrayType]:
        """Calculate the reward of the task."""
        raise NotImplementedError

    @abstractmethod
    def get_termination(self) -> Union[bool, BArrayType]:
        """Check if the task is terminated."""
        raise NotImplementedError
    
    @abstractmethod
    def get_truncation(self) -> Union[bool, BArrayType]:
        """Check if the task is truncated."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the task, this is called after the environment / actor / sensor is reset."""
        pass
    
    def close(self) -> None:
        """Close the task."""
        pass

    def __del__(self):
        self.close()

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "Task":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["Task"]:
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

TaskStateT = TypeVar("TaskStateT")
class FuncTask(
    ABC,
    Generic[StateType, ActorStateT, TaskStateT, BArrayType, BDeviceType, BDtypeType, BRNGType]
):
    observation_space : Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]
    context_space : Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]

    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType]

    @abstractmethod
    def initial(
        self, 
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        actor : FuncActor[StateType, ActorStateT, BDeviceType, BDtypeType, BRNGType],
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[ActorStateT],
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT],
        TaskStateT,
        Optional[Dict[str, Any]] # Optional Context
    ]:
        """Initial state."""
        raise NotImplementedError
    
    def reset(
        self, 
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        actor : FuncActor[StateType, ActorStateT, BDeviceType, BDtypeType, BRNGType],
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[ActorStateT],
        task_state : TaskStateT
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT],
        TaskStateT,
        Optional[Dict[str, Any]] # Optional Context
    ]:
        return self.initial(
            world,
            actor,
            state,
            rng,
            actor_state
        )
    
    def step(
        self,
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[ActorStateT],
        task_state : TaskStateT,
        last_step_elapsed : float
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT],
        TaskStateT
    ]:
        return state, rng, actor_state, task_state
    
    @abstractmethod
    def control_step(
        self,
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[ActorStateT],
        observation : Any,
        task_state : TaskStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT],
        TaskStateT,
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
    ]:
        """Transition."""
        raise NotImplementedError

    def get_data(
        self,
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[ActorStateT],
        observation : Any,
        task_state : TaskStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT],
        TaskStateT,
        Optional[Dict[str, Any]]
    ]:
        """Get the data."""
        return state, rng, actor_state, task_state, None

    def close(
        self,
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[ActorStateT],
        task_state : TaskStateT
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT],
    ]:
        """Close the task."""
        return state, rng, actor_state
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncTask":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["FuncTask"]:
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


TaskWrapperBArrayT = TypeVar("TaskWrapperBArrayType")
TaskWrapperBDeviceT = TypeVar("TaskWrapperBDeviceType")
TaskWrapperBDtypeT = TypeVar("TaskWrapperBDtypeType")
TaskWrapperBRNGT = TypeVar("TaskWrapperBRNGType")

class TaskWrapper(
    Generic[
        TaskWrapperBArrayT, TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT,
        BArrayType, BDeviceType, BDtypeType, BRNGType,
    ],
    Task[TaskWrapperBArrayT, TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]
):
    def __init__(
        self,
        task : Task[BArrayType, BDeviceType, BDtypeType, BRNGType]
    ):
        self.task = task
        self._observation_space : Optional[DictSpace[
            TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT
        ]] = self.task.observation_space
        self._context_space : Optional[DictSpace[
            TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT
        ]] = self.task.context_space
    
    @property
    def observation_space(self) -> Optional[DictSpace[TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]]:
        return self._observation_space
    
    @observation_space.setter
    def observation_space(self, value : Optional[DictSpace[TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]]) -> None:
        self._observation_space = value

    @property
    def context_space(self) -> Optional[DictSpace[TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]]:
        return self._context_space
    
    @context_space.setter
    def context_space(self, value : Optional[DictSpace[TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]]) -> None:
        self._context_space = value

    @property
    def backend(self) -> ComputeBackend[TaskWrapperBArrayT, TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]:
        return self.task.backend
    
    @property
    def device(self) -> Optional[TaskWrapperBDeviceT]:
        return self.task.device

    def update(self, last_step_elapsed : float) -> None:
        self.task.update(last_step_elapsed)
    
    def control_update(self, last_control_step_elapsed : float) -> None:
        self.task.control_update(last_control_step_elapsed)
    
    def get_data(self) -> Optional[Dict[str, Any]]:
        return self.task.get_data()
    
    def get_context(self) -> Optional[Dict[str, Any]]:
        return self.task.get_context()

    def get_reward(self) -> Union[SupportsFloat, TaskWrapperBArrayT]:
        return self.task.get_reward()
    
    def get_termination(self) -> Union[bool, TaskWrapperBArrayT]:
        return self.task.get_termination()
    
    def get_truncation(self) -> Union[bool, TaskWrapperBArrayT]:
        return self.task.get_truncation()
    
    def reset(self) -> None:
        self.task.reset()
    
    def close(self) -> None:
        self.task.close()

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "Task":
        return self.task.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> "Task[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        return self.task
    
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or self.task.has_wrapper_attr(name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.task.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        sub_task = self
        attr_set = False

        while attr_set is False and sub_task is not None:
            if hasattr(sub_task, name):
                setattr(sub_task, name, value)
                attr_set = True
            else:
                sub_task = sub_task.prev_wrapper_layer

        if attr_set is False and sub_task is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )

TaskWrapperStateT = TypeVar("TaskWrapperStateT")
class FuncTaskWrapper(
    Generic[
        TaskWrapperStateT, TaskWrapperBArrayT, TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT,
        StateType, ActorStateT, TaskStateT, BArrayType, BDeviceType, BDtypeType, BRNGType,
    ],
    FuncTask[StateType, ActorStateT, TaskWrapperStateT, TaskWrapperBArrayT, TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]
):
    def __init__(
        self,
        task : FuncTask[StateType, ActorStateT, TaskStateT, BArrayType, BDeviceType, BDtypeType, BRNGType]
    ):
        self.task = task
        self._observation_space : Optional[DictSpace[
            TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT
        ]] = self.task.observation_space
        self._context_space : Optional[DictSpace[
            TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT
        ]] = self.task.context_space

    @property
    def observation_space(self) -> Optional[DictSpace[TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]]:
        return self._observation_space
    
    @observation_space.setter
    def observation_space(self, value : Optional[DictSpace[TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]]) -> None:
        self._observation_space = value

    @property
    def context_space(self) -> Optional[DictSpace[TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]]:
        return self._context_space
    
    @context_space.setter
    def context_space(self, value : Optional[DictSpace[TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]]) -> None:
        self._context_space = value

    @property
    def backend(self) -> ComputeBackend[TaskWrapperBArrayT, TaskWrapperBDeviceT, TaskWrapperBDtypeT, TaskWrapperBRNGT]:
        return self.task.backend
    
    @property
    def device(self) -> Optional[TaskWrapperBDeviceT]:
        return self.task.device

    def initial(
        self,
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        actor : FuncActor[StateType, ActorStateT, BDeviceType, BDtypeType, BRNGType],
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[TaskWrapperStateT],
        *args, 
        **kwargs
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT],
        TaskWrapperStateT,
        Optional[Dict[str, Any]]
    ]:
        return self.task.initial(
            world, 
            actor,
            state, 
            rng, 
            actor_state, 
            *args, 
            **kwargs
        )

    def reset(
        self,
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        actor : FuncActor[StateType, ActorStateT, BDeviceType, BDtypeType, BRNGType],
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[ActorStateT],
        task_state : TaskWrapperStateT
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT],
        TaskWrapperStateT,
        Optional[Dict[str, Any]]
    ]:
        return self.task.reset(
            world, 
            actor,
            state, 
            rng,
            actor_state,
            task_state
        )

    def step(
        self,
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[ActorStateT],
        task_state : TaskWrapperStateT,
        last_step_elapsed : float
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT],
        TaskWrapperStateT
    ]:
        return self.task.step(
            state, 
            rng, 
            actor_state,
            task_state, 
            last_step_elapsed
        )
    
    def control_step(
        self,
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[ActorStateT],
        observation : Any,
        task_state : TaskWrapperStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT],
        TaskWrapperStateT,
        Union[SupportsFloat, TaskWrapperBArrayT],
        Union[bool, TaskWrapperBArrayT],
        Union[bool, TaskWrapperBArrayT],
    ]:
        return self.task.control_step(
            state, 
            rng,
            actor_state, 
            observation, 
            task_state, 
            last_control_step_elapsed
        )
    
    def get_data(
        self,
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[ActorStateT],
        observation : Any,
        task_state : TaskWrapperStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT],
        TaskWrapperStateT,
        Optional[Dict[str, Any]]
    ]:
        return self.task.get_data(
            state, 
            rng, 
            actor_state,
            observation,
            task_state, 
            last_control_step_elapsed
        )
    
    def close(
        self,
        state : StateType,
        rng : BRNGType,
        actor_state : FuncActorCombinedState[ActorStateT],
        task_state : TaskWrapperStateT
    ) -> Tuple[
        StateType,
        BRNGType,
        FuncActorCombinedState[ActorStateT]
    ]:
        return self.task.close(state, rng, actor_state, task_state)
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncTask":
        return self.task.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> "FuncTask[StateType, ActorStateT, TaskStateT, RewardType, TerminationType]":
        return self.task
    
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or self.task.has_wrapper_attr(name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.task.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        sub_task = self
        attr_set = False

        while attr_set is False and sub_task is not None:
            if hasattr(sub_task, name):
                setattr(sub_task, name, value)
                attr_set = True
            else:
                sub_task = sub_task.prev_wrapper_layer

        if attr_set is False and sub_task is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )