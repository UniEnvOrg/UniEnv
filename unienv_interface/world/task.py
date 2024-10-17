from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type
from abc import ABC, abstractmethod
from ..space import Space, Dict as DictSpace
from ..backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from ..env_base.env import RewardType, TerminationType
from ..env_base.funcenv import FuncEnvCommonState, FuncEnv, StateType


class Task(ABC, Generic[RewardType, TerminationType]):
    observation_space : Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]

    @abstractmethod
    def update(self, last_step_elapsed : float) -> None:
        """Update the task with a new state (e.g. from the environment)."""
        raise NotImplementedError
    
    @abstractmethod
    def control_update(self, last_control_step_elapsed : float) -> None:
        """Update the task with a new state (e.g. from the environment)."""
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> Optional[Dict[str, Any]]:
        """Get the optional Obs from the task. This is called after control_update or reset / init."""
        raise NotImplementedError

    @abstractmethod
    def get_reward(self) -> RewardType:
        """Calculate the reward of the task."""
        raise NotImplementedError

    @abstractmethod
    def get_termination(self) -> TerminationType:
        """Check if the task is terminated."""
        raise NotImplementedError
    
    @abstractmethod
    def get_truncation(self) -> TerminationType:
        """Check if the task is truncated."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the task, this is called after the environment / actor / sensor is reset."""
        raise NotImplementedError
    
    @abstractmethod
    def close(self) -> None:
        """Close the task."""
        raise NotImplementedError

TaskStateT = TypeVar("TaskStateT")
class FuncTask(
    ABC,
    Generic[StateType, TaskStateT, RewardType, TerminationType]
):
    observation_space : Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]
    @abstractmethod
    def initial(
        self, 
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        *, 
        seed : int,
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        TaskStateT
    ]:
        """Initial state."""
        raise NotImplementedError
    
    @abstractmethod
    def reset(
        self, 
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        task_state : TaskStateT
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        TaskStateT
    ]:
        """Reset the task."""
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        task_state : TaskStateT,
        last_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        TaskStateT
    ]:
        """Transition."""
        raise NotImplementedError
    
    @abstractmethod
    def control_step(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        task_state : TaskStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        TaskStateT,
        RewardType,
        TerminationType,
        TerminationType,
    ]:
        """Transition."""
        raise NotImplementedError

    def get_data(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        task_state : TaskStateT
    ) -> Optional[Dict[str, Any]]:
        """Get the data."""
        return None

    def close(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        task_state : TaskStateT
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType]
    ]:
        """Close the task."""
        return state, common_state