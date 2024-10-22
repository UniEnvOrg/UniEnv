from typing import Optional, Any, Dict, Callable, Tuple
from unienv_interface.space import Space, Dict as DictSpace
from ..task import StateType, Task, TaskStateT, FuncTask, RewardType, TerminationType, FuncEnvCommonState
from ..world import FuncWorld

class LambdaTask(Task[RewardType, TerminationType]):
    def __init__(
        self,
        observation_space : Optional[DictSpace[Any, Any, Any]],
        observation_fn : Optional[Callable[[], Dict[str, Any]]],
        reward_fn : Callable[[], RewardType],
        termination_fn : Callable[[], TerminationType],
        truncation_fn : Callable[[], TerminationType],
    ):
        assert observation_space is None or not observation_fn is None, "observation_fn must be provided if observation_space is provided"
        self.observation_space = observation_space
        self.observation_fn = observation_fn
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        self.truncation_fn = truncation_fn
    
    def get_data(self) -> Dict[str, Any] | None:
        if self.observation_fn is None or self.observation_space is None:
            return None
        return self.observation_fn()

    def get_reward(self) -> RewardType:
        return self.reward_fn()

    def get_termination(self) -> TerminationType:
        return self.termination_fn()
    
    def get_truncation(self) -> TerminationType:
        return self.truncation_fn()
    
class LambdaFuncTask(
    FuncTask[StateType, None, RewardType, TerminationType],
):
    def __init__(
        self,
        observation_space : Optional[DictSpace[Any, Any, Any]],
        observation_fn : Optional[Callable[[StateType, FuncEnvCommonState], Dict[str, Any]]],
        control_step_fn : Callable[[StateType, FuncEnvCommonState, Any, float], Tuple[RewardType, TerminationType, TerminationType]],
    ):
        assert observation_space is None or not observation_fn is None, "observation_fn must be provided if observation_space is provided"
        self.observation_space = observation_space
        self.observation_fn = observation_fn
        self.control_step_fn = control_step_fn
    
    def initial(
        self, 
        world : FuncWorld[StateType, Any, Any],
        state : StateType,
        common_state : FuncEnvCommonState,
        *, 
        seed : int,
    ) -> Tuple[
        StateType,
        FuncEnvCommonState,
        None
    ]:
        return state, common_state, None
    
    def reset(
        self,
        world : FuncWorld[StateType, Any, Any],
        state : StateType,
        common_state : FuncEnvCommonState,
        task_state : None
    ) -> Tuple[
        StateType,
        FuncEnvCommonState,
        None
    ]:
        return state, common_state, None
    
    def control_step(
        self,
        state : StateType,
        common_state : FuncEnvCommonState,
        observation : Any,
        task_state : None,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState,
        None,
        RewardType,
        TerminationType,
        TerminationType,
    ]:
        return (state, common_state, None, *self.control_step_fn(state, common_state, observation, last_control_step_elapsed))

    def get_data(
        self, 
        state: StateType, 
        common_state: FuncEnvCommonState, 
        task_state: None
    ) -> Dict[str, Any] | None:
        if self.observation_fn is None:
            return None
        return self.observation_fn(state, common_state)
    