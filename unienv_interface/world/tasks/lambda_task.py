from typing import Optional, Any, Dict, Callable, Tuple
from unienv_interface.space import Space, Dict as DictSpace
from ..task import StateType, Task, TaskStateT, FuncTask, RewardType, TerminationType, FuncEnvCommonState
from ..world import FuncWorld
from ..actor import FuncActor, FuncActorCombinedState

class LambdaTask(Task[RewardType, TerminationType]):
    def __init__(
        self,
        reward_fn : Callable[[], RewardType],
        termination_fn : Callable[[], TerminationType],
        truncation_fn : Callable[[], TerminationType],
        observation_space : Optional[DictSpace[Any, Any, Any]] = None,
        observation_fn : Optional[Callable[[], Dict[str, Any]]] = None,
        context_space : Optional[DictSpace[Any, Any, Any]] = None,
        context_fn : Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        assert observation_space is None or not observation_fn is None, "observation_fn must be provided if observation_space is provided"
        assert context_space is None or not context_fn is None, "context_fn must be provided if context_space is provided"

        self.observation_space = observation_space
        self.observation_fn = observation_fn
        self.context_space = context_space
        self.context_fn = context_fn
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        self.truncation_fn = truncation_fn
    
    def get_data(self) -> Dict[str, Any] | None:
        if self.observation_fn is None or self.observation_space is None:
            return None
        return self.observation_fn()
    
    def get_context(self) -> Dict[str, Any] | None:
        if self.context_fn is None or self.context_space is None:
            return None
        return self.context_fn()

    def get_reward(self) -> RewardType:
        return self.reward_fn()

    def get_termination(self) -> TerminationType:
        return self.termination_fn()
    
    def get_truncation(self) -> TerminationType:
        return self.truncation_fn()
    
class LambdaFuncTask(
    FuncTask[StateType, Any, None, RewardType, TerminationType],
):
    def __init__(
        self,
        control_step_fn : Callable[[StateType, FuncEnvCommonState, Any, Any, float], Tuple[RewardType, TerminationType, TerminationType]],
        observation_space : Optional[DictSpace[Any, Any, Any]] = None,
        observation_fn : Optional[Callable[[StateType, FuncEnvCommonState, Any], Dict[str, Any]]] = None,
        context_space : Optional[DictSpace[Any, Any, Any]] = None,
        context_fn : Optional[Callable[[StateType, FuncEnvCommonState, Any], Dict[str, Any]]] = None,
    ):
        assert observation_space is None or not observation_fn is None, "observation_fn must be provided if observation_space is provided"
        assert context_space is None or not context_fn is None, "context_fn must be provided if context_space is provided"

        self.observation_space = observation_space
        self.observation_fn = observation_fn
        self.context_space = context_space
        self.context_fn = context_fn
        self.control_step_fn = control_step_fn
    
    def initial(
        self, 
        world : FuncWorld[StateType, Any, Any, Any],
        actor : FuncActor[StateType, Any, Any, Any, Any],
        state : StateType,
        common_state : FuncEnvCommonState,
        actor_state : FuncActorCombinedState[Any],
        *, 
        seed : int,
    ) -> Tuple[
        StateType,
        FuncEnvCommonState,
        FuncActorCombinedState[Any],
        None,
        Optional[Dict[str, Any]]
    ]:
        if self.context_fn is not None:
            context = self.context_fn(state, common_state)
        else:
            context = None
        
        return state, common_state, actor_state, None, context
    
    def reset(
        self,
        world : FuncWorld[StateType, Any, Any, Any],
        actor : FuncActor[StateType, Any, Any, Any, Any],
        state : StateType,
        common_state : FuncEnvCommonState,
        actor_state : FuncActorCombinedState[Any],
        task_state : None
    ) -> Tuple[
        StateType,
        FuncEnvCommonState,
        FuncActorCombinedState[Any],
        None,
        Optional[Dict[str, Any]]
    ]:
        return self.initial(
            world,
            actor,
            state,
            common_state,
            actor_state,
            seed=0
        )
    
    def control_step(
        self,
        state : StateType,
        common_state : FuncEnvCommonState,
        actor_state : FuncActorCombinedState[Any],
        observation : Any,
        task_state : None,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState,
        FuncActorCombinedState[Any],
        None,
        RewardType,
        TerminationType,
        TerminationType,
    ]:
        return (state, common_state, actor_state, None, *self.control_step_fn(state, common_state, actor_state, observation, last_control_step_elapsed))

    def get_data(
        self, 
        state: StateType, 
        common_state: FuncEnvCommonState, 
        actor_state: FuncActorCombinedState[Any],
        observation : Any,
        task_state : None,
        last_control_step_elapsed : float
    ) -> Dict[str, Any] | None:
        if self.observation_fn is None:
            return None
        return self.observation_fn(state, common_state, actor_state)
    