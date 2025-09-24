from typing import Dict as DictT, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence, TypeVar
import numpy as np
import copy
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.utils import seed_util
from unienv_interface.env_base.funcenv import FuncEnv, ContextType, ObsType, ActType, RenderFrame, StateType, RenderStateType
from unienv_interface.env_base.funcenv_wrapper import *
from unienv_interface.space import Space
from unienv_interface.utils.data_queue import FuncSpaceDataQueue, SpaceDataQueueState
from unienv_interface.utils.stateclass import StateClass, field

class FuncFrameStackWrapperState(
    Generic[StateType], StateClass
):
    env_state : StateType
    obs_queue_state : Optional[SpaceDataQueueState]
    action_queue_state : Optional[SpaceDataQueueState]

class FuncFrameStackWrapper(
    FuncEnvWrapper[
        FuncFrameStackWrapperState[StateType], RenderStateType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        StateType, RenderStateType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        func_env: FuncEnv[
            StateType, RenderStateType,
            BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
        ],
        obs_stack_size: int = 0,
        action_stack_size: int = 0,
        action_default_value: Optional[ActType] = None,
    ):
        assert obs_stack_size >= 0, "Observation stack size must be greater than 0"
        assert action_stack_size >= 0, "Action stack size must be greater than 0"
        assert action_stack_size == 0 or action_default_value is not None, "Action default value must be provided if action stack size is greater than 0"
        assert obs_stack_size > 0 or action_stack_size > 0, "At least one of observation stack size or action stack size must be greater than 0"
        super().__init__(func_env)
        obs_is_dict = isinstance(func_env.observation_space, DictT)
        assert obs_is_dict or action_stack_size == 0, "Action stack size must be 0 if observation space is not a DictSpace"
        
        self.action_stack_size = action_stack_size
        self.obs_stack_size = obs_stack_size

        if action_stack_size > 0:
            self.action_deque = FuncSpaceDataQueue(
                func_env.action_space,
                func_env.batch_size,
                action_stack_size,
            )
            self.action_default_value = action_default_value
        else:
            self.action_deque = None
        
        self.obs_deque = None
        if obs_stack_size > 0:
            self.obs_deque = FuncSpaceDataQueue(
                func_env.observation_space,
                func_env.batch_size,
                obs_stack_size + 1,
            )
            if action_stack_size > 0:
                new_obs_space = copy.copy(self.obs_deque.output_space)
                new_obs_space['past_actions'] = self.action_deque.output_space
                self.observation_space = new_obs_space
            else:
                self.observation_space = self.obs_deque.output_space
        else:
            if action_stack_size > 0:
                self.observation_space = copy.copy(func_env.observation_space)
                self.observation_space['past_actions'] = self.action_deque.output_space
            else:
                raise ValueError("At least one of observation stack size or action stack size must be greater than 0")
    
    def map_observation(
        self, 
        state : FuncFrameStackWrapperState[StateType],
        observation : ObsType
    ) -> ObsType:
        if self.obs_deque is not None:
            observation = self.obs_deque.get_output_data(state.obs_queue_state)
        if self.action_deque is not None:
            stacked_action = self.action_deque.get_output_data(state.action_queue_state)
            observation = copy.copy(observation)
            observation['past_actions'] = stacked_action
        return observation

    def initial(self, *, seed = None, **kwargs):
        init_state, init_context, init_obs, init_info = self.func_env.initial(seed=seed, **kwargs)
        obs_queue_state = None
        action_queue_state = None
        if self.obs_deque is not None:
            obs_queue_state = self.obs_deque.init(init_obs)
        if self.action_deque is not None:
            action_queue_state = self.action_deque.init(self.action_default_value)
        state = FuncFrameStackWrapperState(
            env_state=init_state,
            obs_queue_state=obs_queue_state,
            action_queue_state=action_queue_state,
        )
        return state, init_context, self.map_observation(state, init_obs), init_info

    def reset(self, state, *args, seed = None, mask = None, **kwargs):
        env_state, context, observation, info = self.func_env.reset(
            state.env_state, 
            *args,
            seed=seed,
            mask=mask,
            **kwargs
        )
        obs_queue_state = state.obs_queue_state
        action_queue_state = state.action_queue_state
        if self.obs_deque is not None:
            obs_queue_state = self.obs_deque.reset(
                obs_queue_state,
                initial_data=observation,
                mask=mask
            )
        if self.action_deque is not None:
            action_queue_state = self.action_deque.reset(
                action_queue_state,
                initial_data=self.action_default_value,
                mask=mask
            )
        new_state = FuncFrameStackWrapperState(
            env_state=env_state,
            obs_queue_state=obs_queue_state,
            action_queue_state=action_queue_state,
        )
        return new_state, context, self.map_observation(new_state, observation), info

    def step(self, state, action):
        env_state, observation, reward, terminated, truncated, info = self.func_env.step(state.env_state, action)
        obs_queue_state = state.obs_queue_state
        action_queue_state = state.action_queue_state
        if self.obs_deque is not None:
            obs_queue_state = self.obs_deque.add(obs_queue_state, observation)
        if self.action_deque is not None:
            action_queue_state = self.action_deque.add(action_queue_state, action)
        new_state = FuncFrameStackWrapperState(
            env_state=env_state,
            obs_queue_state=obs_queue_state,
            action_queue_state=action_queue_state,
        )
        return new_state, self.map_observation(new_state, observation), reward, terminated, truncated, info