from typing import Dict as DictT, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence, TypeVar
import numpy as np
import copy

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.utils import seed_util
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.wrapper import ContextObservationWrapper, ActionWrapper, WrapperContextT, WrapperObsT, WrapperActT
from unienv_interface.space import Space, DictSpace
from collections import deque

DataT = TypeVar('DataT')
class SpaceDataQueue(
    Generic[DataT, BArrayType, BDeviceType, BDtypeType, BRNGType]
):
    def __init__(
        self,
        space : Space[DataT, BDeviceType, BDtypeType, BRNGType],
        batch_size : Optional[int],
        maxlen: int,
    ) -> None:
        assert maxlen > 0, "Max length must be greater than 0"
        assert batch_size is None or batch_size > 0, "Batch size must be greater than 0 if provided"
        assert batch_size is None or sbu.batch_size(space) == batch_size, "Batch size must match the space's batch size if provided"
        self.space = space
        self.single_space = space
        self.stacked_space = sbu.batch_space(space, maxlen) # (H, ...) or (L, B, ...)
        self.output_space = sbu.swap_batch_dims(
            self.stacked_space, 0, 1
        ) if batch_size is not None else self.stacked_space # (B, L, ...) or (H, ...)
        self.data = self.stacked_space.create_empty()
        self._maxlen = maxlen
        self._batch_size = batch_size

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def backend(self) -> ComputeBackend:
        return self.space.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.space.device
    
    def reset(
        self, 
        initial_data : DataT,
        mask : Optional[BArrayType] = None,
    ) -> None:
        assert self.batch_size is None or mask is None, \
            "Mask should not be provided if batch size is empty"
        index = (
            slice(None), mask
        ) if mask is not None else slice(None)
        
        expanded_data = sbu.get_at( # Add a singleton horizon dimension to the data
            self.space,
            initial_data,
            None
        )
        self.data = sbu.set_at(
            self.stacked_space,
            self.data,
            index,
            expanded_data
        )

    def append(self, data : DataT) -> None:
        self.data = self.backend.map_fn_over_arrays(
            self.data,
            lambda x: self.backend.roll(x, shift=-1, axis=0),
        )
        self.data = sbu.set_at(
            self.stacked_space,
            self.data,
            -1,
            data
        )
    
    def get_output_data(self) -> DataT:
        if self.batch_size is None:
            return self.data
        else:
            return sbu.swap_batch_dims_in_data(
                self.backend,
                self.data,
                0, 1
            ) # (L, B, ...) -> (B, L, ...)

class FrameStackWrapper(
    ContextObservationWrapper[
        ContextType, Union[DictT[str, Any], Any],
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self, 
        env: Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        obs_stack_size: int = 0,
        action_stack_size: int = 0,
        action_default_value: Optional[ActType] = None,
    ):
        assert obs_stack_size >= 0, "Observation stack size must be greater than 0"
        assert action_stack_size >= 0, "Action stack size must be greater than 0"
        assert action_stack_size == 0 or action_default_value is not None, "Action default value must be provided if action stack size is greater than 0"
        assert obs_stack_size > 0 or action_stack_size > 0, "At least one of observation stack size or action stack size must be greater than 0"
        super().__init__(env)
        obs_is_dict = isinstance(env.observation_space, DictSpace)
        assert obs_is_dict or action_stack_size == 0, "Action stack size must be 0 if observation space is not a DictSpace"
        
        self.action_stack_size = action_stack_size
        self.obs_stack_size = obs_stack_size

        if action_stack_size > 0:
            self.action_deque = SpaceDataQueue(
                env.action_space,
                env.batch_size,
                action_stack_size,
            )
            self.action_default_value = action_default_value
        else:
            self.action_deque = None
        
        self.obs_deque = None
        if obs_stack_size > 0:
            self.obs_deque = SpaceDataQueue(
                env.observation_space,
                env.batch_size,
                obs_stack_size + 1
            )
            
            if action_stack_size > 0:
                new_obs_spaces = self.obs_deque.output_space.spaces.copy()
                new_obs_spaces['past_actions'] = self.action_deque.output_space
                self.observation_space = DictSpace(
                    env.backend,
                    new_obs_spaces,
                    device=env.observation_space.device
                )
            else:
                self.observation_space = self.obs_deque.output_space
        else:
            if action_stack_size > 0:
                spaces = env.observation_space.spaces.copy()
                spaces['past_actions'] = self.action_deque.output_space
                self.observation_space = DictSpace(
                    env.backend,
                    spaces,
                    device=env.observation_space.device
                )
            self.obs_deque = None

    def reverse_map_context(self, context: ContextType) -> ContextType:
        return context

    def map_observation(self, observation: ObsType) -> Union[DictT[str, Any], Any]:
        if self.obs_deque is not None:
            observation = self.obs_deque.get_output_data()
        
        if self.action_deque is not None:
            stacked_action = self.action_deque.get_output_data()
            observation['past_actions'] = stacked_action
        return observation
    
    def reverse_map_observation(self, observation: Union[DictT[str, Any], Any]) -> ObsType:
        if isinstance(observation, dict):
            stacked_obs = observation.copy()
            stacked_obs.pop('past_actions', None)
        else:
            stacked_obs = observation
        
        if self.obs_deque is not None:
            obs_last = sbu.get_at(
                self.obs_deque.output_space,
                stacked_obs,
                -1
            ) if self.env.batch_size is None else sbu.get_at(
                self.obs_deque.output_space,
                stacked_obs,
                (slice(None), -1)
            )
            return obs_last
        else:
            return stacked_obs

    def reset(
        self,
        *args,
        mask: Optional[BArrayType] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, Union[DictT[str, Any], Any], DictT[str, Any]]:
        # TODO: If a mask is provided, we should only reset the stack for the masked indices
        context, obs, info = self.env.reset(
            *args,
            mask=mask,
            seed=seed,
            **kwargs
        )

        if self.action_deque is not None:
            self.action_deque.reset(
                initial_data=sbu.get_at( # Add a singleton batch dimension to the action
                    self.env.action_space,
                    self.action_default_value,
                    None
                ) if self.env.batch_size is not None else self.action_default_value,
                mask=mask
            )
        if self.obs_deque is not None:
            self.obs_deque.reset(
                initial_data=obs,
                mask=mask
            )
        
        return context, self.map_observation(obs), info
    
    def step(
        self,
        action: ActType
    ) -> Tuple[
        Union[DictT[str, Any], Any],
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
        DictT[str, Any]
    ]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        if self.action_deque is not None:
            self.action_deque.append(action)
        if self.obs_deque is not None:
            self.obs_deque.append(obs)
        
        return self.map_observation(obs), rew, terminated, truncated, info