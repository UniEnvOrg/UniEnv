from typing import Dict as DictT, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence, TypeVar
import numpy as np
import copy
from unienv_interface.utils import seed_util
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.wrapper import ContextObservationWrapper, ActionWrapper, WrapperContextT, WrapperObsT, WrapperActT
from unienv_interface.backends import ComputeBackend
from unienv_interface.space import Space, Dict, batch_utils as sbu, flatten_utils as sfu
from collections import deque

DataT = TypeVar('DataT')
class SpaceDataQueue(
    Generic[DataT, BArrayType, BDeviceType, BDtypeType, BRNGType]
):
    def __init__(
        self,
        space : Space[DataT, Any, BDeviceType, BDtypeType, BRNGType],
        batch_size : Optional[int],
        maxlen: int,
        default_value: Optional[DataT] = None
    ) -> None:
        self.space = space
        self.stacked_space = sbu.batch_space(space, maxlen)
        self.batch_size = batch_size
        self.default_value = default_value
        self.flat_default_value = None # Will be set later

        if batch_size is not None:
            self.count_valid : BArrayType = self.backend.array_api_namespace.zeros(
                batch_size,
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )
            self.output_space = sbu.swap_batch_dims(
                self.stacked_space,
                0,
                1
            ) # (T, B, ...) => (B, T, ...)
            self.flat_stacked_space = sfu.flatten_space(self.stacked_space, start_dim=2) # (T, B, ...) => (T, B, D)
            if default_value is not None:
                self.flat_default_value = sfu.flatten_data(self.space, default_value, start_dim=1) # (B, ...) => (B, D)
        else:
            self.count_valid = 0
            self.output_space = self.stacked_space # (T, ...)
            self.flat_stacked_space = sfu.flatten_space(self.stacked_space, start_dim=1) # (T, ...) => (T, D)
            if default_value is not None:
                self.flat_default_value = sfu.flatten_data(self.space, default_value) # (...) => (D)

        self.flat_data_queue = deque(maxlen=maxlen)

    @property
    def maxlen(self) -> int:
        return self.flat_data_queue.maxlen

    @property
    def backend(self) -> ComputeBackend:
        return self.space.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.space.device
    
    def reset(self, mask : Optional[BArrayType] = None) -> None:
        if self.batch_size is not None:
            if mask is not None and self.batch_size > 1:
                self.count_valid = self.backend.array_api_namespace.where(
                    mask,
                    self.backend.array_api_namespace.zeros_like(self.count_valid),
                    self.count_valid
                )
            else:
                self.count_valid = self.backend.array_api_namespace.zeros(
                    self.batch_size,
                    dtype=self.backend.default_integer_dtype,
                    device=self.device
                )
                self.flat_data_queue.clear()
        else:
            self.count_valid = 0
            self.flat_data_queue.clear()

    def append(self, data : DataT) -> None:
        self.count_valid += 1
        
        if self.batch_size is not None:
            flat_data = sfu.flatten_data(self.space, data, start_dim=1) # (B, ...) => (B, D)
        else:
            flat_data = sfu.flatten_data(self.space, data) # (...) => (D)

        self.flat_data_queue.append(flat_data)

    def get_stacked_flat_data(self) -> BArrayType:
        # Since the flat_data_queue can be shorter than maxlen, we need to pad the data
        queue_t = len(self.flat_data_queue)
        assert queue_t > 0 or self.flat_default_value is not None, "Data queue is empty and no default value is provided"
        if queue_t == 0:
            to_stack = [self.flat_default_value] * self.maxlen
        else:
            remaining_t = self.maxlen - queue_t
            to_stack = [self.flat_data_queue[0]] * remaining_t + list(self.flat_data_queue)
        
        stacked_data = sbu.concatenate(
            self.flat_stacked_space,
            to_stack
        ) # (T, B, ...) or (T, ...)

        if self.batch_size is None or queue_t == 0:
            return stacked_data
        elif self.batch_size == 1:
            stacked_data = sbu.swap_batch_dims_in_data(
                self.flat_stacked_space,
                stacked_data,
                0,
                1
            ) # (T, B, ...) => (B, T, ...)
            return stacked_data
        else:
            stacked_data = sbu.swap_batch_dims_in_data(
                self.flat_stacked_space,
                stacked_data,
                0,
                1
            ) # (T, B, ...) => (B, T, ...)
            # Now we need to replace the invalid data with the last valid data
            for batch_idx in range(self.batch_size):
                valid_t = self.count_valid[batch_idx]
                if valid_t < self.maxlen:
                    valid_mask = self.backend.array_api_namespace.arange(
                        self.maxlen,
                        device=self.device
                    ) >= (self.maxlen - valid_t)
                    stacked_data = self.backend.replace_inplace( # (B, T, D)
                        stacked_data,
                        batch_idx,
                        self.backend.replace_inplace( # (T, D)
                            stacked_data[batch_idx],
                            valid_mask,
                            stacked_data[batch_idx, self.maxlen - valid_t][None]
                        )
                    )
            return stacked_data
    
    def get_output_data(self) -> DataT:
        stacked_data = self.get_stacked_flat_data()
        if self.batch_size is None:
            return sfu.unflatten_data(
                self.output_space,
                stacked_data,
                start_dim=1
            )
        else:
            return sfu.unflatten_data(
                self.output_space,
                stacked_data,
                start_dim=2
            )

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
        action_default_value: Optional[ActType] = None
    ):
        assert obs_stack_size >= 0, "Observation stack size must be greater than 0"
        assert action_stack_size >= 0, "Action stack size must be greater than 0"
        assert action_stack_size == 0 or action_default_value is not None, "Action default value must be provided if action stack size is greater than 0"
        assert obs_stack_size > 0 or action_stack_size > 0, "At least one of observation stack size or action stack size must be greater than 0"
        super().__init__(env)
        obs_is_dict = isinstance(env.observation_space, Dict)
        assert obs_is_dict or action_stack_size == 0, "Action stack size must be 0 if observation space is not a Dict"
        
        self.action_stack_size = action_stack_size
        self.obs_stack_size = obs_stack_size

        if action_stack_size > 0:
            self.action_deque = SpaceDataQueue(
                env.action_space,
                env.batch_size,
                action_stack_size,
                default_value=action_default_value
            )
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
                self.observation_space = Dict(
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
                self.observation_space = Dict(
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
                mask=mask
            )
        if self.obs_deque is not None:
            self.obs_deque.reset(
                mask=mask
            )
            self.obs_deque.append(obs)
        
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