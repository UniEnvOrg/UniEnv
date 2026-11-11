from typing import Dict, Any, Optional, Sequence, Tuple, Union, SupportsFloat
import numpy as np
import copy

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.wrapper import ContextObservationWrapper, ActionWrapper, WrapperContextT, WrapperObsT, WrapperActT
from unienv_interface.space import Space, DictSpace
from unienv_interface.utils.framestack_queue import SpaceDataQueue

class FrameStackWrapper(
    ContextObservationWrapper[
        ContextType, Union[Dict[str, Any], Any],
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    """Stack the last ``obs_stack_size + 1`` observations (and optionally past actions) along a new time axis.

    ``latest_only_keys`` (dict observations): top-level keys kept at their latest frame only, with a time axis of
    length 1, so ``obs[key][..., -1]``-style indexing still works while large entries (e.g. camera images whose
    history the consumer never reads) are not copied ``obs_stack_size + 1`` times.
    """

    def __init__(
        self, 
        env: Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        obs_stack_size: int = 0,
        action_stack_size: int = 0,
        action_default_value: Optional[ActType] = None,
        latest_only_keys: Optional[Sequence[str]] = None,
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
        
        self.latest_only_keys = tuple(latest_only_keys or ())
        assert not self.latest_only_keys or obs_is_dict, "latest_only_keys needs a DictSpace observation space"
        missing = [key for key in self.latest_only_keys if key not in env.observation_space.spaces] if obs_is_dict else []
        assert not missing, f"latest_only_keys not in the observation space: {missing}"

        self.obs_deque = None
        self.latest_deque = None
        if obs_stack_size > 0:
            stacked_space = env.observation_space
            if self.latest_only_keys:
                stacked_space = DictSpace(
                    env.observation_space.backend,
                    {k: v for k, v in env.observation_space.spaces.items() if k not in self.latest_only_keys},
                    device=env.observation_space.device,
                )
                self.latest_deque = SpaceDataQueue(
                    DictSpace(
                        env.observation_space.backend,
                        {k: env.observation_space.spaces[k] for k in self.latest_only_keys},
                        device=env.observation_space.device,
                    ),
                    env.batch_size,
                    1,
                )
            self.obs_deque = SpaceDataQueue(
                stacked_space,
                env.batch_size,
                obs_stack_size + 1
            )
            if self.latest_deque is not None:
                self._stacked_output_space = DictSpace(
                    env.observation_space.backend,
                    {
                        k: (self.latest_deque.output_space.spaces[k] if k in self.latest_only_keys
                            else self.obs_deque.output_space.spaces[k])
                        for k in env.observation_space.spaces.keys()
                    },
                    device=env.observation_space.device,
                )
            else:
                self._stacked_output_space = self.obs_deque.output_space
            if action_stack_size > 0:
                new_obs_space = copy.copy(self._stacked_output_space)
                new_obs_space['past_actions'] = self.action_deque.output_space
                self.observation_space = new_obs_space
            else:
                self.observation_space = self._stacked_output_space
        else:
            if action_stack_size > 0:
                new_obs_space = copy.copy(env.observation_space)
                new_obs_space['past_actions'] = self.action_deque.output_space
                self.observation_space = new_obs_space
            else:
                raise ValueError("At least one of observation stack size or action stack size must be greater than 0")

    def reverse_map_context(self, context: ContextType) -> ContextType:
        return context

    def map_observation(self, observation: ObsType) -> Union[Dict[str, Any], Any]:
        return self._stacked_observation(observation)

    def _stacked_observation(
        self,
        observation: ObsType,
        mask: Optional[BArrayType] = None
    ) -> Union[Dict[str, Any], Any]:
        """Build the stacked observation; with ``mask`` only the selected rows are returned
        (``observation`` is then the masked observation returned by the inner reset)."""
        if self.obs_deque is not None:
            observation = self.obs_deque.get_output_data()
            if self.latest_deque is not None:
                observation = {**observation, **self.latest_deque.get_output_data()}
            if mask is not None:
                observation = sbu.get_at(self._stacked_output_space, observation, mask)
        
        if self.action_deque is not None:
            stacked_action = self.action_deque.get_output_data()
            if mask is not None:
                stacked_action = sbu.get_at(self.action_deque.output_space, stacked_action, mask)
            observation = copy.copy(observation)
            observation['past_actions'] = stacked_action
        return observation
    
    def reverse_map_observation(self, observation: Union[Dict[str, Any], Any]) -> ObsType:
        if isinstance(observation, dict):
            stacked_obs = observation.copy()
            stacked_obs.pop('past_actions', None)
        else:
            stacked_obs = observation
        
        if self.obs_deque is not None:
            obs_last = sbu.get_at(
                self._stacked_output_space,
                stacked_obs,
                -1
            ) if self.env.batch_size is None else sbu.get_at(
                self._stacked_output_space,
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
    ) -> Tuple[ContextType, Union[Dict[str, Any], Any], Dict[str, Any]]:
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
            stacked, latest = self._split(obs)
            self.obs_deque.reset(
                initial_data=stacked,
                mask=mask
            )
            if self.latest_deque is not None:
                self.latest_deque.reset(initial_data=latest, mask=mask)
        
        return context, self._stacked_observation(obs, mask), info
    
    def step(
        self,
        action: ActType
    ) -> Tuple[
        Union[Dict[str, Any], Any],
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
        Dict[str, Any]
    ]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        if self.action_deque is not None:
            self.action_deque.add(action)
        if self.obs_deque is not None:
            stacked, latest = self._split(obs)
            self.obs_deque.add(stacked)
            if self.latest_deque is not None:
                self.latest_deque.add(latest)
        return self.map_observation(obs), rew, terminated, truncated, info

    def _split(self, obs: ObsType) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """``(stacked part, latest-only part)`` of an observation."""
        if self.latest_deque is None:
            return obs, None
        stacked = {k: v for k, v in obs.items() if k not in self.latest_only_keys}
        latest = {k: obs[k] for k in self.latest_only_keys}
        return stacked, latest