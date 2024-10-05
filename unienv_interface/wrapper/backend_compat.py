from typing import Dict, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence
import gymnasium as gym
import numpy as np
import copy
from ..env_base.env import Env, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT
from ..env_base.wrapper import Wrapper, WrapperObsType, WrapperActType, WrapperRewardType, WrapperTerminationType, WrapperRenderFrame, WrapperBDeviceT, WrapperBRngT
from ..backends import ComputeBackend
from ..space import Space
import array_api_compat

def backend_dict_transform(
    target_backend : Type[ComputeBackend[Any, BDeviceT, Any, BRngT]],
    source_backend : Type[ComputeBackend[Any, WrapperBDeviceT, Any, WrapperBRngT]],
    data : Dict[str, Any]
) -> Dict[str, Any]:
    new_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            new_data[key] = backend_dict_transform(
                target_backend,
                source_backend,
                value
            )
        elif source_backend.is_backendarray(value):
            new_data[key] = target_backend.from_dlpack(
                value
            )
        else:
            new_data[key] = value
    return new_data

def backend_array_transform(
    target_backend : Type[ComputeBackend[Any, BDeviceT, Any, BRngT]],
    source_backend : Type[ComputeBackend[Any, WrapperBDeviceT, Any, WrapperBRngT]],
    data : Any
) -> Any:
    if source_backend.is_backendarray(data):
        return target_backend.from_dlpack(
            data
        )
    else:
        return data

class ToBackendWrapper(
    Wrapper[
        WrapperObsType, WrapperActType, WrapperRewardType, WrapperTerminationType, WrapperRenderFrame, WrapperBDeviceT, WrapperBRngT,
        ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT
    ]
):
    def __init__(
        self,
        env : Env[ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceT, BRngT],
        backend : Type[ComputeBackend[Any, BDeviceT, Any, BRngT]],
        device : Optional[WrapperBDeviceT] = None,
    ) -> None:
        super().__init__(env)
        self._rng : WrapperBRngT = backend.random_number_generator(
            seed=env.np_rng.integers(0),
            device=device
        )
        self._action_space = env.action_space.to_backend(
            backend,
            device
        )
        self._observation_space = env.observation_space.to_backend(
            backend,
            device
        )

        self._backend = backend
        self._device = device

    @property
    def backend(self) -> Type[ComputeBackend[Any, WrapperBDeviceT, Any, WrapperBRngT]]:
        return self._backend
    
    @property
    def device(self) -> Optional[WrapperBDeviceT]:
        return self._device

    @property
    def rng(self) -> WrapperBRngT:
        return self._rng

    def step(self, action: WrapperActType) -> Tuple[
        WrapperObsType, 
        WrapperRewardType, 
        WrapperTerminationType,
        WrapperTerminationType, 
        Dict[str, Any]
    ]:
        c_action = self.env.action_space.from_other_backend(
            action
        )
        obs, reward, terminated, truncated, info = self.env.step(c_action)
        c_obs = self.observation_space.from_other_backend(
            obs
        )
        c_reward = float(reward) if not self.env.backend.is_backendarray(reward) else self.backend.from_dlpack(reward)
        c_terminated = bool(terminated) if not self.env.backend.is_backendarray(terminated) else self.backend.from_dlpack(terminated)
        c_truncated = bool(truncated) if not self.env.backend.is_backendarray(truncated) else self.backend.from_dlpack(truncated)
        c_info = backend_dict_transform(
            self.backend,
            self.env.backend,
            info
        )
        return c_obs, c_reward, c_terminated, c_truncated, c_info
    
    def reset(
        self,
        *args,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        if seed is not None:
            self._rng = self.backend.random_number_generator(
                seed=seed,
                device=self._device
            )
        obs, info = self.env.reset(
            *args,
            seed=seed,
            **kwargs
        )

        c_obs = self.observation_space.from_other_backend(
            obs
        )
        c_info = backend_dict_transform(
            self.backend,
            self.env.backend,
            info
        )
        return c_obs, c_info
    
    def render(self) -> WrapperRenderFrame | Sequence[WrapperRenderFrame] | None:
        frame = self.env.render()
        if frame is None:
            return None
        elif isinstance(frame, Sequence):
            return list([
                backend_array_transform(
                    self.backend,
                    self.env.backend,
                    f
                )
                for f in frame
            ])
        else:
            return backend_array_transform(
                self.backend,
                self.env.backend,
                frame
            )

class ToDeviceWrapper(
    Wrapper[
        WrapperObsType, WrapperActType, WrapperRewardType, WrapperTerminationType, WrapperRenderFrame, WrapperBDeviceT, WrapperBRngT,
        WrapperObsType, WrapperActType, WrapperRewardType, WrapperTerminationType, WrapperRenderFrame, WrapperBDeviceT, WrapperBRngT
    ]
):
    def __init__(
        self,
        env : Env[WrapperObsType, WrapperActType, WrapperRewardType, WrapperTerminationType, WrapperRenderFrame, WrapperBDeviceT, WrapperBRngT],
        device : BDeviceT
    ) -> None:
        super().__init__(env)
        self._rng : BRngT = env.backend.random_number_generator(
            seed=env.np_rng.integers(0),
            device=device
        )
        self._action_space = env.action_space.to_device(
            device
        )
        self._observation_space = env.observation_space.to_device(
            device
        )

        self._device = device
    
    @property
    def device(self) -> WrapperBDeviceT:
        return self._device

    @property
    def rng(self) -> BRngT:
        return self._rng

    def step(self, action: WrapperActType) -> Tuple[
        WrapperObsType, 
        WrapperRewardType, 
        WrapperTerminationType,
        WrapperTerminationType, 
        Dict[str, Any]
    ]:
        c_action = self.env.action_space.from_same_backend(
            action
        )
        obs, reward, terminated, truncated, info = self.env.step(c_action)
        c_obs = self.observation_space.from_same_backend(
            obs
        )
        c_reward = reward if not self.env.backend.is_backendarray(reward) else array_api_compat.to_device(reward, self._device)
        c_terminated = terminated if not self.env.backend.is_backendarray(terminated) else array_api_compat.to_device(terminated, self._device)
        c_truncated = truncated if not self.env.backend.is_backendarray(truncated) else array_api_compat.to_device(truncated, self._device)
        c_info = info
        return c_obs, c_reward, c_terminated, c_truncated, c_info

    def reset(
        self,
        *args,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        if seed is not None:
            self._rng = self.env.backend.random_number_generator(
                seed=seed,
                device=self._device
            )
        obs, info = self.env.reset(
            *args,
            seed=seed,
            **kwargs
        )

        c_obs = self.observation_space.from_same_backend(
            obs
        )
        return c_obs, info

    def render(self) -> WrapperRenderFrame | Sequence[WrapperRenderFrame] | None:
        frame = self.env.render()
        if frame is None:
            return None
        elif isinstance(frame, Sequence):
            return list([
                array_api_compat.to_device(
                    f,
                    self._device
                ) if self.env.backend.is_backendarray(f) else f
                for f in frame
            ])
        else:
            return array_api_compat.to_device(
                frame,
                self._device
            ) if self.env.backend.is_backendarray(frame) else frame
