from typing import Dict, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence
import gymnasium as gym
import numpy as np
import copy
from unienv_interface.utils import seed_util
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.env_base.wrapper import Wrapper, WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space

def backend_array_transform(
    target_backend : ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT],
    target_device : Optional[WrapperBDeviceT],
    source_backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    data : Any
) -> Any:
    if source_backend.is_backendarray(data):
        ret = target_backend.from_other_backend(
            data,
            source_backend
        )
        if target_device is not None:
            ret = target_backend.to_device(
                ret,
                target_device
            )
    else:
        ret = data
    return ret

def backend_array_device_transform(
    target_backend : ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT],
    target_device : Optional[WrapperBDeviceT],
    data : Any
) -> Any:
    if target_device is not None:
        ret = target_backend.to_device(
            data,
            target_device
        )
    else:
        ret = data
    return ret

def backend_dict_transform(
    target_backend : ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT],
    target_device : Optional[WrapperBDeviceT],
    source_backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    data : Dict[str, Any]
) -> Dict[str, Any]:
    new_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            new_data[key] = backend_dict_transform(
                target_backend,
                target_device,
                source_backend,
                value
            )
        else:
            new_data[key] = backend_array_transform(
                target_backend,
                target_device,
                source_backend,
                value
            )
    return new_data

def backend_dict_device_transform(
    target_backend : ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT],
    target_device : Optional[WrapperBDeviceT],
    data : Dict[str, Any]
) -> Dict[str, Any]:
    new_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            new_data[key] = backend_dict_device_transform(
                target_backend,
                target_device,
                value
            )
        else:
            new_data[key] = backend_array_device_transform(
                target_backend,
                target_device,
                value
            )
    return new_data

class ToBackendWrapper(
    Wrapper[
        WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        backend : ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT],
        device : Optional[WrapperBDeviceT] = None,
    ) -> None:
        super().__init__(env)
        self._backend = backend
        self._device = device

        self._rng : WrapperBRngT = backend.random_number_generator(
            seed=seed_util.next_seed_rng(env.rng, env.backend),
            device=device
        )
        self.action_space = env.action_space.to_backend(
            backend,
            device
        )
        self.observation_space = env.observation_space.to_backend(
            backend,
            device
        )
        self.context_space = None if env.context_space is None else env.context_space.to_backend(
            backend,
            device
        )

    @property
    def backend(self) -> ComputeBackend[Any, WrapperBDeviceT, Any, WrapperBRngT]:
        return self._backend
    
    @property
    def device(self) -> Optional[WrapperBDeviceT]:
        return self._device

    @property
    def rng(self) -> WrapperBRngT:
        return self._rng

    def step(
        self, 
        action: WrapperActT
    ) -> Tuple[
        WrapperObsT, 
        Union[SupportsFloat, WrapperBArrayT], 
        Union[bool, WrapperBArrayT],
        Union[bool, WrapperBArrayT], 
        Dict[str, Any]
    ]:
        c_action = self.env.action_space.from_other_backend(
            action, self.backend
        )
        obs, reward, terminated, truncated, info = self.env.step(c_action)
        c_obs = self.observation_space.from_other_backend(
            obs, self.env.backend
        )
        c_reward = float(reward) if not self.env.backend.is_backendarray(reward) else self.backend.from_other_backend(reward, self.env.backend)
        c_terminated = bool(terminated) if not self.env.backend.is_backendarray(terminated) else self.backend.from_other_backend(terminated, self.env.backend)
        c_truncated = bool(truncated) if not self.env.backend.is_backendarray(truncated) else self.backend.from_other_backend(truncated, self.env.backend)
        c_info = backend_dict_transform(
            self.backend,
            self.device,
            self.env.backend,
            info
        )
        return c_obs, c_reward, c_terminated, c_truncated, c_info
    
    def reset(
        self,
        *args,
        mask : Optional[WrapperBArrayT] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[WrapperContextT, WrapperObsT, Dict[str, Any]]:
        if seed is not None:
            self._rng = self.backend.random_number_generator(
                seed=seed,
                device=self._device
            )
        
        context, obs, info = self.env.reset(
            *args,
            mask=None if mask is None else backend_array_transform(
                self.env.backend,
                self.env.device,
                self.backend,
                mask
            ),
            seed=seed,
            **kwargs
        )

        c_context = None if self.context_space is None else self.context_space.from_other_backend(
            context
        )
        c_obs = self.observation_space.from_other_backend(
            obs, self.env.backend
        )
        c_info = backend_dict_transform(
            self.backend,
            self.device,
            self.env.backend,
            info
        )
        return c_context, c_obs, c_info
    
    def render(self) -> WrapperRenderFrame | Sequence[WrapperRenderFrame] | None:
        frame = self.env.render()
        if frame is None or isinstance(frame, np.ndarray):
            return frame
        elif self.env.backend.is_backendarray(frame):
            return backend_array_transform(
                self.backend,
                self.device,
                self.env.backend,
                frame
            )
        elif isinstance(frame, Sequence):
            return list([
                backend_array_transform(
                    self.backend,
                    self.device,
                    self.env.backend,
                    f
                )
                for f in frame
            ])
        else:
            return frame

class ToDeviceWrapper(
    Wrapper[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        device : BDeviceType
    ) -> None:
        assert device is not None
        super().__init__(env)
        self._action_space = env.action_space.to_device(
            device
        )
        self._observation_space = env.observation_space.to_device(
            device
        )
        self._context_space = None if env.context_space is None else env.context_space.to_device(
            device
        )

        self._rng : BRNGType = env.backend.random_number_generator(
            seed=seed_util.next_seed_rng(env.rng, env.backend),
            device=device
        )

        self._device = device

    @property
    def device(self) -> BDeviceType:
        return self._device

    @property
    def rng(self) -> BRNGType:
        return self._rng

    def step(self, action: ActType) -> Tuple[
        ObsType, 
        Union[SupportsFloat, BArrayType], 
        Union[bool, BArrayType],
        Union[bool, BArrayType], 
        Dict[str, Any]
    ]:
        c_action = self.env.action_space.from_same_backend(
            action
        )
        obs, reward, terminated, truncated, info = self.env.step(c_action)
        c_obs = self.observation_space.from_same_backend(
            obs
        )
        c_reward = backend_array_device_transform(
            self.backend,
            self.device,
            reward
        )
        c_terminated = backend_array_device_transform(
            self.backend,
            self.device,
            terminated
        )
        c_truncated = backend_array_device_transform(
            self.backend,
            self.device,
            truncated
        )
        c_info = backend_dict_device_transform(
            self.backend,
            self.device,
            info
        )
        return c_obs, c_reward, c_terminated, c_truncated, c_info

    def reset(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[
        ContextType,
        ObsType,
        Dict[str, Any]
    ]:
        if seed is not None:
            self._rng = self.env.backend.random_number_generator(
                seed=seed,
                device=self._device
            )
        
        context, obs, info = self.env.reset(
            *args,
            mask=None if mask is None else backend_array_device_transform(
                self.env.backend,
                self.env.device,
                mask
            ),
            seed=seed,
            **kwargs
        )

        c_context = None if self.context_space is None else self.context_space.from_same_backend(
            context
        )
        c_obs = self.observation_space.from_same_backend(
            obs
        )
        c_info = backend_dict_device_transform(
            self.backend,
            self.device,
            info
        )
        return c_context, c_obs, c_info

    def render(self) -> RenderFrame | Sequence[RenderFrame] | None:
        frame = self.env.render()
        if frame is None or isinstance(frame, np.ndarray):
            return frame
        elif self.env.backend.is_backendarray(frame):
            return backend_array_device_transform(
                self.backend,
                self.device,
                frame
            )
        elif isinstance(frame, Sequence):
            return list([
                backend_array_device_transform(
                    self.backend,
                    self.device,
                    f
                )
                for f in frame
            ])
        else:
            return frame