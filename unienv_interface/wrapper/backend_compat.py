from typing import Dict, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence, Mapping, List, NamedTuple
import numpy as np
import copy

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils import seed_util
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.env_base.wrapper import Wrapper, WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT
from unienv_interface.space import Space

def data_to(
    data : Any,
    source_backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    target_backend : Optional[ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]] = None,
    target_device : Optional[WrapperBDeviceT] = None,
):
    if source_backend.is_backendarray(data):
        if target_backend is not None:
            data = target_backend.from_other_backend(
                source_backend,
                data
            )
        if target_device is not None:
            data = (source_backend or target_backend).to_device(
                data,
                target_device
            )
    elif isinstance(data, Mapping):
        data = {
            key: data_to(value, source_backend, target_backend, target_device)
            for key, value in data.items()
        }
    elif isinstance(data, Sequence):
        data = [
            data_to(value, source_backend, target_backend, target_device)
            for value in data
        ]
        try:
            data = type(data)(data)  # Preserve the type of the original sequence
        except:
            pass
    return data

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

        env.rng, seed = seed_util.next_seed_rng(env.rng, env.backend)
        self._rng = backend.random.random_number_generator(
            seed=seed,
            device=device
        )
        self.action_space = env.action_space.to(
            backend,
            device
        )
        self.observation_space = env.observation_space.to(
            backend,
            device
        )
        self.context_space = None if env.context_space is None else env.context_space.to(
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
        c_action = self.action_space.data_to(
            action, backend=self.env.action_space.backend, device=self.env.action_space.device
        )
        obs, reward, terminated, truncated, info = self.env.step(c_action)
        c_obs = self.env.observation_space.data_to(
            obs, backend=self.observation_space.backend, device=self.observation_space.device
        )
        c_reward = float(reward) if not self.env.backend.is_backendarray(reward) else self.backend.from_other_backend(reward, self.env.backend)
        c_terminated = bool(terminated) if not self.env.backend.is_backendarray(terminated) else self.backend.from_other_backend(terminated, self.env.backend)
        c_truncated = bool(truncated) if not self.env.backend.is_backendarray(truncated) else self.backend.from_other_backend(truncated, self.env.backend)
        c_info = data_to(
            info,
            self.env.backend,
            self.backend,
            self.device,
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
            self._rng = self.backend.random.random_number_generator(
                seed=seed,
                device=self._device
            )
        
        context, obs, info = self.env.reset(
            *args,
            mask=None if mask is None else data_to(
                mask,
                self.backend,
                self.env.backend,
                self.env.device,
            ),
            seed=seed,
            **kwargs
        )

        c_context = None if self.context_space is None else self.env.context_space.data_to(
            context,
            backend=self.context_space.backend,
            device=self.context_space.device
        )
        c_obs = self.env.observation_space.data_to(
            obs,
            self.observation_space.backend,
            self.observation_space.device
        )
        c_info = data_to(
            info,
            self.env.backend,
            self.backend,
            self.device,
        )
        return c_context, c_obs, c_info
    
    def render(self) -> WrapperRenderFrame | Sequence[WrapperRenderFrame] | None:
        frame = self.env.render()
        return data_to(
            frame,
            self.env.backend,
            self.backend,
            self.device
        ) if frame is not None else None

class ToDeviceWrapper(
    Wrapper[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        device : BDeviceType,
        original_device : Optional[BDeviceType] = None
    ) -> None:
        assert device is not None
        super().__init__(env)
        self._action_space = env.action_space.to(
            device=device
        )
        self._observation_space = env.observation_space.to(
            device=device
        )
        self._context_space = None if env.context_space is None else env.context_space.to(
            device=device
        )
        self.original_device = original_device if original_device is not None else env.device

        env.rng, seed = seed_util.next_seed_rng(env.rng, env.backend)
        self._rng : BRNGType = env.backend.random.random_number_generator(
            seed=seed,
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
        c_action = self.action_space.data_to(
            action,
            device=self.env.action_space.device or self.original_device
        )
        obs, reward, terminated, truncated, info = self.env.step(c_action)
        c_obs = self.env.observation_space.data_to(
            obs,
            device=self.observation_space.device
        )
        c_reward = data_to(
            reward,
            self.backend,
            target_device=self.device
        )
        c_terminated = data_to(
            terminated,
            self.backend,
            target_device=self.device
        )
        c_truncated = data_to(
            truncated,
            self.backend,
            target_device=self.device
        )
        c_info = data_to(
            info,
            self.backend,
            target_device=self.device,
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
            self._rng = self.backend.random.random_number_generator(
                seed=seed,
                device=self._device
            )
        
        context, obs, info = self.env.reset(
            *args,
            mask=None if mask is None else data_to(
                mask,
                self.backend,
                target_device=self.env.device,
            ),
            seed=seed,
            **kwargs
        )

        c_context = None if self.context_space is None else self.env.context_space.data_to(
            context,
            self.backend,
            target_device=self.device
        )
        c_obs = self.env.observation_space.data_to(
            obs,
            self.observation_space.backend,
            device=self.observation_space.device
        )
        c_info = data_to(
            info,
            self.backend,
            target_device=self.device,
        )
        return c_context, c_obs, c_info

    def render(self) -> RenderFrame | Sequence[RenderFrame] | None:
        frame = self.env.render()
        frame = data_to(
            frame,
            self.backend,
            target_device=self.device
        )