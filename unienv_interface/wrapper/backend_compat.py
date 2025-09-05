from typing import Dict, Any, Optional, Tuple, Union, Generic, SupportsFloat, Type, Sequence, Mapping, List, NamedTuple
from typing_extensions import deprecated
import numpy as np
import copy

from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils import seed_util
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.env_base.wrapper import Wrapper, WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT
from unienv_interface.space import Space

def data_to(
    data : Any,
    source_backend : Optional[ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]] = None,
    target_backend : Optional[ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]] = None,
    target_device : Optional[WrapperBDeviceT] = None,
):
    if source_backend.is_backendarray(data):
        if source_backend is not None and target_backend is not None and target_backend != source_backend:
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

class ToBackendOrDeviceWrapper(
    Wrapper[
        WrapperBArrayT, WrapperContextT, WrapperObsT, WrapperActT, WrapperRenderFrame, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        backend : Optional[ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT]] = None,
        device : Optional[WrapperBDeviceT] = None,
    ) -> None:
        super().__init__(env)
        self._backend = backend
        self._device = device

        # Set new rng compatible with the new backend and device
        env.rng, seed = seed_util.next_seed_rng(env.rng, env.backend)
        self._rng = (backend or env.backend).random.random_number_generator(
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
        return self._backend or self.env.backend
    
    @property
    def device(self) -> Optional[WrapperBDeviceT]:
        return self._device

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
        o_action = self.action_space.data_to(
            action, backend=self.env.action_space.backend, device=self.env.action_space.device
        )
        c_obs, c_reward, c_terminated, c_truncated, c_info = self.env.step(o_action)
        obs = self.env.observation_space.data_to(
            c_obs, backend=self.backend, device=self.observation_space.device
        )
        reward = data_to(
            c_reward, source_backend=self.env.backend, target_backend=self.backend, target_device=self.device
        )
        terminated = data_to(
            c_terminated, source_backend=self.env.backend, target_backend=self.backend, target_device=self.device
        )
        truncated = data_to(
            c_truncated, source_backend=self.env.backend, target_backend=self.backend, target_device=self.device
        )
        info = data_to(
            c_info, source_backend=self.env.backend, target_backend=self.backend, target_device=self.device
        )
        return obs, reward, terminated, truncated, info
    
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
                device=self.device
            )
        
        c_context, c_obs, c_info = self.env.reset(
            *args,
            mask=None if mask is None else data_to(
                mask, source_backend=self.backend, target_backend=self.env.backend, target_device=self.env.device
            ),
            seed=seed,
            **kwargs
        )

        context = None if self.context_space is None else self.env.context_space.data_to(
            c_context,
            backend=self.context_space.backend,
            device=self.context_space.device
        )
        obs = self.env.observation_space.data_to(
            c_obs,
            self.observation_space.backend,
            self.observation_space.device
        )
        info = data_to(
            c_info, source_backend=self.env.backend, target_backend=self.backend, target_device=self.device
        )
        return context, obs, info

    def render(self) -> WrapperRenderFrame | Sequence[WrapperRenderFrame] | None:
        frame = self.env.render()
        return data_to(
            frame,
            source_backend=self.env.backend,
            target_backend=self.backend,
            target_device=self.device
        ) if frame is not None else None

@deprecated("Use ToBackendOrDeviceWrapper instead")
def ToBackendWrapper(
    env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
    backend : ComputeBackend[WrapperBArrayT, WrapperBDeviceT, WrapperBDtypeT, WrapperBRngT],
    device : Optional[WrapperBDeviceT] = None,
):
    return ToBackendOrDeviceWrapper(
        env, 
        backend=backend, 
        device=device
    )

@deprecated("Use ToBackendOrDeviceWrapper instead")
def ToDeviceWrapper(
    env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
    device : BDeviceType,
    original_device : Optional[BDeviceType] = None
):
    return ToBackendOrDeviceWrapper(
        env,
        device=device
    )