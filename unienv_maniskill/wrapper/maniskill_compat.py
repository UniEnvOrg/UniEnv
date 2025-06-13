from typing import Any, Optional, Tuple, Dict, Union, SupportsFloat, Sequence
from mani_skill.envs.sapien_env import BaseEnv as ManiSkillBaseEnv
import gymnasium as gym
import torch
import numpy as np

from unienv_interface.env_base.env import Env
from unienv_interface.space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space.space_utils import batch_utils as space_batch_utils, gym_utils as space_gym_utils
from unienv_interface.wrapper import backend_compat, gym_compat

MANISKILL_ENV_ARRAYTYPE = Union[PyTorchComputeBackend.ARRAY_TYPE, NumpyComputeBackend.ARRAY_TYPE]
MANISKILL_ENV_DEVICETYPE = Union[PyTorchComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DEVICE_TYPE]
MANISKILL_ENV_DTYPET = Union[PyTorchComputeBackend.DTYPE_TYPE, NumpyComputeBackend.DTYPE_TYPE]
MANISKILL_ENV_RNGTYPE = Union[PyTorchComputeBackend.RNG_TYPE, NumpyComputeBackend.RNG_TYPE]

def convert_maniskill_dict_to_backend(
    dict: Dict[str, MANISKILL_ENV_ARRAYTYPE],
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    device: Optional[BDeviceType] = None,
):
    return {
        key: convert_maniskill_array_to_backend(value, backend, device)
        for key, value in dict.items()
    }

def convert_maniskill_array_to_backend(
    array: MANISKILL_ENV_ARRAYTYPE,
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    device: Optional[BDeviceType] = None,
):
    if isinstance(array, torch.Tensor):
        source_backend = PyTorchComputeBackend
    elif isinstance(array, np.ndarray):
        source_backend = NumpyComputeBackend
    else:
        return array
    
    if backend is source_backend:
        if device is not None:
            return backend.to_device(array, device)
        else:
            return array
    else:
        return backend.from_other_backend(
            array,
            source_backend
        )

def convert_maniskill_to_backend(
    data: Any,
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    device: Optional[BDeviceType] = None,
):
    if isinstance(data, dict):
        return convert_maniskill_dict_to_backend(data, backend, device)
    else:
        return convert_maniskill_array_to_backend(data, backend, device)

class FromManiSkillEnv(
    Env[
        PyTorchComputeBackend.ARRAY_TYPE,
        None,
        Any,
        Any,
        PyTorchComputeBackend.ARRAY_TYPE,
        PyTorchComputeBackend.DEVICE_TYPE,
        PyTorchComputeBackend.DTYPE_TYPE,
        PyTorchComputeBackend.RNG_TYPE
    ]
):
    def __init__(
        self,
        env: ManiSkillBaseEnv,
    ) -> None:
        self.env = env
        self.backend = PyTorchComputeBackend
        self.device = env.get_wrapper_attr("device")
        self.batch_size = env.get_wrapper_attr("num_envs")

        self.action_space = space_gym_utils.from_gym_space(
            env.action_space,
            self.backend,
            device=self.device
        )
        if env.get_wrapper_attr("num_envs") <= 1:
            # Weirdly Maniskill doesn't batch the action space when num_envs is 1 but will batch the observation space
            self.action_space = space_batch_utils.batch_space(
                self.action_space,
                1
            )
        
        self.observation_space = space_gym_utils.from_gym_space(
            env.observation_space,
            self.backend,
            device=self.device
        )
        self.context_space = None
        self.rng = torch.Generator(device=self.device)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.env.metadata

    def get_render_camera_params(
        self
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        ret = {}
        for name, camera in getattr(self.env, "_human_render_cameras", {}).items():
            ret[name] = camera.get_params()
        return ret

    @property
    def render_mode(self) -> Optional[str]:
        return self.env.render_mode
    
    @property
    def render_fps(self) -> Optional[int]:
        return self.env.get_wrapper_attr("control_freq")

    def reset(
        self,
        *args,
        mask: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[
        None,
        Any,
        Dict[str, Any]
    ]:
        options = None if mask is None else {
            "env_idx": torch.nonzero(mask).flatten()
        }
        obs, info = self.env.reset(
            *args,
            seed=seed,
            options=options,
            **kwargs
        )
        # We don't convert the obs using from_gym_data here 
        # because the array may not be numpy array
        obs = convert_maniskill_to_backend(
            obs,
            self.backend,
            self.device
        )
        if mask is not None:
            obs = space_batch_utils.get_at(
                self.observation_space,
                obs,
                mask
            )

        return None, obs, info

    def step(
        self,
        action: Any
    ) -> Tuple[
        Any,
        SupportsFloat,
        bool,
        bool,
        Dict[str, Any]
    ]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        c_obs = convert_maniskill_to_backend(
            obs,
            self.backend,
            self.device
        )
        c_rew = convert_maniskill_array_to_backend(
            rew,
            self.backend,
            self.device
        )
        c_terminated = convert_maniskill_array_to_backend(
            terminated,
            self.backend,
            self.device
        )
        c_truncated = convert_maniskill_array_to_backend(
            truncated,
            self.backend,
            self.device
        )
        c_info = convert_maniskill_dict_to_backend(
            info,
            self.backend,
            self.device
        )
        return c_obs, c_rew, c_terminated, c_truncated, c_info

    def render(
        self
    ) -> Optional[
        PyTorchComputeBackend.ARRAY_TYPE
    ]:
        render_ret = self.env.render()
        if render_ret is None:
            return None
        else:
            dat = convert_maniskill_to_backend(
                render_ret,
                self.backend,
                self.device
            )
            if PyTorchComputeBackend.is_backendarray(dat) and dat.shape[0] == 1:
                return dat.squeeze(0)
            else:
                return None
        
    # =========== Wrapper methods ==========
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or hasattr(self.env, name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        elif hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError(f"Attribute {name} not found in the environment.")
        
    def set_wrapper_attr(self, name: str, value: Any):
        if hasattr(self, name):
            setattr(self, name, value)
        elif hasattr(self.env, name):
            setattr(self.env, name, value)
        else:
            raise AttributeError(f"Attribute {name} not found in the environment.")