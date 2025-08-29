from typing import Any, Optional, Tuple, Dict, Union, SupportsFloat, Sequence, Callable, Mapping
import jax
import jax.numpy as jnp
import numpy as np

from unienv_interface.env_base import FuncEnv
from unienv_interface.space import Space, BoxSpace, DictSpace
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend, JaxArrayType, JaxDeviceType, JaxDtypeType, JaxRNGType
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.wrapper import backend_compat

import mujoco.mjx as mjx
from mujoco_playground import MjxEnv, State as MjxState
from mujoco_playground._src.wrapper import Wrapper, MadronaWrapper, BraxDomainRandomizationVmapWrapper
from brax.envs.wrappers import training as brax_training
from flax import struct

AxisMapSingleT = Union[Mapping[str, "AxisMapSingleT"], int, None]
AxisMapT = Union[AxisMapSingleT, Tuple[AxisMapSingleT, ...]]
JaxTreeOrArrayT = Union[JaxArrayType, Dict[str, Any]]
RandomizationFnT = Callable[[mjx.Model], Tuple[mjx.Model, AxisMapSingleT]]

@struct.dataclass
class MJXPlaygroundState:
    state : MjxState
    rng : JaxRNGType

def is_mjx_env_vision(
    env: MjxEnv
) -> bool:
    try:
        from madrona_mjx.renderer import BatchRenderer  # pytype: disable=import-error
    except ImportError:
        return False
    if hasattr(env, "renderer") and isinstance(env.renderer, BatchRenderer):
        return True
    return False

def wrap_mjx_env(
    env: MjxEnv,
    batch_size: int,
    is_vision : bool = False,
    randomization_fn : Optional[
        RandomizationFnT
    ] = None,
) -> Wrapper:
    if is_vision:
        env = MadronaWrapper(env, batch_size, randomization_fn)
    elif randomization_fn is not None:
        env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    else:
        env = brax_training.VmapWrapper(env)
    return env

def space_from_size(
    size : AxisMapSingleT,
    device : Optional[JaxDeviceType] = None,
) -> Union[
    BoxSpace[JaxArrayType, JaxDeviceType, JaxDtypeType, JaxRNGType],
    DictSpace[JaxDeviceType, JaxDtypeType, JaxRNGType]
]:
    if isinstance(size, Mapping):
        return DictSpace(
            JaxComputeBackend,
            {
                key: space_from_size(size_inner, device=device)
                for key, size_inner in size.items()
            },
            device=device
        )
    else:
        return BoxSpace(
            JaxComputeBackend,
            -jnp.inf,
            jnp.inf,
            shape=size if not isinstance(size, int) else (size,),
            dtype=jnp.float32,
            device=device
        )

class FromMJXPlaygroundEnv(
    FuncEnv[
        MjxState,
        None,
        JaxArrayType,
        None,
        JaxTreeOrArrayT,
        JaxTreeOrArrayT,
        None,
        JaxDeviceType,
        JaxDtypeType,
        JaxRNGType
    ]
):
    metadata = {
        "render_modes": ['rgb_array', 'human']
    }
    backend = JaxComputeBackend
    def __init__(
        self,
        single_env: MjxEnv,
        batch_size: int,
        randomization_fn : Optional[
            RandomizationFnT
        ] = None,
        device : Optional[JaxDeviceType] = None,
        jit : bool = True
    ) -> None:
        self.single_env = single_env
        self.env = wrap_mjx_env(
            single_env, 
            batch_size,
            is_vision=is_mjx_env_vision(single_env),
            randomization_fn=randomization_fn
        )
        if jit:
            self.vanilla_reset_fn = jax.jit(self.env.reset, device=self.device)
            self.vanilla_step_fn = jax.jit(self.env.step, device=self.device)
        else:
            self.vanilla_reset_fn = self.env.reset
            self.vanilla_step_fn = self.env.step
        
        self.batch_size = batch_size
        self.device = device

        self.action_space = sbu.batch_space(
            space_from_size(
                single_env.action_size,
                device=self.device
            ),
            batch_size
        )
        self.observation_space = sbu.batch_space(
            space_from_size(
                single_env.observation_size,
                device=self.device
            ),
            batch_size
        )
        
        self.context_space = None

    def initial(self, *, seed : Optional[int]) -> Tuple[
        MJXPlaygroundState,
        None,
        JaxTreeOrArrayT,
        Dict[str, Any]
    ]:
        rng = jax.random.PRNGKey(seed)
        rng, reset_rng = jax.random.split(rng)
        reset_rng = jax.random.split(reset_rng, self.batch_size)
        raw_state = self.vanilla_reset_fn(rng=reset_rng)
        state = MJXPlaygroundState(
            state=raw_state,
            rng=rng
        )
        return state, None, raw_state.obs, raw_state.info

    def reset(
        self,
        state : MJXPlaygroundState,
        *,
        seed : Optional[int] = None,
        mask : Optional[JaxArrayType] = None
    ) -> Tuple[
        MJXPlaygroundState,
        None,
        JaxTreeOrArrayT,
        Dict[str, Any]
    ]:
        if seed is None:
            rng = state.rng
        else:
            rng = jax.random.PRNGKey(seed)
        rng, reset_rng = jax.random.split(rng)
        reset_rng = jax.random.split(reset_rng, self.batch_size)
        reset_state = self.env.reset(
            rng=reset_rng
        )
        if mask is None:
            return MJXPlaygroundState(
                state=reset_state,
                rng=rng
            ), None, reset_state.obs, reset_state.info
        else:
            def where_reset(
                x,y
            ) -> bool:
                mask_casted = jnp.reshape(mask, [mask.shape[0]] + [1]*(len(x.shape)-1))
                return jnp.where(mask_casted, x, y)
            def pick_reset(x):
                return x[mask]

            new_state = jax.tree.map(
                where_reset, reset_state, state.state
            )
            reset_obs = jax.tree.map(
                pick_reset, reset_state.obs
            ) if isinstance(reset_state.obs, Mapping) else reset_state.obs[mask]
            reset_info = jax.tree.map(
                pick_reset, reset_state.info
            )
            return MJXPlaygroundState(
                state=new_state,
                rng=rng
            ), None, reset_obs, reset_info

    def step(
        self,
        state : MJXPlaygroundState,
        action : JaxTreeOrArrayT
    ) -> Tuple[
        MJXPlaygroundState,
        JaxTreeOrArrayT,
        JaxArrayType,
        JaxArrayType,
        JaxArrayType,
        Dict[str, Any]
    ]:
        step_state = self.vanilla_step_fn(
            state.state,
            action
        )
        return (
            MJXPlaygroundState(
                state=step_state,
                rng=state.rng
            ), 
            step_state.obs, 
            step_state.reward, 
            step_state.done, 
            jnp.zeros_like(step_state.done, dtype=jnp.bool, device=self.device), 
            step_state.info
        )
        
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