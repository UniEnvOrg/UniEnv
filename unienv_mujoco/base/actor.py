from typing import Generic
from .world import MujocoFuncWorld, MujocoFuncWorldState

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world.actor import FuncActor, ActorStateT, ActorWrapperStateT, FuncActorWrapper

class MujocoFuncActor(
    FuncActor[MujocoFuncWorldState, ActorStateT, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE],
    Generic[ActorStateT]
):
    pass

class MujocoFuncActorWrapper(
    FuncActorWrapper[
        MujocoFuncWorldState, 
        ActorWrapperStateT, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE,
        ActorStateT, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE
    ],
    Generic[ActorWrapperStateT, ActorStateT]
):
    pass