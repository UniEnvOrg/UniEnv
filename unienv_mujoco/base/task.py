from typing import Generic, Any
from .world import MujocoFuncWorld, MujocoFuncWorldState

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world.actor import ActorStateT
from unienv_interface.world.task import FuncTask, TaskStateT, FuncTaskWrapper, TaskWrapperStateT

class MujocoFuncTask(
    FuncTask[
        MujocoFuncWorldState, Any, 
        TaskStateT, NumpyComputeBackend.ARRAY_TYPE, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE
    ],
    Generic[TaskStateT]
):
    pass

class MujocoFuncTaskWrapper(
    FuncTaskWrapper[
        MujocoFuncWorldState, Any,
        TaskWrapperStateT, NumpyComputeBackend.ARRAY_TYPE, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE,
        TaskStateT, NumpyComputeBackend.ARRAY_TYPE, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE
    ],
    Generic[TaskWrapperStateT, TaskStateT]
):
    pass