from typing import Generic
from .world import MujocoFuncWorld, MujocoFuncWorldState

from xbarray import numpy as NumpyComputeBackend
from unienv_interface.world.sensor import FuncSensor, SensorDataT, SensorStateT, FuncSensorWrapper, WrapperStateT, WrapperDataT

class MujocoFuncSensor(
    FuncSensor[MujocoFuncWorldState, SensorStateT, SensorDataT, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE],
    Generic[SensorStateT, SensorDataT]
):
    pass

class MujocoFuncSensorWrapper(
    FuncSensorWrapper[
        MujocoFuncWorldState, 
        WrapperStateT, WrapperDataT, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE,
        SensorStateT, SensorDataT, NumpyComputeBackend.DEVICE_TYPE, NumpyComputeBackend.DTYPE_TYPE, NumpyComputeBackend.RNG_TYPE
    ],
    Generic[WrapperStateT, WrapperDataT, SensorStateT, SensorDataT]
):
    pass