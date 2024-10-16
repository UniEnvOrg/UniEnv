from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type

import PIL.Image
from ..sensor import Sensor, SensorDataT, FuncSensor, SensorStateT
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, Box
from unienv_interface.env_base.funcenv import FuncEnvCommonState, StateType
import PIL

class CameraSensor(
    Sensor[SensorDataT, BDeviceType, BDtypeType, BRNGType],
    Generic[SensorDataT, BDeviceType, BDtypeType, BRNGType]
):
    observation_space : Box[SensorDataT, BDeviceType, BDtypeType, BRNGType]
    
    """Mode of the camera (rgb_array, depth_array, segmentation_array) """
    camera_mode : str

    width : int
    
    height : int

    channels : int

    def to_image(self, data : BArrayType) -> PIL.Image:
        raise NotImplementedError

class FuncCameraSensor(
    FuncSensor[StateType, SensorStateT, SensorDataT, BDeviceType, BDtypeType, BRNGType],
    Generic[StateType, SensorStateT, SensorDataT, BDeviceType, BDtypeType, BRNGType]
):
    observation_space : Box[SensorDataT, BDeviceType, BDtypeType, BRNGType]
    
    """Mode of the camera (rgb_array, depth_array, segmentation_array) """
    camera_mode : str

    width : int
    
    height : int
    
    channels : int

    def to_image(self, data : BArrayType) -> PIL.Image:
        raise NotImplementedError