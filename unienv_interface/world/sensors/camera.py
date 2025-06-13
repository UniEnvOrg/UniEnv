from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
import PIL.Image

from ..sensor import Sensor, SensorDataT, FuncSensor, SensorStateT
from unienv_interface.space import Space, BoxSpace
from unienv_interface.env_base.funcenv import StateType

class CameraSensor(
    Sensor[SensorDataT, BDeviceType, BDtypeType, BRNGType],
    Generic[SensorDataT, BDeviceType, BDtypeType, BRNGType]
):
    observation_space : BoxSpace[SensorDataT, BDeviceType, BDtypeType, BRNGType]
    
    """Mode of the camera (rgb_array, depth_array, segmentation_array) """
    camera_mode : str

    width : int
    
    height : int

    channels : int

    # Shape 4x4 (Homogeneous Transformation Matrix)
    extrinsic_matrix : Optional[BArrayType] = None

    # Shape 3x3
    intrinsic_matrix : Optional[BArrayType] = None

    def to_image(self, data : BArrayType) -> PIL.Image.Image:
        raise NotImplementedError

class FuncCameraSensor(
    FuncSensor[StateType, SensorStateT, SensorDataT, BDeviceType, BDtypeType, BRNGType],
    Generic[StateType, SensorStateT, SensorDataT, BDeviceType, BDtypeType, BRNGType]
):
    observation_space : BoxSpace[SensorDataT, BDeviceType, BDtypeType, BRNGType]
    
    """Mode of the camera (rgb_array, depth_array, segmentation_array) """
    camera_mode : str

    width : int
    
    height : int
    
    channels : int

    def to_image(self, data : BArrayType) -> PIL.Image:
        raise NotImplementedError
    
    def get_intrinsic_matrix(
        self,
        state: StateType,
        rng: BRNGType,
    ) -> BArrayType:
        """
        Shape 3x3
        """
        raise NotImplementedError
    
    def get_extrinsic_matrix(
        self,
        state: StateType,
        rng: BRNGType,
    ) -> BArrayType:
        """
        Shape 4x4 (Homogeneous Transformation Matrix)
        """
        raise NotImplementedError