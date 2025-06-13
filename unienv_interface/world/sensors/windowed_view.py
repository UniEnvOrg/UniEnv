from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type

import PIL.Image
from ..sensor import Sensor, SensorDataT, FuncSensor, SensorStateT
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, TupleSpace
from unienv_interface.env_base.funcenv import StateType
import PIL

class WindowedViewSensor(
    Sensor[None, BDeviceType, BDtypeType, BRNGType],
    Generic[BDeviceType, BDtypeType, BRNGType]
):
    observation_space : TupleSpace[BDeviceType, BDtypeType, BRNGType]

class FuncWindowedViewSensor(
    FuncSensor[StateType, SensorStateT, Tuple, BDeviceType, BDtypeType, BRNGType],
    Generic[StateType, SensorStateT, BDeviceType, BDtypeType, BRNGType]
):
    observation_space : TupleSpace[BDeviceType, BDtypeType, BRNGType]