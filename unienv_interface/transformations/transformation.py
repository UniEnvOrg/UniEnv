import abc
from typing import Generic, TypeVar, Tuple, Dict, Any, Optional, SupportsFloat, Type, Sequence, Union
from unienv_interface.space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

SourceDataT = TypeVar("SourceDataT")
SourceBArrT = TypeVar("SourceBArrTypeT")
SourceBDeviceT = TypeVar("TargetBDeviceT")
SourceBDTypeT = TypeVar("TargetBDTypeT")
SourceBDRNGT = TypeVar("TargetBDRNGT")
TargetDataT = TypeVar("TargetDataT")
class DataTransformation(
    abc.ABC,
    Generic[
        TargetDataT, BArrayType, BDeviceType, BDtypeType, BRNGType,
        SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT
    ]
):
    target_space : Space[TargetDataT, Any, BDeviceType, BDtypeType, BRNGType]
    source_space : Space[SourceDataT, Any, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT]

    @property
    def backend(self) -> ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]:
        return self.target_space.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.target_space.device
    
    def transform(self, data : SourceDataT) -> TargetDataT:
        raise NotImplementedError

    def inverse_transform(self, data : TargetDataT) -> SourceDataT:
        raise NotImplementedError
    
    def direction_inverse(self) -> """DirectionInverseTransformation[
        SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT,
        TargetDataT, BArrayType, BDeviceType, BDtypeType, BRNGType,
    ]""":
        return DirectionInverseTransformation(self)

class DirectionInverseTransformation(
    DataTransformation[
        TargetDataT, BArrayType, BDeviceType, BDtypeType, BRNGType,
        SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT
    ]
):
    def __init__(
        self,
        transformation: DataTransformation[
            SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT,
            TargetDataT, BArrayType, BDeviceType, BDtypeType, BRNGType
        ]
    ):
        self.transformation = transformation
    
    @property
    def target_space(self) -> Space[TargetDataT, Any, BDeviceType, BDtypeType, BRNGType]:
        return self.transformation.source_space
    
    @property
    def source_space(self) -> Space[SourceDataT, Any, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT]:
        return self.transformation.target_space
    
    def transform(self, data : SourceDataT) -> TargetDataT:
        return self.transformation.inverse_transform(data)
    
    def inverse_transform(self, data : TargetDataT) -> SourceDataT:
        return self.transformation.transform(data)

    def direction_inverse(self) -> DataTransformation[
        SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT,
        TargetDataT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ]:
        return self.transformation