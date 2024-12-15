import abc
from typing import Generic, TypeVar, Tuple, Dict, Any, Optional, SupportsFloat, Type, Sequence, Union
from unienv_interface.space import Space, batch_utils as sbu, flatten_utils as sfu
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

    def get_source_space_batched(self, batch_size : int) -> Space[SourceDataT, Any, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT]:
        return sbu.batch_space(self.source_space, batch_size)
    
    def get_target_space_batched(self, batch_size : int) -> Space[TargetDataT, Any, BDeviceType, BDtypeType, BRNGType]:
        return sbu.batch_space(self.target_space, batch_size)

    has_inverse : bool = False

    @property
    def backend(self) -> ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]:
        return self.target_space.backend
    
    @property
    def device(self) -> Optional[BDeviceType]:
        return self.target_space.device
    
    @abc.abstractmethod
    def transform(self, data : SourceDataT) -> TargetDataT:
        raise NotImplementedError

    def transform_batched(self, data : SourceDataT) -> TargetDataT:
        batched_source_space = self.get_source_space_batched(1)
        data_iter = sbu.iterate(batched_source_space, data)
        
        transformed_data = []
        for dat in data_iter:
            transformed_data.append(self.transform(dat))
        
        batched_target_space = self.get_target_space_batched(1)
        return sbu.concatenate(batched_target_space, transformed_data)

    def inverse_transform(self, data : TargetDataT) -> SourceDataT:
        raise NotImplementedError
    
    def inverse_transform_batched(self, data : TargetDataT) -> SourceDataT:
        batched_target_space = self.get_target_space_batched(1)
        data_iter = sbu.iterate(batched_target_space, data)
        
        transformed_data = []
        for dat in data_iter:
            transformed_data.append(self.inverse_transform(dat))
        
        batched_source_space = self.get_source_space_batched(1)
        return sbu.concatenate(batched_source_space, transformed_data)
    
    def direction_inverse(self) -> """DirectionInverseTransformation[
        SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT,
        TargetDataT, BArrayType, BDeviceType, BDtypeType, BRNGType,
    ]""":
        assert self.has_inverse, "This transformation does not have an inverse"
        return DirectionInverseTransformation(self)
    
    def close(self):
        pass

    def __del__(self):
        self.close()

class DirectionInverseTransformation(
    DataTransformation[
        TargetDataT, BArrayType, BDeviceType, BDtypeType, BRNGType,
        SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT
    ]
):
    has_inverse = True

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
    
    def transform_batched(self, data : SourceDataT) -> TargetDataT:
        return self.transformation.inverse_transform_batched(data)

    def inverse_transform(self, data : TargetDataT) -> SourceDataT:
        return self.transformation.transform(data)
    
    def inverse_transform_batched(self, data : TargetDataT) -> SourceDataT:
        return self.transformation.transform_batched(data)

    def direction_inverse(self) -> DataTransformation[
        SourceDataT, SourceBArrT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT,
        TargetDataT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ]:
        return self.transformation
