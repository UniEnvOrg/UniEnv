from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import Space, batch_utils as sbu
from typing import Union, Any, Optional
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

class BatchifyTransformation(DataTransformation[
    BArrayType, BArrayType, BDeviceType, BDtypeType, BRNGType, 
    BArrayType, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    has_inverse = True
    def __init__(
        self,
        source_space : Space[Any, Any, BDeviceType, BDtypeType, BRNGType],
    ):  
        self.source_space = source_space
        self.target_space = sbu.batch_space(source_space, 1)

    def transform(self, data: BArrayType) -> BArrayType:
        target_data = sbu.concatenate(self.target_space, [data])
        return target_data
    
    def transform_batched(self, data: BArrayType) -> BArrayType:
        return self.transform(data)
    
    def inverse_transform(self, data: BArrayType) -> BArrayType:
        return self.inverse_transform_batched(data)

    def inverse_transform_batched(self, data: BArrayType) -> BArrayType:
        source_data = next(sbu.iterate(self.target_space, data))
        return source_data

def UnBatchifyTransformation(
    source_space : Space[Any, Any, BDeviceType, BDtypeType, BRNGType],
) -> DataTransformation[
    BArrayType, BArrayType, BDeviceType, BDtypeType, BRNGType,
    BArrayType, BArrayType, BDeviceType, BDtypeType, BRNGType
]:
    return BatchifyTransformation(source_space).direction_inverse()