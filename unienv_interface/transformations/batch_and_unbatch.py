from unienv_interface.space.space_utils import batch_utils as sbu
from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import Space
from typing import Union, Any, Optional
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

class BatchifyTransformation(DataTransformation):
    has_inverse = True
    
    def get_target_space_from_source(self, source_space):
        return sbu.batch_space(source_space, 1)

    def transform(self, source_space, data):
        return sbu.concatenate(
            source_space,
            [data]
        )
    
    def direction_inverse(self, source_space = None):
        return UnBatchifyTransformation()

class UnBatchifyTransformation(DataTransformation):
    has_inverse = True
    
    def get_target_space_from_source(self, source_space):
        return next(iter(sbu.unbatch_spaces(source_space)))

    def transform(self, source_space, data):
        return sbu.get_at(
            source_space,
            data,
            0
        )
    
    def direction_inverse(self, source_space = None):
        return BatchifyTransformation()