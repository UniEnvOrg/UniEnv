from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import BoxSpace
from typing import Union, Any, Optional
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

class IdentityTransformation(DataTransformation):
    has_inverse = True

    def get_target_space_from_source(self, source_space):
        return source_space

    def transform(self, source_space, data):
        return data
    
    def direction_inverse(self, source_space = None):
        return self