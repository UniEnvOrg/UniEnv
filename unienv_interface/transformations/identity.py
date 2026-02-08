from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import BoxSpace
from typing import Union, Any, Optional, Dict
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils.symbol_util import get_full_class_name


class IdentityTransformation(DataTransformation):
    has_inverse = True

    def get_target_space_from_source(self, source_space):
        return source_space

    def transform(self, source_space, data):
        return data

    def direction_inverse(self, source_space = None):
        return self

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": get_full_class_name(type(self)),
        }
    
    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        backend: Optional[ComputeBackend] = None,
        device: Optional[BDeviceType] = None,
    ) -> "IdentityTransformation":
        return cls()