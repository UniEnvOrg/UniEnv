from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import Space, BoxSpace
from typing import Union, Any, Optional, Dict
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType


class IdentityTransformation(DataTransformation):
    has_inverse = True

    def get_target_space_from_source(self, source_space):
        return source_space

    def transform(self, source_space, data):
        return data

    def direction_inverse(self, source_space = None):
        return self

    def serialize(
        self,
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> Dict[str, Any]:
        return {}
    
    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> "IdentityTransformation":
        return cls()
