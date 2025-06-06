from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import DictSpace
from typing import Union, Any, Optional, Dict, Set, Iterable
from xbarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

class FilterDictTransformation(DataTransformation[
    Dict[str, Any], BArrayType, BDeviceType, BDtypeType, BRNGType, 
    Dict[str, Any], BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    has_inverse = False
    def __init__(
        self,
        source_space : DictSpace[BDeviceType, BDtypeType, BRNGType],
        enabled_keys : Iterable[str],
    ):
        self.source_space = source_space
        enabled_keys = set(enabled_keys)
        
        assert isinstance(source_space, DictSpace), "Source space must be a DictSpace"
        assert all(key in source_space.spaces for key in enabled_keys), "All enabled keys must be in the source space"
        
        self.target_space = DictSpace(
            source_space.backend,
            {
                key: source_space.spaces[key] for key in enabled_keys
            },
            device=source_space.device
        )
        self.enabled_keys = enabled_keys

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: data[key] for key in self.enabled_keys
        }

    def transform_batched(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.transform(data)