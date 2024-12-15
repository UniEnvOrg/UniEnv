from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import Box
from typing import Union, Any, Optional
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

class RescaleTransformation(DataTransformation[
    BArrayType, BArrayType, BDeviceType, BDtypeType, BRNGType, 
    BArrayType, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    has_inverse = True
    def __init__(
        self,
        source_space : Box[BArrayType, BDeviceType, BDtypeType, BRNGType],
        new_low : Union[BArrayType,float] = -1.0,
        new_high : Union[BArrayType,float] = 1.0,
    ):
        assert isinstance(source_space, Box), "RescaleTransformation only supports Box action spaces"
        assert source_space.backend.dtype_is_real_floating(source_space.dtype), "RescaleTransformation only supports real-valued floating spaces"
        assert source_space.is_bounded('both'), "source_space only supports bounded spaces"
        
        self.source_space = source_space
        self.target_space = Box(
            backend=source_space.backend,
            low=new_low,
            high=new_high,
            dtype=source_space.dtype,
            device=source_space.device,
            shape=source_space.shape
        )
        self._source_space_span = self.source_space.high - self.source_space.low
        self._new_span = self.target_space.high - self.target_space.low

    def transform(self, data: BArrayType) -> BArrayType:
        normalized_data = (data - self.source_space.low) / self._source_space_span
        target_data = normalized_data * self._new_span + self.target_space.low
        return target_data
    
    def transform_batched(self, data: BArrayType) -> BArrayType:
        normalized_data = (data - self.source_space.low[None]) / self._source_space_span[None]
        target_data = normalized_data * self._new_span[None] + self.target_space.low[None]
        return target_data
    
    def inverse_transform(self, data: BArrayType) -> BArrayType:
        normalized_data = (data - self.target_space.low) / self._new_span
        source_data = normalized_data * self._source_space_span + self.source_space.low
        return source_data

    def inverse_transform_batched(self, data: BArrayType) -> BArrayType:
        normalized_data = (data - self.target_space.low[None]) / self._new_span[None]
        source_data = normalized_data * self._source_space_span[None] + self.source_space.low[None]
        return source_data