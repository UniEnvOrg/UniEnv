from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import BoxSpace
from typing import Union, Any, Optional
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

def _get_broadcastable_value(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    value: Union[BArrayType, float],
    target_ndim: int
) -> Union[BArrayType, float]:
    if isinstance(value, (int, float)):
        return value
    else:
        assert target_ndim <= len(value), "Value must have at least as many dimensions as target space"
        target_shape = tuple([1] * (target_ndim - len(value)) + list(value.shape))
        return backend.reshape(value, target_shape)

class RescaleTransformation(DataTransformation):
    has_inverse = True
    def __init__(
        self,
        new_low : Union[BArrayType,float] = -1.0,
        new_high : Union[BArrayType,float] = 1.0,
        new_dtype : Optional[BDtypeType] = None,
    ):
        # assert isinstance(source_space, BoxSpace), "RescaleTransformation only supports Box action spaces"
        # assert source_space.backend.dtype_is_real_floating(source_space.dtype), "RescaleTransformation only supports real-valued floating spaces"
        # assert source_space.is_bounded('both'), "source_space only supports bounded spaces"
        
        self.new_low = new_low
        self.new_high = new_high
        self._new_span = new_high - new_low
        self.new_dtype = new_dtype

    def get_target_space_from_source(self, source_space):
        assert isinstance(source_space, BoxSpace), "RescaleTransformation only supports Box spaces"
        assert source_space.is_bounded('both'), "source_space only supports bounded spaces"
        target_ndim = len(source_space.shape)
        target_low = _get_broadcastable_value(
            source_space.backend,
            self.new_low,
            target_ndim
        )
        target_high = _get_broadcastable_value(
            source_space.backend,
            self.new_high,
            target_ndim
        )
        if self.new_dtype is not None:
            target_dtype = self.new_dtype
        else:
            target_dtype = source_space.backend.result_type(source_space.dtype, target_low, target_high)
        target_space = BoxSpace(
            source_space.backend,
            low=target_low,
            high=target_high,
            dtype=target_dtype,
            shape=source_space.shape,
            device=source_space.device
        )
        return target_space

    def transform(self, source_space, data):
        assert isinstance(source_space, BoxSpace), "RescaleTransformation only supports Box spaces"
        target_ndim = len(source_space.shape)
        target_low = source_space.backend.to_device(_get_broadcastable_value(
            source_space.backend,
            self.new_low,
            target_ndim
        ), source_space.backend.device(data))
        target_high = source_space.backend.to_device(_get_broadcastable_value(
            source_space.backend,
            self.new_high,
            target_ndim
        ), source_space.backend.device(data))
        scaling_factor = (target_high - target_low) / (source_space._high - source_space._low)
        target_data = (data - source_space._low) * scaling_factor + target_low
        return target_data
    
    def direction_inverse(self, source_space = None):
        assert source_space is not None, f"{__class__.__name__} requires a source space for inverse transformation"
        assert isinstance(source_space, BoxSpace), "RescaleTransformation only supports Box spaces"
        assert source_space.is_bounded('both'), "source_space only supports bounded spaces"
        new_low = source_space._low
        new_high = source_space._high
        target_shape = source_space.backend.broadcast_shapes(new_low.shape, new_high.shape)
        while len(target_shape) >= 1:
            if target_shape[0] == 1:
                target_shape = target_shape[1:]
            else:
                break
        new_low = source_space.backend.reshape(new_low, target_shape)
        new_high = source_space.backend.reshape(new_high, target_shape)
        return RescaleTransformation(
            new_low=new_low,
            new_high=new_high,
            new_dtype=source_space.dtype
        )