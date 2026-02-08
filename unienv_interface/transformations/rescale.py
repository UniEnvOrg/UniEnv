from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import BoxSpace
from typing import Union, Any, Optional, Dict
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType, get_backend_from_tensor
from unienv_interface.backends.serialization import serialize_backend, deserialize_backend, serialize_dtype, deserialize_dtype
from unienv_interface.backends import serialization as bsu
from unienv_interface.utils.symbol_util import get_full_class_name
import numpy as np

def _get_broadcastable_value(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    value: Union[BArrayType, float],
    target_ndim: int
) -> Union[BArrayType, float]:
    if isinstance(value, (int, float)):
        return value
    else:
        assert target_ndim >= len(value.shape), "Target space must have at least as many dimensions as the value"
        target_shape = tuple([1] * (target_ndim - len(value.shape)) + list(value.shape))
        return backend.reshape(value, target_shape)


class RescaleTransformation(DataTransformation):
    has_inverse = True
    def __init__(
        self,
        new_low : Union[BArrayType,float] = -1.0,
        new_high : Union[BArrayType,float] = 1.0,
        new_dtype : Optional[BDtypeType] = None,
        nan_to : Optional[Union[float, int, BArrayType]] = None,
    ):
        # assert isinstance(source_space, BoxSpace), "RescaleTransformation only supports Box action spaces"
        # assert source_space.backend.dtype_is_real_floating(source_space.dtype), "RescaleTransformation only supports real-valued floating spaces"
        # assert source_space.is_bounded('both'), "source_space only supports bounded spaces"
        
        self.new_low = new_low
        self.new_high = new_high
        self._new_span = new_high - new_low
        self.new_dtype = new_dtype
        self.nan_to = nan_to

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
        ), source_space.backend.device(data)) if source_space.backend.is_backendarray(self.new_low) else self.new_low
        target_high = source_space.backend.to_device(_get_broadcastable_value(
            source_space.backend,
            self.new_high,
            target_ndim
        ), source_space.backend.device(data)) if source_space.backend.is_backendarray(self.new_high) else self.new_high
        
        source_span = source_space._high - source_space._low
        source_span_zero_mask = source_span == 0
        scaling_factor = (target_high - target_low) / source_space.backend.where(source_span_zero_mask, 1, source_span)
        target_data = (data - source_space._low) * scaling_factor + target_low
        target_data = source_space.backend.where(source_span_zero_mask, target_low, target_data)
        
        if self.nan_to is not None:
            target_data = source_space.backend.where(
                source_space.backend.isnan(target_data),
                self.nan_to,
                target_data
            )

        if self.new_dtype is not None:
            if source_space.backend.dtype_is_real_integer(self.new_dtype) and source_space.backend.dtype_is_real_floating(target_data.dtype):
                target_data = source_space.backend.round(target_data)
            target_data = source_space.backend.astype(target_data, self.new_dtype)
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
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "nan_to"):
            self.nan_to = None

    def serialize(self) -> Dict[str, Any]:
        # Handle new_low and new_high - they can be scalars or arrays
        def serialize_value(value):
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return None, {"scalar": True, "value": value, "backend": None}
            else:
                # It's an array - convert to numpy and then to list
                backend = get_backend_from_tensor(value)
                return backend, {"scalar": False, "value": backend.to_numpy(value).tolist(), "backend": serialize_backend(backend)}

        new_low_backend, new_low_data = serialize_value(self.new_low)
        new_high_backend, new_high_data = serialize_value(self.new_high)
        nan_to_result = serialize_value(self.nan_to)
        if nan_to_result is None:
            nan_to_backend, nan_to_data = None, None
        else:
            nan_to_backend, nan_to_data = nan_to_result
        backend = new_low_backend or new_high_backend or nan_to_backend
        result = {
            "type": get_full_class_name(type(self)),
            "new_low": new_low_data,
            "new_high": new_high_data,
            "nan_to": nan_to_data,
        }
        # Store dtype as string if present (dtype is backend-agnostic string representation)
        if self.new_dtype is not None:
            result["new_dtype"] = str(self.new_dtype)
        return result

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        backend: Optional[ComputeBackend] = None,
        device: Optional[BDeviceType] = None,
    ) -> "RescaleTransformation":
        def deserialize_value(value_data, override_backend: Optional[ComputeBackend] = None, override_device: Optional[BDeviceType] = None):
            if value_data is None:
                return None, None
            if value_data.get("scalar", False):
                return None, value_data["value"]
            else:
                # Use provided backend if available, otherwise deserialize from saved backend
                if override_backend is not None:
                    value_backend = override_backend
                else:
                    value_backend = deserialize_backend(value_data["backend"])
                value = value_backend.from_numpy(np.array(value_data['value']))
                if override_device is not None:
                    value = value_backend.to_device(value, override_device)
                return value_backend, value

        # Deserialize dtype from string if present
        # The dtype will be converted to the appropriate backend-specific dtype 
        # when the transformation is actually applied
        
        new_low_backend, new_low = deserialize_value(json_data["new_low"], backend, device)
        new_high_backend, new_high = deserialize_value(json_data["new_high"], backend, device)
        nan_to_backend, nan_to = deserialize_value(json_data.get("nan_to"), backend, device)
        result_backend = backend or new_low_backend or new_high_backend or nan_to_backend

        new_dtype = None
        if json_data.get("new_dtype") is not None and result_backend is not None:
            # Store as numpy dtype for now - will be converted when applied
            new_dtype = deserialize_dtype(result_backend, json_data["new_dtype"])

        return cls(
            new_low=new_low,
            new_high=new_high,
            new_dtype=new_dtype,
            nan_to=nan_to,
        )