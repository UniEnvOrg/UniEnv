from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.transformations import DataTransformation
from unienv_interface.space import Space, BoxSpace, TextSpace
from typing import Union, Any, Optional, Tuple, List, Dict
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType, get_backend_from_tensor
from unienv_interface.backends.serialization import serialize_backend, deserialize_backend
from unienv_interface.utils.symbol_util import get_full_class_name
from .identity import IdentityTransformation
import numpy as np


class CropTransformation(DataTransformation):
    has_inverse = True

    def __init__(
        self,
        crop_low: Union[int, float, BArrayType],
        crop_high: Union[int, float, BArrayType],
    ) -> None:
        """
        Initialize Crop Transformation.
        Args:
            crop_low: Lower bound for cropping the data.
            crop_high: Upper bound for cropping the data.
        """
        self.crop_low = crop_low
        self.crop_high = crop_high

    def validate_source_space(self, source_space: Space[Any, BDeviceType, BDtypeType, BRNGType]) -> None:
        assert isinstance(source_space, BoxSpace), "CropTransformation only supports Box spaces"

    def get_crop_range(self, source_space: BoxSpace[Any, BDeviceType, BDtypeType, BRNGType]) -> Tuple[BArrayType, BArrayType]:
        new_low = self.crop_low
        if source_space.backend.is_backendarray(new_low):
            if len(new_low.shape) < len(source_space.shape):
                new_low = source_space.backend.reshape(
                    new_low,
                    (-1,) * (len(source_space.shape) - len(new_low.shape)) + new_low.shape
                )
            new_low = source_space.backend.astype(new_low, source_space.dtype)
            if source_space.device is not None:
                new_low = source_space.backend.to_device(new_low, source_space.device)
        new_high = self.crop_high
        if source_space.backend.is_backendarray(new_high):
            if len(new_high.shape) < len(source_space.shape):
                new_high = source_space.backend.reshape(
                    new_high,
                    (-1,) * (len(source_space.shape) - len(new_high.shape)) + new_high.shape
                )
            new_high = source_space.backend.astype(new_high, source_space.dtype)
            if source_space.device is not None:
                new_high = source_space.backend.to_device(new_high, source_space.device)
        return new_low, new_high

    def get_target_space_from_source(self, source_space):
        self.validate_source_space(source_space)
        crop_low, crop_high = self.get_crop_range(source_space)
        return BoxSpace(
            backend=source_space.backend,
            low=crop_low,
            high=crop_high,
            shape=source_space.shape,
            dtype=source_space.dtype,
            device=source_space.device,
        )

    def transform(self, source_space, data):
        self.validate_source_space(source_space)
        crop_low, crop_high = self.get_crop_range(source_space)
        return source_space.backend.clip(data, crop_low, crop_high)

    def direction_inverse(self, source_space=None):
        return IdentityTransformation()

    def serialize(self) -> Dict[str, Any]:
        def serialize_value(value):
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return None, {"scalar": True, "value": value, "backend": None}
            else:
                # It's an array - detect backend and convert to numpy list
                backend = get_backend_from_tensor(value)
                return backend, {"scalar": False, "value": backend.to_numpy(value).tolist(), "backend": serialize_backend(backend)}

        crop_low_backend, crop_low_data = serialize_value(self.crop_low)
        crop_high_backend, crop_high_data = serialize_value(self.crop_high)

        return {
            "type": get_full_class_name(type(self)),
            "crop_low": crop_low_data,
            "crop_high": crop_high_data,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        backend: Optional[ComputeBackend] = None,
        device: Optional[BDeviceType] = None,
    ) -> "CropTransformation":
        def deserialize_value(value_data, override_backend: Optional[ComputeBackend] = None, override_device: Optional[BDeviceType] = None):
            if value_data is None:
                return None, None
            if value_data.get("scalar", False):
                return None, value_data["value"]
            else:
                # Use provided backend if available, otherwise deserialize from saved backend
                if override_backend is not None:
                     backend = override_backend
                else:
                    backend = deserialize_backend(value_data["backend"])
                value = backend.from_numpy(np.array(value_data['value']))
                if override_device is not None:
                    value = backend.to_device(value, override_device)
                return backend, value

        _, crop_low = deserialize_value(json_data["crop_low"], backend, device)
        _, crop_high = deserialize_value(json_data["crop_high"], backend, device)

        return cls(
            crop_low=crop_low,
            crop_high=crop_high,
        )