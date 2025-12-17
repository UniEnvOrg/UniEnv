from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.transformations import DataTransformation
from unienv_interface.space import Space, BoxSpace, TextSpace
from typing import Union, Any, Optional, Tuple, List
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from .identity import IdentityTransformation

class CropTransformation(DataTransformation):
    has_inverse = True

    def __init__(
        self,
        crop_low : Union[int, float, BArrayType],
        crop_high : Union[int, float, BArrayType],
    ) -> None:
        """
        Initialize Crop Transformation.
        Args:
            crop_low: Lower bound for cropping the data.
            crop_high: Upper bound for cropping the data.
        """
        self.crop_low = crop_low
        self.crop_high = crop_high

    def validate_source_space(self, source_space : Space[Any, BDeviceType, BDtypeType, BRNGType]) -> None:
        assert isinstance(source_space, BoxSpace), "CropTransformation only supports Box spaces"

    def get_crop_range(self, source_space : BoxSpace[Any, BDeviceType, BDtypeType, BRNGType]) -> Tuple[BArrayType, BArrayType]:
        new_low = self.crop_low
        if source_space.backend.is_backendarray(new_low):
            if len(new_low.shape) < len(source_space.shape):
                new_low = source_space.backend.reshape(
                    new_low,
                    (-1,)*(len(source_space.shape) - len(new_low.shape)) + new_low.shape
                )
            new_low = source_space.backend.astype(new_low, source_space.dtype)
            if source_space.device is not None:
                new_low = source_space.backend.to_device(new_low, source_space.device)
        new_high = self.crop_high
        if source_space.backend.is_backendarray(new_high):
            if len(new_high.shape) < len(source_space.shape):
                new_high = source_space.backend.reshape(
                    new_high,
                    (-1,)*(len(source_space.shape) - len(new_high.shape)) + new_high.shape
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
    
    def direction_inverse(self, source_space = None):
        return IdentityTransformation()