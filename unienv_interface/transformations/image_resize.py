from unienv_interface.space.space_utils import batch_utils as sbu
from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import Space, BoxSpace
from typing import Union, Any, Optional, Dict
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils.symbol_util import get_full_class_name


class ImageResizeTransformation(DataTransformation):
    has_inverse = True

    def __init__(
        self,
        new_height: int,
        new_width: int
    ):
        self.new_height = new_height
        self.new_width = new_width

    def _validate_source_space(self, source_space: Space[Any, BDeviceType, BDtypeType, BRNGType]) -> BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        assert isinstance(source_space, BoxSpace), \
            f"ImageResizeTransformation only supports BoxSpace, got {type(source_space)}"
        assert len(source_space.shape) >= 3, \
            f"ImageResizeTransformation only supports spaces with at least 3 dimensions (H, W, C), got shape {source_space.shape}"
        assert source_space.shape[-3] > 0 and source_space.shape[-2] > 0, \
            f"ImageResizeTransformation requires positive height and width, got shape {source_space.shape}"
        return source_space

    def get_target_space_from_source(self, source_space):
        source_space = self._validate_source_space(source_space)

        backend = source_space.backend
        new_shape = (
            *source_space.shape[:-3],
            self.new_height,
            self.new_width,
            source_space.shape[-1]
        )
        new_low = backend.min(source_space.low, axis=(-3, -2), keepdims=True)
        new_high = backend.max(source_space.high, axis=(-3, -2), keepdims=True)

        return BoxSpace(
            source_space.backend,
            new_low,
            new_high,
            dtype=source_space.dtype,
            device=source_space.device,
            shape=new_shape
        )

    def transform(self, source_space, data):
        source_space = self._validate_source_space(source_space)
        backend = source_space.backend
        if backend.simplified_name == "jax":
            target_shape = (
                *data.shape[:-3],
                self.new_height,
                self.new_width,
                source_space.shape[-1]
            )
            import jax.image
            resized_data = jax.image.resize(
                data,
                shape=target_shape,
                method='bilinear',
                antialias=True
            )
        elif backend.simplified_name == "pytorch":
            import torch.nn.functional as F
            # PyTorch expects (B, C, H, W)
            data_permuted = backend.permute_dims(data, (*range(len(data.shape[:-3])), -1, -3, -2))
            resized_data_permuted = F.interpolate(
                data_permuted,
                size=(self.new_height, self.new_width),
                mode='bilinear',
                align_corners=False,
                antialias=True
            )
            # Permute back to original shape
            resized_data = backend.permute_dims(resized_data_permuted, (*range(len(resized_data_permuted.shape[:-3])), -2, -1, -3))
        elif backend.simplified_name == "numpy":
            import cv2
            flat_data = backend.reshape(data, (-1, *source_space.shape[-3:]))
            resized_flat_data = []
            for i in range(flat_data.shape[0]):
                img = flat_data[i]
                resized_img = cv2.resize(
                    img,
                    (self.new_width, self.new_height),
                    interpolation=cv2.INTER_LINEAR
                )
                resized_flat_data.append(resized_img)
            resized_flat_data = backend.stack(resized_flat_data, axis=0)
            resized_data = backend.reshape(
                resized_flat_data,
                (*data.shape[:-3], self.new_height, self.new_width, source_space.shape[-1])
            )
        else:
            raise ValueError(f"Unsupported backend: {backend.simplified_name}")
        return resized_data

    def direction_inverse(self, source_space=None):
        assert source_space is not None, "Inverse transformation requires source_space"
        source_space = self._validate_source_space(source_space)
        return ImageResizeTransformation(
            new_height=source_space.shape[-3],
            new_width=source_space.shape[-2]
        )

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": get_full_class_name(type(self)),
            "new_height": self.new_height,
            "new_width": self.new_width,
        }

    @classmethod
    def deserialize_from(cls, json_data: Dict[str, Any]) -> "ImageResizeTransformation":
        return cls(
            new_height=json_data["new_height"],
            new_width=json_data["new_width"],
        )
