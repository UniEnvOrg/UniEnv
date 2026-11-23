from unienv_interface.space.space_utils import batch_utils as sbu
from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import Space, BoxSpace
from typing import Union, Any, Optional, Dict
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType


def _antialiased_bilinear_weights(in_size: int, out_size: int):
    """``(out_size, in_size)`` float64 interpolation matrix of antialiased bilinear resizing.

    Output pixel ``i`` is centered at input coordinate ``(i + 0.5) / scale - 0.5``
    and averages the input pixels with a triangle filter whose half-width is
    ``max(1 / scale, 1)`` input pixels; the weights of each row are normalized to
    sum to one (which handles the image borders). This matches PyTorch's
    ``interpolate(mode="bilinear", antialias=True)`` and JAX's
    ``image.resize(method="bilinear", antialias=True)``.
    """
    import numpy as np
    scale = out_size / in_size
    support = max(1.0 / scale, 1.0)
    centers = (np.arange(out_size, dtype=np.float64) + 0.5) / scale - 0.5
    positions = np.arange(in_size, dtype=np.float64)
    weights = np.maximum(0.0, 1.0 - np.abs(positions[None, :] - centers[:, None]) / support)
    return weights / weights.sum(axis=1, keepdims=True)


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
            shape=new_shape,
            # Still the same image (kind, format, layout, source), only smaller
            metadata=source_space.metadata,
        )

    def transform(self, source_space, data):
        """Resize ``data`` of shape ``(..., H, W, C)`` to ``(..., new_height, new_width, C)``.

        All backends compute the same result: bilinear interpolation with
        antialiasing (half-pixel centers, a triangle filter widened by the
        downscaling factor, weights renormalized at the borders), i.e. the
        semantics of ``torch.nn.functional.interpolate(mode="bilinear",
        antialias=True, align_corners=False)`` and ``jax.image.resize(
        method="bilinear", antialias=True)``. The output keeps the source dtype;
        integer images are interpolated in floating point, rounded and clipped to
        the dtype range.
        """
        source_space = self._validate_source_space(source_space)
        backend = source_space.backend
        dtype = data.dtype
        is_integer = backend.dtype_is_real_integer(dtype)
        target_shape = (
            *data.shape[:-3],
            self.new_height,
            self.new_width,
            data.shape[-1]
        )
        if backend.simplified_name == "jax":
            import jax.image
            import jax.numpy as jnp
            resized_data = jax.image.resize(
                data,
                shape=target_shape,
                method='bilinear',
                antialias=True
            )
            if is_integer:
                info = jnp.iinfo(dtype)
                resized_data = jnp.clip(jnp.round(resized_data), info.min, info.max)
            resized_data = resized_data.astype(dtype)
        elif backend.simplified_name == "pytorch":
            import torch
            import torch.nn.functional as F
            height, width, channels = data.shape[-3:]
            # F.interpolate expects (N, C, H, W): fold all leading dims into N.
            flat_data = data.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
            # Integer (and low-precision) images are interpolated in float32; CUDA has
            # no integer bilinear kernel.
            compute_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
            if flat_data.shape[0] == 0:
                resized_flat = flat_data.new_zeros((0, channels, self.new_height, self.new_width), dtype=compute_dtype)
            else:
                resized_flat = F.interpolate(
                    flat_data.to(compute_dtype),
                    size=(self.new_height, self.new_width),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )
            if is_integer:
                info = torch.iinfo(dtype)
                resized_flat = torch.clamp(torch.round(resized_flat), info.min, info.max)
            resized_data = resized_flat.to(dtype).permute(0, 2, 3, 1).reshape(target_shape)
        elif backend.simplified_name == "numpy":
            import numpy as np
            height_weights = _antialiased_bilinear_weights(data.shape[-3], self.new_height)
            width_weights = _antialiased_bilinear_weights(data.shape[-2], self.new_width)
            resized_data = np.einsum('oh,...hwc->...owc', height_weights, np.asarray(data, dtype=np.float64))
            resized_data = np.einsum('pw,...owc->...opc', width_weights, resized_data)
            if is_integer:
                info = np.iinfo(dtype)
                resized_data = np.clip(np.round(resized_data), info.min, info.max)
            resized_data = resized_data.astype(dtype)
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

    def serialize(
        self,
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> Dict[str, Any]:
        return {
            "new_height": self.new_height,
            "new_width": self.new_width,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> "ImageResizeTransformation":
        return cls(
            new_height=json_data["new_height"],
            new_width=json_data["new_width"],
        )
