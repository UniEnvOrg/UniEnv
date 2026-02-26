from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.transformations import DataTransformation
from unienv_interface.space import Space, BoxSpace, TextSpace
from typing import Union, Any, Optional, Dict
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from PIL import Image
import numpy as np
import io
import math

# https://stackoverflow.com/questions/3471663/jpeg-compression-ratio
JPEG_QUALITY_COMPRESSION_MAP = {
    "quality": np.array([55, 60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=int),
    "compression_ratio": np.array([43.27, 36.90, 34.24, 31.50, 26.00, 25.06, 19.08, 14.30, 9.88, 5.27], dtype=float),
    "conservative_ratio": 0.6,
}
def get_jpeg_compression_ratio(init_quality : int) -> int:
    if init_quality <= JPEG_QUALITY_COMPRESSION_MAP['quality'][0]:
        return math.floor(JPEG_QUALITY_COMPRESSION_MAP['compression_ratio'][0] * JPEG_QUALITY_COMPRESSION_MAP['conservative_ratio'])
    if init_quality >= JPEG_QUALITY_COMPRESSION_MAP['quality'][-1]:
        return math.floor(JPEG_QUALITY_COMPRESSION_MAP['compression_ratio'][-1] * JPEG_QUALITY_COMPRESSION_MAP['conservative_ratio'])

    for i in range(1, len(JPEG_QUALITY_COMPRESSION_MAP['quality'])):
        if init_quality <= JPEG_QUALITY_COMPRESSION_MAP['quality'][i]:
            q_low = JPEG_QUALITY_COMPRESSION_MAP['quality'][i - 1]
            q_high = JPEG_QUALITY_COMPRESSION_MAP['quality'][i]
            r_low = JPEG_QUALITY_COMPRESSION_MAP['compression_ratio'][i - 1]
            r_high = JPEG_QUALITY_COMPRESSION_MAP['compression_ratio'][i]
            ratio = r_low + (r_high - r_low) * (init_quality - q_low) / (q_high - q_low)
            return math.floor(ratio * JPEG_QUALITY_COMPRESSION_MAP['conservative_ratio'])

CONSERVATIVE_COMPRESSION_RATIOS = {
    "JPEG": get_jpeg_compression_ratio,
}

class ImageCompressTransformation(DataTransformation):
    has_inverse = True

    def __init__(
        self,
        init_quality : int = 70,
        max_size_bytes : Optional[int] = None,
        compression_ratio : Optional[float] = None,
        mode : Optional[str] = None,
        format : str = "JPEG",
        last_channel : bool = True,
    ) -> None:
        """
        Initialize JPEG compression transformation.
        Args:
            init_quality: Initial JPEG quality setting (1-100).
            max_size_bytes: Maximum allowed size of compressed JPEG in bytes.
            mode: Optional mode for PIL Image (e.g., "RGB", "L"). If None, inferred from input.
            format: Image format to use for compression (default "JPEG"). See https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html for options.
        """
        assert not (max_size_bytes is not None and compression_ratio is not None), "Specify either max_size_bytes or compression_ratio, not both."
        assert max_size_bytes is not None or compression_ratio is not None or format in CONSERVATIVE_COMPRESSION_RATIOS, "Either max_size_bytes must be specified or format must have a conservative compression ratio defined."

        self.init_quality = init_quality
        self.max_size_bytes = max_size_bytes
        self.compression_ratio = compression_ratio if compression_ratio is not None else (CONSERVATIVE_COMPRESSION_RATIOS.get(format, None)(init_quality) if max_size_bytes is None else None)
        self.mode = mode
        self.format = format
        self.last_channel = last_channel

    def validate_source_space(self, source_space: Space[Any, BDeviceType, BDtypeType, BRNGType]) -> None:
        assert isinstance(source_space, BoxSpace), "JPEGCompressTransformation only supports BoxSpace source spaces."
        if not self.last_channel:
            assert len(source_space.shape) >= 2
        else:
            assert len(source_space.shape) >= 3 and (
                source_space.shape[-1] == 3 or
                source_space.shape[-1] == 1
            ), "JPEGCompressTransformation only supports BoxSpace source spaces with shape (..., H, W, 1 or 3)."

    @staticmethod
    def get_uint8_dtype(
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    ) -> BDtypeType:
        return backend.__array_namespace_info__().dtypes()['uint8']

    def _get_max_compressed_size(self, source_space : BoxSpace):
        if self.last_channel:
            H, W, C = source_space.shape[-3], source_space.shape[-2], source_space.shape[-1]
        else:
            H, W = source_space.shape[-2], source_space.shape[-1]
            C = 1
        return self.max_size_bytes if self.max_size_bytes is not None else (H * W * C // self.compression_ratio) + 1

    def get_target_space_from_source(self, source_space):
        self.validate_source_space(source_space)
        
        max_compressed_size = self._get_max_compressed_size(source_space)

        if not self.last_channel:
            new_shape = source_space.shape[:-2] + (max_compressed_size,)
        else:
            new_shape = source_space.shape[:-3] + (max_compressed_size,)
        
        return BoxSpace(
            source_space.backend,
            shape=new_shape,
            low=-source_space.backend.inf,
            high=source_space.backend.inf,
            dtype=self.get_uint8_dtype(source_space.backend),
            device=source_space.device,
        )
    
    def encode_to_size(self, img_array, max_bytes, min_quality=20, mode=None):
        """
        Encode an image (H, W, 3) or (H, W, 1) as JPEG bytes,
        reducing quality until <= max_bytes.

        Args:
            img_array: np.ndarray, uint8, shape (H, W, 3) RGB or (H, W, 1) grayscale
            max_bytes: maximum allowed size of JPEG file
            min_quality: minimum JPEG quality before giving up

        Returns:
            jpeg_bytes (bytes), final_quality (int)
        """
        # Handle grayscale (H, W, 1) → (H, W)
        if img_array.ndim == 3 and img_array.shape[-1] == 1:
            img_array = np.squeeze(img_array, axis=-1)

        # Create PIL Image (mode inferred automatically)
        img = Image.fromarray(img_array, mode=mode)

        quality = self.init_quality
        while quality >= min_quality:
            buf = io.BytesIO()
            img.save(buf, format=self.format, quality=quality)
            image_bytes = buf.getvalue()
            if len(image_bytes) <= max_bytes:
                return image_bytes, quality
            quality -= 10

        img.close()
        # Return lowest quality attempt if still too large
        return image_bytes, quality
    
    def transform(self, source_space, data):
        self.validate_source_space(source_space)
        
        max_compressed_size = self._get_max_compressed_size(source_space)
        data_numpy = source_space.backend.to_numpy(data)
        if not self.last_channel:
            flat_data_numpy = data_numpy.reshape(-1, *data_numpy.shape[-2:])
        else:
            flat_data_numpy = data_numpy.reshape(-1, *data_numpy.shape[-3:])
        
        flat_compressed_data = np.zeros((flat_data_numpy.shape[0], max_compressed_size), dtype=np.uint8)
        for i in range(flat_data_numpy.shape[0]):
            img_array = flat_data_numpy[i]
            image_bytes, _ = self.encode_to_size(
                img_array,
                max_compressed_size,
                mode=self.mode
            )
            byte_array = np.frombuffer(image_bytes, dtype=np.uint8)
            flat_compressed_data[i, :len(byte_array)] = byte_array
        
        if not self.last_channel:
            compressed_data = flat_compressed_data.reshape(data_numpy.shape[:-2] + (max_compressed_size, ))
        else:
            compressed_data = flat_compressed_data.reshape(data_numpy.shape[:-3] + (max_compressed_size, ))
        compressed_data_backend = source_space.backend.from_numpy(compressed_data, dtype=self.get_uint8_dtype(source_space.backend), device=source_space.device)
        return compressed_data_backend
    
    def direction_inverse(self, source_space = None):
        assert source_space is not None, "Source space must be provided to get inverse transformation."
        self.validate_source_space(source_space)
        
        if not self.last_channel:
            height = source_space.shape[-2]
            width = source_space.shape[-1]
            channels = None
        else:
            height = source_space.shape[-3]
            width = source_space.shape[-2]
            channels = source_space.shape[-1]
        return ImageDecompressTransformation(
            target_height=height,
            target_width=width,
            target_channels=channels,
            mode=self.mode,
            format=self.format,
        )
    
    def __setstate__(self, state):
        # for backward compatibility
        self.__dict__.update(state)
        if 'last_channel' not in state:
            self.last_channel = True

    def serialize(
        self,
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> Dict[str, Any]:
        return {
            "init_quality": self.init_quality,
            "max_size_bytes": self.max_size_bytes,
            "compression_ratio": self.compression_ratio,
            "mode": self.mode,
            "format": self.format,
            "last_channel": self.last_channel,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ):
        return cls(
            init_quality=json_data.get("init_quality", 70),
            max_size_bytes=json_data.get("max_size_bytes", None),
            compression_ratio=json_data.get("compression_ratio", None),
            mode=json_data.get("mode", None),
            format=json_data.get("format", "JPEG"),
            last_channel=json_data.get("last_channel", True),
        )

class ImageDecompressTransformation(DataTransformation):
    has_inverse = True
    
    def __init__(
        self,
        target_height : int,
        target_width : int,
        target_channels : Optional[int] = 3,
        mode : Optional[str] = None,
        format : Optional[str] = None,
    ) -> None:
        """
        Initialize JPEG decompression transformation.
        Args:
            target_height: Height of the decompressed image.
            target_width: Width of the decompressed image.
            mode: Optional mode for PIL Image (e.g., "RGB", "L"). If None, inferred from input.
            format: Image format to use for decompression (default None, which will try everything). See https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html for options.
        """
        self.target_height = target_height
        self.target_width = target_width
        self.target_channels = target_channels
        self.mode = mode
        self.format = format

    @staticmethod
    def validate_source_space(source_space: Space[Any, BDeviceType, BDtypeType, BRNGType]) -> None:
        assert isinstance(source_space, BoxSpace), "JPEGDecompressTransformation only supports BoxSpace source spaces."
        assert len(source_space.shape) >= 1, "JPEGDecompressTransformation requires source space with at least 1 dimension."

    @staticmethod
    def get_uint8_dtype(backend):
        return ImageCompressTransformation.get_uint8_dtype(backend)

    def get_target_space_from_source(self, source_space):
        self.validate_source_space(source_space)
        new_shape = source_space.shape[:-1] + (self.target_height, self.target_width, self.target_channels)
        if self.target_channels is None:
            new_shape = new_shape[:-1]
        
        return BoxSpace(
            source_space.backend,
            shape=new_shape,
            low=0,
            high=255,
            dtype=self.get_uint8_dtype(source_space.backend),
            device=source_space.device,
        )
    
    def decode_bytes(self, jpeg_bytes : bytes, mode=None):
        """
        Decode JPEG bytes to an image array (H, W, 3).

        Args:
            jpeg_bytes: bytes of JPEG image

        Returns:
            img_array: np.ndarray, uint8, shape (H, W, 3)
        """
        buf = io.BytesIO(jpeg_bytes)
        img = Image.open(buf, formats=[self.format] if self.format is not None else None)
        if mode is not None:
            img = img.convert(mode)
        img_array = np.array(img)
        img.close()
        return img_array
    
    def transform(self, source_space, data):
        self.validate_source_space(source_space)
        data_numpy = source_space.backend.to_numpy(data)
        flat_data_numpy = data_numpy.reshape(-1, data_numpy.shape[-1])
        flat_decompressed_image = np.zeros((flat_data_numpy.shape[0], self.target_height, self.target_width, self.target_channels), dtype=np.uint8)
        for i in range(flat_data_numpy.shape[0]):
            byte_array : np.ndarray = flat_data_numpy[i]
            flat_decompressed_image[i] = self.decode_bytes(
                byte_array.tobytes(),
                mode=self.mode
            )
        if self.target_channels is None:
            decompressed_image = flat_decompressed_image.reshape(data_numpy.shape[:-1] + (self.target_height, self.target_width))
        else:
            decompressed_image = flat_decompressed_image.reshape(data_numpy.shape[:-1] + (self.target_height, self.target_width, self.target_channels))
        decompressed_image_backend = source_space.backend.from_numpy(decompressed_image, dtype=self.get_uint8_dtype(source_space.backend), device=source_space.device)
        return decompressed_image_backend
    
    def direction_inverse(self, source_space = None):
        assert source_space is not None, "Source space must be provided to get inverse transformation."
        self.validate_source_space(source_space)
        return ImageCompressTransformation(
            max_size_bytes=source_space.shape[-1],
            mode=self.mode,
            format=self.format if self.format is not None else "JPEG",
            last_channel=self.target_channels is not None,
        )
    
    def serialize(
        self,
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> Dict[str, Any]:
        return {
            "target_height": self.target_height,
            "target_width": self.target_width,
            "target_channels": self.target_channels,
            "mode": self.mode,
            "format": self.format,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ):
        return cls(
            target_height=json_data["target_height"],
            target_width=json_data["target_width"],
            target_channels=json_data.get("target_channels", 3),
            mode=json_data.get("mode", None),
            format=json_data.get("format", None),
        )
