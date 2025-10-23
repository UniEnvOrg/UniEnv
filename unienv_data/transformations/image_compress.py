from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.transformations import DataTransformation
from unienv_interface.space import Space, BoxSpace, TextSpace
from typing import Union, Any, Optional
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from PIL import Image
import numpy as np
import io

class ImageCompressTransformation(DataTransformation):
    has_inverse = True

    def __init__(
        self,
        init_quality : int = 75,
        max_size_bytes : int = 65536,
        mode : Optional[str] = None,
        format : str = "JPEG",
    ) -> None:
        """
        Initialize JPEG compression transformation.
        Args:
            init_quality: Initial JPEG quality setting (1-100).
            max_size_bytes: Maximum allowed size of compressed JPEG in bytes.
            mode: Optional mode for PIL Image (e.g., "RGB", "L"). If None, inferred from input.
            format: Image format to use for compression (default "JPEG"). See https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html for options.
        """

        self.init_quality = init_quality
        self.max_size_bytes = max_size_bytes
        self.mode = mode
        self.format = format

    @staticmethod
    def validate_source_space(source_space: Space[Any, BDeviceType, BDtypeType, BRNGType]) -> None:
        assert isinstance(source_space, BoxSpace), "JPEGCompressTransformation only supports BoxSpace source spaces."
        assert len(source_space.shape) >= 3 and (
            source_space.shape[-1] == 3 or
            source_space.shape[-1] == 1
        ), "JPEGCompressTransformation only supports BoxSpace source spaces with shape (..., H, W, 1 or 3)."

    @staticmethod
    def get_uint8_dtype(
        backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    ) -> BDtypeType:
        return backend.__array_namespace_info__().dtypes()['uint8']

    def get_target_space_from_source(self, source_space):
        self.validate_source_space(source_space)
        new_shape = source_space.shape[:-3] + (self.max_size_bytes,)
        
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

        quality = 95
        while quality >= min_quality:
            buf = io.BytesIO()
            img.save(buf, format=self.format, quality=quality)
            image_bytes = buf.getvalue()
            if len(image_bytes) <= max_bytes:
                return image_bytes, quality
            quality -= 5

        img.close()
        # Return lowest quality attempt if still too large
        return image_bytes, quality
    
    def transform(self, source_space, data):
        self.validate_source_space(source_space)
        data_numpy = source_space.backend.to_numpy(data)
        flat_data_numpy = data_numpy.reshape(-1, *data_numpy.shape[-3:])
        flat_compressed_data = np.zeros((flat_data_numpy.shape[0], self.max_size_bytes), dtype=np.uint8)
        for i in range(flat_data_numpy.shape[0]):
            img_array = flat_data_numpy[i]
            image_bytes, _ = self.encode_to_size(
                img_array,
                self.max_size_bytes,
                mode=self.mode
            )
            byte_array = np.frombuffer(image_bytes, dtype=np.uint8)
            flat_compressed_data[i, :len(byte_array)] = byte_array
        compressed_data = flat_compressed_data.reshape(data_numpy.shape[:-3] + (self.max_size_bytes, ))
        compressed_data_backend = source_space.backend.from_numpy(compressed_data, dtype=self.get_uint8_dtype(source_space.backend), device=source_space.device)
        return compressed_data_backend
    
    def direction_inverse(self, source_space = None):
        assert source_space is not None, "Source space must be provided to get inverse transformation."
        self.validate_source_space(source_space)
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

class ImageDecompressTransformation(DataTransformation):
    has_inverse = True
    
    def __init__(
        self,
        target_height : int,
        target_width : int,
        target_channels : int = 3,
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
        decompressed_image = flat_decompressed_image.reshape(data_numpy.shape[:-1] + (self.target_height, self.target_width, self.target_channels))
        decompressed_image_backend = source_space.backend.from_numpy(decompressed_image, dtype=self.get_uint8_dtype(source_space.backend), device=source_space.device)
        return decompressed_image_backend
    
    def direction_inverse(self, source_space = None):
        assert source_space is not None, "Source space must be provided to get inverse transformation."
        self.validate_source_space(source_space)
        return ImageCompressTransformation(
            init_quality=75,
            max_size_bytes=source_space.shape[-1],
            mode=self.mode,
            format=self.format if self.format is not None else "JPEG",
        )