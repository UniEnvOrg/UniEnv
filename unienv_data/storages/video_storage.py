from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable, Type, Literal, cast
from fractions import Fraction
from unienv_interface.space import Space, BoxSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils.symbol_util import *

from unienv_data.base import SpaceStorage
from ._episode_storage import IndexableType, EpisodeStorageBase

import numpy as np
import os
import json
import shutil

import av
import imageio.v3 as iio
from imageio.plugins.pyav import PyAVPlugin
from av.codec.hwaccel import HWAccel, hwdevices_available
from av.codec import codecs_available

class VideoStorage(EpisodeStorageBase[
    BArrayType,
    BArrayType,
    BDeviceType,
    BDtypeType,
    BRNGType,
]):
    """
    A storage for RGB or depth video data using video files
    If encoding RGB video
    - Set `buffer_pixel_format` to `rgb24`
        - Set `file_pixel_format` to `None`
        - Set `file_ext` to anything you like (e.g., "mp4", "avi", "mkv", etc.)
    If encoding depth video
        - Set `buffer_pixel_format` to `gray16le` (You can use rescale transform inside a `TransformedStorage` to convert depth values to this format, where `dtype` should be `np.uint16`) - if in meters, set min to 0 and max to 65.535 as the multiplication factor is 1000 (i.e., depth in mm)
        - Set `file_pixel_format` to `gray16le`
        - Set `codec` to a lossless codec that supports gray16le, e.g., `ffv1`
        - Set `file_ext` to `mkv`
    """

    # ========== Class Attributes ==========
    @classmethod
    def create(
        cls,
        single_instance_space: BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType],
        *args,
        hardware_acceleration : Optional[Union[HWAccel, Literal['auto']]] = 'auto',
        codec : Union[str, Literal['auto']] = 'auto',
        file_ext : str = "mp4",
        file_pixel_format : Optional[str] = None,
        buffer_pixel_format : str = "rgb24",
        fps : int = 15,
        capacity : Optional[int] = None,
        cache_path : Optional[str] = None,
        multiprocessing : bool = False,
        **kwargs
    ) -> "VideoStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        assert not multiprocessing, "VideoStorage does not support multiprocessing mode when creating a new mutable storage"
        if cache_path is None:
            raise ValueError("cache_path must be provided for ImageStorage.create")
        assert not os.path.exists(cache_path), f"Cache path {cache_path} already exists"
        os.makedirs(cache_path, exist_ok=True)
        return VideoStorage(
            single_instance_space,
            cache_filename=cache_path,
            hardware_acceleration=hardware_acceleration,
            codec=codec,
            file_ext=file_ext,
            file_pixel_format=file_pixel_format,
            buffer_pixel_format=buffer_pixel_format,
            fps=fps,
            capacity=capacity,
        )

    @classmethod
    def load_from(
        cls,
        path : Union[str, os.PathLike],
        single_instance_space : BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType],
        *,
        hardware_acceleration : Optional[Union[HWAccel, Literal['auto']]] = 'auto',
        codec : Union[str, Literal['auto']] = 'auto',
        capacity : Optional[int] = None,
        read_only : bool = True,
        multiprocessing : bool = False,
        **kwargs
    ) -> "VideoStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        assert read_only or (not multiprocessing), "VideoStorage does not support multiprocessing mode when loading a mutable storage"
        metadata_path = os.path.join(path, "image_metadata.json")
        assert os.path.exists(metadata_path), f"Metadata file {metadata_path} does not exist"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        assert metadata["storage_type"] == cls.__name__, \
            f"Expected storage type {cls.__name__}, but found {metadata['storage_type']}"
        
        if codec == "auto":
            codec_base = metadata.get("codec_base", None)
            if hardware_acceleration is not None:
                codec = cls.get_auto_codec(base=codec_base)
            else:
                assert codec_base is not None, "Codec base must be specified in metadata to load without hardware acceleration"
                codec = codec_base
        file_ext = metadata["file_ext"]
        fps = int(metadata["fps"])
        file_pixel_format = metadata.get("file_pixel_format", None)
        buffer_pixel_format = metadata.get("buffer_pixel_format", "rgb24")

        if "capacity" in metadata:
            capacity = None if metadata['capacity'] is None else int(metadata["capacity"])
        length = None if capacity is None else metadata["length"]

        return VideoStorage(
            single_instance_space,
            cache_filename=path,
            hardware_acceleration=hardware_acceleration,
            codec=codec,
            file_ext=file_ext,
            file_pixel_format=file_pixel_format,
            buffer_pixel_format=buffer_pixel_format,
            fps=fps,
            mutable=not read_only,
            capacity=capacity,
            length=length,
        )

    # ========== Instance Implementations ==========
    single_file_ext = None

    @staticmethod
    def get_auto_hwaccel() -> Optional[HWAccel]:
        if hasattr(__class__, "_auto_hwaccel"):
            return __class__._auto_hwaccel
        available_hwdevices = hwdevices_available()
        target_hwaccel = None
        if "d3d11va" in available_hwdevices:
            target_hwaccel = HWAccel(device_type="d3d11va", allow_software_fallback=True)
        elif "cuda" in available_hwdevices:
            target_hwaccel = HWAccel(device_type="cuda", allow_software_fallback=True)
        elif "vaapi" in available_hwdevices:
            target_hwaccel = HWAccel(device_type="vaapi", allow_software_fallback=True)
        elif "videotoolbox" in available_hwdevices:
            target_hwaccel = HWAccel(device_type="videotoolbox", allow_software_fallback=True)
        __class__._auto_hwaccel = target_hwaccel
        return target_hwaccel

    @staticmethod
    def get_auto_codec(
        base : Optional[str] = None
    ) -> str:
        if hasattr(__class__, "_auto_codec"):
            return __class__._auto_codec
        preferred_codecs = ["av1", "hevc", "h264", "mpeg4", "vp9", "vp8"] if base is None else [base]
        preferred_suffixes = ["_nvenc", "_amf", "_qsv"]
        
        target_codec = None
        for codec in preferred_codecs:
            for suffix in preferred_suffixes:
                full_codec_name = codec + suffix
                if full_codec_name in codecs_available:
                    target_codec = full_codec_name
                    break

        if target_codec is None:
            for codec in preferred_codecs:
                if codec in codecs_available:
                    target_codec = codec
                    break
        
        if target_codec is None:
            raise RuntimeError("No suitable video codec found in available codecs.")

        __class__._auto_codec = target_codec
        return target_codec

    def __init__(
        self,
        single_instance_space: BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType],
        cache_filename : Union[str, os.PathLike],
        hardware_acceleration : Optional[Union[HWAccel, Literal['auto']]] = 'auto',
        codec : Union[str, Literal['auto']] = 'auto',
        file_ext : str = "mp4",
        file_pixel_format : Optional[str] = None,
        buffer_pixel_format : str = "rgb24",
        fps : int = 15,
        mutable : bool = True,
        capacity : Optional[int] = None,
        length : int = 0,
    ):
        super().__init__(
            single_instance_space,
            file_ext=file_ext,
            cache_filename=cache_filename,
            mutable=mutable,
            capacity=capacity,
            length=length,
        )
        self.hwaccel = None if hardware_acceleration is None else (
            self.get_auto_hwaccel() if hardware_acceleration == 'auto' else hardware_acceleration
        )
        self.codec = self.get_auto_codec() if codec == 'auto' else codec
        self.fps = fps
        self.file_pixel_format = file_pixel_format
        self.buffer_pixel_format = buffer_pixel_format

    def get_from_file(self, filename : str, index : Union[IndexableType, BArrayType], total_length : int) -> BArrayType:
        with iio.imopen(filename, 'r', plugin='pyav', hwaccel=self.hwaccel) as video:
            video = cast(PyAVPlugin, video)
            if isinstance(index, int):
                frame_np = video.read(index=index, format=self.buffer_pixel_format)
                frame = self.backend.from_numpy(frame_np)
                if self.device is not None:
                    frame = self.backend.to_device(frame, self.device)
                return frame
            else:
                if index is Ellipsis:
                    index = np.arange(total_length)
                elif isinstance(index, slice):
                    index = np.arange(*index.indices(total_length))
                elif self.backend.is_backendarray(index) and self.backend.dtype_is_boolean(index.dtype):
                    index = self.backend.nonzero(index)[0]
                if self.backend.is_backendarray(index):
                    index = self.backend.to_numpy(index)

                argsorted_indices = np.argsort(index)
                sorted_index = index[argsorted_indices]
                reserve_index = np.argsort(argsorted_indices)
                
                if len(index) < total_length // 2:
                    all_frames_np = []
                    for frame_i in sorted_index:
                        frame_np = video.read(index=frame_i, format=self.buffer_pixel_format)
                        all_frames_np.append(frame_np)
                    all_frames_np = np.stack(all_frames_np, axis=0)
                    # Reorder from sorted order back to original order
                    all_frames_np = all_frames_np[reserve_index]
                else:
                    # Create a set for O(1) lookup and a mapping from frame index to position in sorted_index
                    sorted_index_set = set(sorted_index)
                    frame_to_sorted_pos = {int(frame_idx): pos for pos, frame_idx in enumerate(sorted_index)}
                    
                    # Pre-allocate array to store frames in sorted order
                    all_frames_list = [None] * len(sorted_index)
                    past_frame_np = None
                    video_iter = video.iter(format=self.buffer_pixel_format)
                    for frame_i in range(total_length):
                        try:
                            frame_np = next(video_iter)
                        except StopIteration:
                            frame_np = past_frame_np
                        if frame_i in sorted_index_set:
                            # Store at the position corresponding to sorted_index order
                            all_frames_list[frame_to_sorted_pos[frame_i]] = frame_np
                        past_frame_np = frame_np
                    all_frames_np = np.stack(all_frames_list, axis=0)
                    # Reorder from sorted order back to original order
                    all_frames_np = all_frames_np[reserve_index]
                all_frames = self.backend.from_numpy(all_frames_np)
                if self.device is not None:
                    all_frames = self.backend.to_device(all_frames, self.device)
                return all_frames

    def set_to_file(self, filename : str, value : BArrayType):
        value_np = self.backend.to_numpy(value)
        with iio.imopen(
            filename,
            'w',
            plugin='pyav',
        ) as video:
            video = cast(PyAVPlugin, video)
            video.init_video_stream(self.codec, fps=self.fps, pixel_format=self.file_pixel_format)
            
            # Fix codec time base if not set:
            if video._video_stream.codec_context.time_base is None:
                video._video_stream.codec_context.time_base = Fraction(1 / self.fps).limit_denominator(int(2**16 - 1))

            for i, frame in enumerate(value_np):
                video.write_frame(frame, pixel_format=self.buffer_pixel_format)

    def dumps(self, path):
        assert os.path.samefile(path, self.cache_filename), \
            f"Dump path {path} does not match cache filename {self.cache_filename}"
        metadata = {
            "storage_type": __class__.__name__,
            "codec_base": self.codec if "_" not in self.codec else self.codec.split("_")[0],
            "file_ext": self.file_ext,
            "fps": self.fps,
            "file_pixel_format": self.file_pixel_format,
            "buffer_pixel_format": self.buffer_pixel_format,
            "capacity": self.capacity,
            "length": self.length,
        }
        with open(os.path.join(path, "image_metadata.json"), "w") as f:
            json.dump(metadata, f)

    def close(self):
        pass
