from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable, Type, Literal, cast
from fractions import Fraction

from unienv_interface.space import BoxSpace
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.utils.symbol_util import *

from ._episode_storage import IndexableType, EpisodeStorageBase

import numpy as np
import os
import json
import logging
import importlib
import importlib.util

import av
from av.codec.hwaccel import HWAccel as PyAvHWAccel
from av.codec import codecs_available
import av.error
from av.video.reformatter import VideoReformatter

# TorchCodec backend for video encoding / decoding
# import av
# from av.codec import codecs_available
# from torchcodec.decoders import VideoDecoder, set_cuda_backend

try:
    import torch
except ImportError:
    torch = None

LOGGER = logging.getLogger(__name__)

class PyAvVideoReader:
    HWAccel = PyAvHWAccel

    @staticmethod
    def available_hwdevices() -> List[str]:
        from av.codec.hwaccel import hwdevices_available
        return hwdevices_available()
    
    @staticmethod
    def get_auto_hwaccel() -> Optional[HWAccel]:
        available_hwdevices = __class__.available_hwdevices()
        target_hwaccel = None
        if "d3d11va" in available_hwdevices:
            target_hwaccel = PyAvHWAccel(device_type="d3d11va", allow_software_fallback=True)
        elif "cuda" in available_hwdevices:
            target_hwaccel = PyAvHWAccel(device_type="cuda", allow_software_fallback=True)
        elif "vaapi" in available_hwdevices:
            target_hwaccel = PyAvHWAccel(device_type="vaapi", allow_software_fallback=True)
        elif "videotoolbox" in available_hwdevices:
            target_hwaccel = PyAvHWAccel(device_type="videotoolbox", allow_software_fallback=True)
        return target_hwaccel

    def __init__(
        self, 
        backend : ComputeBackend,
        filename: str, 
        buffer_pixel_format : Optional[str] = None,
        hwaccel: Optional[Union[HWAccel, Literal['auto']]] = None, 
        seek_mode: Literal['exact', 'approximate'] = 'exact',
        device : Optional[BDeviceType] = None,
    ):
        if seek_mode != 'exact':
            LOGGER.warning("PyAvVideoReader only supports 'exact' seek mode. Falling back to 'exact'.")
        
        if hwaccel == 'auto':
            hwaccel = __class__.get_auto_hwaccel()
        self.container = av.open(filename, mode='r', hwaccel=hwaccel)
        self.video_stream = self.container.streams.video[0]
        if buffer_pixel_format is not None:
            self.video_reformatter = VideoReformatter()
        else:
            self.video_reformatter = None
        self.buffer_pixel_format = buffer_pixel_format
        self.total_frames = self.video_stream.frames # Sometimes this reads 0 for some containers (as they don't explicitly store it)
        self.frame_iterator = self.container.decode(self.video_stream)
        self.backend = backend
        self.device = device
    
    def seek(self, frame_index: int):
        self.container.seek(frame_index, any_frame=True, backward=True, stream=self.video_stream)
        self.frame_iterator = self.container.decode(self.video_stream)
    
    def __next__(self):
        try:
            frame = next(self.frame_iterator)
            if self.video_reformatter is not None:
                frame = self.video_reformatter.reformat(
                    frame,
                    format=self.buffer_pixel_format,
                )
            frame = frame.to_ndarray(channel_last=True)
        except av.error.EOFError:
            raise StopIteration
        if np.prod(frame.shape) == 0:
            raise StopIteration
    
        return frame

    def __iter__(self):
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.container.close()
    
    def read(self, index : Union[IndexableType, BArrayType], total_length : int) -> BArrayType:
        if isinstance(index, int):
            self.seek(index)
            frame_np = next(self)
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
                past_frame_np = None
                for frame_i in sorted_index:
                    self.seek(int(frame_i))
                    
                    try:
                        frame_np = next(self)
                    except StopIteration:
                        frame_np = past_frame_np
                    past_frame_np = frame_np
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
                self.seek(0)
                for frame_i in range(total_length):
                    try:
                        frame_np = next(self)
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

class TorchCodecVideoReader:
    HWAccel = Literal['beta', 'ffmpeg']
    def __init__(
        self, 
        backend : ComputeBackend,
        filename: str, 
        buffer_pixel_format : Optional[str] = None,
        hwaccel: Optional[Union[HWAccel, Literal['auto']]] = None, 
        seek_mode: Literal['exact', 'approximate'] = 'exact',
        device : Optional[BDeviceType] = None,
    ):
        from torchcodec.decoders import VideoDecoder, set_cuda_backend
        assert torch is not None, "TorchCodecVideoReader requires PyTorch and TorchCodec to be installed."
        assert buffer_pixel_format == 'rgb24', "TorchCodecVideoReader currently only supports 'rgb24' buffer pixel format."
        if hwaccel == 'auto':
            hwaccel = __class__.get_auto_hwaccel()
        if hwaccel is not None:
            with set_cuda_backend(hwaccel):
                self.decoder = VideoDecoder(filename, device='cuda', seek_mode=seek_mode)
        else:
            self.decoder = VideoDecoder(filename, seek_mode=seek_mode)
        self.backend = backend
        self.device = device
        self.buffer_pixel_format = buffer_pixel_format

    @staticmethod
    def get_auto_hwaccel() -> Optional[HWAccel]:
        assert torch is not None, "TorchCodecVideoReader requires PyTorch to be installed."
        if torch.cuda.is_available():
            return 'beta'
        else:
            return None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        pass
    
    def read(self, index : Union[IndexableType, np.ndarray], total_length : int) -> np.ndarray:
        if isinstance(index, int):
            ret = self.decoder.get_frame_at(index).data.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
        elif index is Ellipsis:
            ret = self.decoder.get_frames_in_range(0, len(self.decoder)).data.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            ret = self.decoder.get_frames_in_range(start, stop, step=step).data.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        else:
            if self.backend.is_backendarray(index) and self.backend.dtype_is_boolean(index.dtype):
                index = self.backend.nonzero(index)[0]
            if self.backend.simplified_name != 'pytorch':
                index = PyTorchComputeBackend.from_other_backend(self.backend, index).to('cpu', torch.int64)
            ret = self.decoder.get_frames_at(index).data.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        
        if self.backend.simplified_name != 'pytorch':
            ret = self.backend.from_other_backend(PyTorchComputeBackend, ret)
        if self.device is not None:
            ret = self.backend.to_device(ret, self.device)
        return ret

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
        - Set `file_pixel_format` to `None` (especially when running with nvenc codec)
        - Set `file_ext` to anything you like (e.g., "mp4", "avi", "mkv", etc.)
    If encoding depth video
        - Set `buffer_pixel_format` to `gray16le` (You can use rescale transform inside a `TransformedStorage` to convert depth values to this format, where `dtype` should be `np.uint16`) - if in meters, set min to 0 and max to 65.535 as the multiplication factor is 1000 (i.e., depth in mm)
        - Set `file_pixel_format` to `gray16le`
        - Set `codec` to a lossless codec that supports gray16le, e.g., `ffv1`
        - Set `file_ext` to `mkv`
    """

    # ========== Class Attributes ==========
    PyAV_LOG_LEVEL = av.logging.WARNING

    @classmethod
    def create(
        cls,
        single_instance_space: BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType],
        *args,
        seek_mode : Literal['exact', 'approximate'] = 'exact',
        hardware_acceleration : Optional[Union[Any, Literal['auto']]] = 'auto',
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
            seek_mode=seek_mode,
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
        seek_mode : Literal['exact', 'approximate'] = 'exact',
        hardware_acceleration : Optional[Union[Any, Literal['auto']]] = 'auto',
        codec : Union[str, Literal['auto']] = 'auto',
        capacity : Optional[int] = None,
        read_only : bool = True,
        multiprocessing : bool = False,
        **kwargs
    ) -> "VideoStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        assert read_only or (not multiprocessing), "VideoStorage does not support multiprocessing mode when loading a mutable storage"
        metadata_path = os.path.join(path, "video_metadata.json")
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
            seek_mode=seek_mode,
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
    def get_auto_codec(
        base : Optional[str] = None
    ) -> str:
        if hasattr(__class__, "_auto_codec"):
            return __class__._auto_codec
        
        # --- PyAV Implementation ---
        preferred_codecs = ["av1", "hevc", "h264", "mpeg4", "vp9", "vp8"] if base is None else [base]
        preferred_suffixes = ["_nvenc", "_vaapi", "_amf", "_qsv", "_videotoolbox"]

        # --- TorchCodec Implementation ---
        # preferred_codecs = ["hevc", "av1", "h264", "mpeg4", "vp9", "vp8"] if base is None else [base]
        # preferred_suffixes = ["_nvenc"]
        
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
        seek_mode : Literal['exact', 'approximate'] = 'exact',
        hardware_acceleration : Optional[Union[Any, Literal['auto']]] = 'auto',
        codec : Union[str, Literal['auto']] = 'auto',
        decode_backend : Literal['torchcodec', 'pyav', 'auto'] = 'auto',
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
        self.seek_mode = seek_mode
        self.hwaccel = hardware_acceleration
        self.codec = self.get_auto_codec() if codec == 'auto' else codec
        
        if decode_backend == 'auto':
            if importlib.util.find_spec("torchcodec"):
                decode_backend = 'torchcodec'
            else:
                decode_backend = 'pyav'
        self.decode_backend = decode_backend

        self.fps = fps
        self.file_pixel_format = file_pixel_format
        self.buffer_pixel_format = buffer_pixel_format
        
    def get_from_file(self, filename : str, index : Union[IndexableType, BArrayType], total_length : int) -> BArrayType:
        if self.decode_backend == 'pyav':
            reader_cls = PyAvVideoReader
        elif self.decode_backend == 'torchcodec':
            reader_cls = TorchCodecVideoReader
        else:
            raise ValueError(f"Unknown decode_backend {self.decode_backend}")
        with reader_cls(
            backend=self.backend,
            filename=filename,
            buffer_pixel_format=self.buffer_pixel_format,
            hwaccel=self.hwaccel,
            seek_mode=self.seek_mode,
            device=self.device,
        ) as video_reader:
            return video_reader.read(index, total_length)

    # PyAV Implementation (Commented out due to bugs with hevc codec)
    def set_to_file(self, filename : str, value : BArrayType):
        # ImageIO Implementation
        # with iio.imopen(
        #     filename,
        #     'w',
        #     plugin='pyav',
        # ) as video:
        #     video = cast(PyAVPlugin, video)
        #     video.init_video_stream(self.codec, fps=self.fps, pixel_format=self.file_pixel_format)
            
        #     # Fix codec time base if not set:
        #     if video._video_stream.codec_context.time_base is None:
        #         video._video_stream.codec_context.time_base = Fraction(1 / self.fps).limit_denominator(int(2**16 - 1))

        #     for i, frame in enumerate(value_np):
        #         video.write_frame(frame, pixel_format=self.buffer_pixel_format)

        # PyAV Implementation
        logging.getLogger("libav").setLevel(self.PyAV_LOG_LEVEL)
        with av.open(filename, mode='w') as container:
            output_stream = container.add_stream(self.codec, rate=self.fps)
            if len(self.single_instance_space.shape) == 3: # (H, W, C)
                output_stream.width = self.single_instance_space.shape[1]
                output_stream.height = self.single_instance_space.shape[0]
            else: # (H, W)
                output_stream.width = self.single_instance_space.shape[1]
                output_stream.height = self.single_instance_space.shape[0]
            if self.file_pixel_format is not None:
                output_stream.pix_fmt = self.file_pixel_format
            # output_stream.time_base = Fraction(1, self.fps).limit_denominator(int(2**16 - 1))
            value_np = self.backend.to_numpy(value)
            for i, frame_np in enumerate(value_np):
                frame = av.VideoFrame.from_ndarray(frame_np, format=self.buffer_pixel_format, channel_last=True)
                packets = output_stream.encode(frame)
                if packets:
                    container.mux(packets)
            # Flush stream
            packets = output_stream.encode()
            if packets:
                container.mux(packets)
            
            # Close container
            if hasattr(output_stream, 'close'):
                output_stream.close()
            # container.close()

            # Restore logging level
            av.logging.restore_default_callback()

    # TorchCodec Implementation (extremely slow, and seems to have double-write issues)
    # def set_to_file(self, filename : str, value : BArrayType):
    #     if self.backend.simplified_name != 'pytorch':
    #         value_pt = PyTorchComputeBackend.from_other_backend(self.backend, value)
    #     else:
    #         value_pt = value
    #     if self.codec.endswith('_nvenc'):
    #         value_pt = value_pt.to("cuda")
    #     else:
    #         value_pt = value_pt.to("cpu")
        
    #     if len(self.single_instance_space.shape) < 3:
    #         # Add channel dimension for grayscale images
    #         value_pt = value_pt[..., None]  # (N, H, W) -> (N, H, W, 1)
        
    #     encoder = VideoEncoder(
    #         value_pt.permute(0, 3, 1, 2), # (N, H, W, C) -> (N, C, H, W)
    #         frame_rate=self.fps,
    #     )
    #     encoder.to_file(
    #         filename,
    #         codec=self.codec,
    #         pixel_format=self.file_pixel_format,
    #     )

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
        with open(os.path.join(path, "video_metadata.json"), "w") as f:
            json.dump(metadata, f)
        super().dumps(path)

    def close(self):
        pass
