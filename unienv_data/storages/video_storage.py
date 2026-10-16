from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable, Type, Literal, cast
from fractions import Fraction
from collections import deque
import uuid

from unienv_interface.space import BoxSpace
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.utils.symbol_util import *

from unienv_data.base import SegmentWriter
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
    def _to_fraction(value: Any) -> Optional[Fraction]:
        if value is None:
            return None
        if isinstance(value, Fraction):
            return value

        numerator = getattr(value, 'numerator', None)
        denominator = getattr(value, 'denominator', None)
        if numerator is not None and denominator not in (None, 0):
            return Fraction(int(numerator), int(denominator))

        try:
            return Fraction(value)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

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
        elif "vulkan" in available_hwdevices:
            target_hwaccel = PyAvHWAccel(device_type="vulkan", allow_software_fallback=True)
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
        self.container = None # so close()/__del__ work if opening fails
        if hwaccel == 'auto':
            hwaccel = __class__.get_auto_hwaccel()
        try:
            self.container = av.open(filename, mode='r', hwaccel=hwaccel)
        except av.error.FFmpegError:
            # A listed hardware device can still fail to initialise (e.g. CUDA hidden
            # via CUDA_VISIBLE_DEVICES or a driver mismatch); decode in software then
            if hwaccel is None or not getattr(hwaccel, 'allow_software_fallback', False):
                raise
            self.container = av.open(filename, mode='r')
        self.video_stream = self.container.streams.video[0]
        
        self.video_reformatter : Optional[VideoReformatter] = None
        self.buffer_pixel_format = buffer_pixel_format
        self.total_frames = self.video_stream.frames # Sometimes this reads 0 for some containers (as they don't explicitly store it)
        self._frame_iterator = self.container.decode(self.video_stream)
        self._pointer_position = 0
        self._frame_buffer = deque()
        self.seek_mode = seek_mode
        self.backend = backend
        self.device = device

    def _start_pts(self) -> int:
        start_time = getattr(self.video_stream, 'start_time', None)
        return 0 if start_time is None else int(start_time)

    def _time_base(self) -> Optional[Fraction]:
        time_base = self._to_fraction(getattr(self.video_stream, 'time_base', None))
        if time_base is None or time_base <= 0:
            return None
        return time_base

    def _frame_rate(self) -> Optional[Fraction]:
        for attr_name in ('average_rate', 'base_rate', 'guessed_rate'):
            frame_rate = self._to_fraction(getattr(self.video_stream, attr_name, None))
            if frame_rate is not None and frame_rate > 0:
                return frame_rate
        return None

    def frame_index_to_pts(self, frame_index: int) -> int:
        if frame_index < 0:
            raise IndexError(f"Frame index must be non-negative, got {frame_index}")

        time_base = self._time_base()
        frame_rate = self._frame_rate()
        if time_base is None or frame_rate is None:
            raise RuntimeError("Cannot map frame index to pts without video stream time base and frame rate")

        pts = Fraction(self._start_pts()) + Fraction(frame_index, 1) / (frame_rate * time_base)
        return int(round(pts))

    def pts_to_frame_index(self, pts: int) -> int:
        time_base = self._time_base()
        frame_rate = self._frame_rate()
        if time_base is None or frame_rate is None:
            raise RuntimeError("Cannot map pts to frame index without video stream time base and frame rate")

        frame_index = (Fraction(int(pts)) - Fraction(self._start_pts())) * time_base * frame_rate
        return int(round(frame_index))

    def _reset_decoder(self, seek_pts: Optional[int] = None):
        if seek_pts is None:
            seek_pts = self._start_pts()
        self._frame_buffer.clear()
        self.container.seek(seek_pts, backward=True, any_frame=False, stream=self.video_stream)
        self._frame_iterator = self.container.decode(self.video_stream)

    def seek(self, frame_index: int):
        if frame_index < 0:
            raise IndexError(f"Frame index must be non-negative, got {frame_index}")
        if self.tell() == frame_index:
            return

        target_pts = self.frame_index_to_pts(frame_index)
        self._reset_decoder(target_pts)

        if self.seek_mode == 'exact':
            past_frame = None
            past_frame_index = None
            while True:
                try:
                    frame = next(self._frame_iterator)
                except (av.error.EOFError, StopIteration):
                    if past_frame is None:
                        raise RuntimeError(f"Reached end of video while seeking to frame index {frame_index}")
                    else:
                        self._pointer_position = past_frame_index
                        self._frame_buffer.append(past_frame)
                        return

                if frame.pts is None:
                    continue

                landed_frame_index = self.pts_to_frame_index(frame.pts)
                if landed_frame_index < frame_index:
                    past_frame = frame
                    past_frame_index = landed_frame_index
                    continue

                if landed_frame_index == frame_index:
                    self._pointer_position = frame_index
                    self._frame_buffer.append(frame)
                    return
                
                # Overshot the target frame, need to decide whether to return this frame or the previous one (if exists)
                if past_frame is not None and past_frame_index is not None and (frame_index - past_frame_index) < (landed_frame_index - frame_index):
                    self._pointer_position = past_frame_index
                    self._frame_buffer.extend((past_frame, frame))
                else:
                    self._pointer_position = landed_frame_index
                    self._frame_buffer.append(frame)

                return
        else:
            # Approximate mode trusts the stream's pts / frame-rate metadata and does not
            # pick the nearest frame on overshoot, but it still has to decode forward from
            # the keyframe the demuxer landed on (which can be many frames before the target).
            last_frame = None
            last_frame_index = None
            while True:
                try:
                    next_frame = next(self._frame_iterator)
                except (av.error.EOFError, StopIteration):
                    if last_frame is None:
                        raise RuntimeError(f"Reached end of video while seeking to frame index {frame_index}")
                    next_frame, landed_frame_index = last_frame, last_frame_index
                    break
                landed_frame_index = self.pts_to_frame_index(next_frame.pts) if next_frame.pts is not None else frame_index
                if landed_frame_index >= frame_index:
                    break
                last_frame, last_frame_index = next_frame, landed_frame_index
            self._frame_buffer.append(next_frame)
            self._pointer_position = landed_frame_index

    def tell(self) -> int:
        return self._pointer_position

    def __next__(self):
        try:
            if self._frame_buffer:
                frame = self._frame_buffer.popleft()
            else:
                frame = next(self._frame_iterator)
            
            self._pointer_position += 1
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
        if self.video_reformatter is None and self.buffer_pixel_format is not None:
            self.video_reformatter = VideoReformatter()
        return self
    
    def __exit__(self, *args, **kwargs):
        self.video_reformatter = None
    
    def __del__(self):
        self.close()
    
    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None

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
                # Map each requested frame index to all of its positions in sorted_index
                # (an index may be requested more than once).
                frame_to_sorted_pos : Dict[int, List[int]] = {}
                for pos, frame_idx in enumerate(sorted_index):
                    frame_to_sorted_pos.setdefault(int(frame_idx), []).append(pos)
                
                # Pre-allocate array to store frames in sorted order
                all_frames_list = [None] * len(sorted_index)
                past_frame_np = None
                self.seek(0)
                for frame_i in range(total_length):
                    try:
                        frame_np = next(self)
                    except StopIteration:
                        frame_np = past_frame_np
                    # Store at every position corresponding to sorted_index order
                    for pos in frame_to_sorted_pos.get(frame_i, ()):
                        all_frames_list[pos] = frame_np
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
    
    def __del__(self):
        self.close()

    def close(self):
        self.decoder = None

    def read(self, index : Union[IndexableType, np.ndarray], total_length : int) -> np.ndarray:
        if isinstance(index, int):
            ret = self.decoder.get_frame_at(index).data.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
        elif index is Ellipsis:
            ret = self.decoder.get_frames_in_range(0, len(self.decoder)).data.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self.decoder))
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
    `encoder_options` are passed to the encoder (e.g. `{"crf": "28", "g": "15"}`) and persisted in the metadata,
    so a loaded storage keeps encoding new segments the same way.
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
        decode_backend : Literal['torchcodec', 'pyav', 'auto'] = 'auto',
        cache_decoders : bool = False,
        file_ext : str = "mp4",
        file_pixel_format : Optional[str] = None,
        buffer_pixel_format : str = "rgb24",
        fps : int = 15,
        encoder_options : Optional[Dict[str, str]] = None,
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
        return cls(
            single_instance_space,
            cache_filename=cache_path,
            seek_mode=seek_mode,
            hardware_acceleration=hardware_acceleration,
            codec=codec,
            decode_backend=decode_backend,
            cache_decoders=cache_decoders,
            file_ext=file_ext,
            file_pixel_format=file_pixel_format,
            buffer_pixel_format=buffer_pixel_format,
            fps=fps,
            encoder_options=encoder_options,
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
        decode_backend : Literal['torchcodec', 'pyav', 'auto'] = 'auto',
        cache_decoders : bool = False,
        capacity : Optional[int] = None,
        read_only : bool = True,
        multiprocessing : bool = False,
        **kwargs
    ) -> "VideoStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        assert read_only or (not multiprocessing), "VideoStorage does not support multiprocessing mode when loading a mutable storage"

        metadata_path = os.path.join(path, "video_metadata.json")
        if not os.path.exists(metadata_path):
            # Try legacy metadata filename for backward compatibility
            legacy_metadata_path = os.path.join(path, "image_metadata.json")
            if not os.path.exists(legacy_metadata_path):
                raise FileNotFoundError(f"Metadata file not found at {metadata_path} or {legacy_metadata_path}")
            
            # try:
            #     os.rename(legacy_metadata_path, os.path.join(path, "video_metadata.json"))
            # except:
            metadata_path = legacy_metadata_path

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        assert metadata["storage_type"] == __class__.__name__, \
            f"Expected storage type {__class__.__name__}, but found {metadata['storage_type']}"
        
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
        encoder_options = metadata.get("encoder_options", None)

        if "capacity" in metadata:
            capacity = None if metadata['capacity'] is None else int(metadata["capacity"])
        length = metadata["length"]

        return cls(
            single_instance_space,
            cache_filename=path,
            seek_mode=seek_mode,
            hardware_acceleration=hardware_acceleration,
            codec=codec,
            decode_backend=decode_backend,
            cache_decoders=cache_decoders,
            file_ext=file_ext,
            file_pixel_format=file_pixel_format,
            buffer_pixel_format=buffer_pixel_format,
            fps=fps,
            encoder_options=encoder_options,
            mutable=not read_only,
            capacity=capacity,
            length=length,
        )

    # ========== Instance Implementations ==========
    single_file_ext = None

    def _invalidate_decoder_cache(self):
        for decoder in self._decoder_cache.values():
            try:
                decoder.close()
            except Exception:
                pass
        self._decoder_cache.clear()

    def _get_active_segment_temp_filename(self) -> str:
        suffix = f".{self.file_ext}" if self.file_ext is not None else ""
        return os.path.join(self._cache_path, f".active-{uuid.uuid4().hex}.tmp{suffix}")

    @property
    def pending_segment_length(self) -> int:
        writer = getattr(self, "_active_segment_writer", None)
        return 0 if writer is None else len(writer)

    @property
    def has_open_segment(self) -> bool:
        return getattr(self, "_active_segment_start_index", None) is not None

    def open_segment_writer(self) -> "_VideoSegmentWriter":
        assert self.is_mutable, "Cannot open a segment on a read-only storage"
        return _VideoSegmentWriter(self)

    def mark_segment_start(self, start_index):
        assert self.is_mutable, "Cannot open a segment on a read-only storage"
        assert not self.has_open_segment, "A video segment is already open"

        normalized_start = int(start_index) if self.capacity is None else int(start_index) % self.capacity
        # Create the writer first: it may open the encoder eagerly and raise (e.g. bad codec),
        # which must not leave the storage with an open segment but no writer
        writer = _VideoSegmentWriter(self)
        self._active_segment_start_index = normalized_start
        self._active_segment_index = normalized_start
        self._active_segment_writer = writer

    def append(self, frame):
        if not self.has_open_segment:
            raise RuntimeError("Cannot append a video frame without an open segment")
        if self.capacity is not None and self.pending_segment_length >= self.capacity:
            raise RuntimeError("Active video segment is full; finalize it before appending more frames")

        try:
            self._active_segment_writer.append(frame)
        except Exception:
            self.abort_segment()
            raise
        if self.capacity is None:
            self._active_segment_index = self._active_segment_index + 1
        else:
            self._active_segment_index = (self._active_segment_index + 1) % self.capacity

    def mark_segment_end(self):
        if not self.has_open_segment:
            return
        try:
            self._active_segment_writer.commit(self._active_segment_start_index)
        finally:
            self._clear_active_segment_state()
            self._invalidate_decoder_cache()

    def abort_segment(self):
        if not self.has_open_segment:
            return
        self._active_segment_writer.abort()
        self._clear_active_segment_state()
        self._invalidate_decoder_cache()

    def _clear_active_segment_state(self):
        self._active_segment_start_index = None
        self._active_segment_index = None
        self._active_segment_writer = None

    def get_segments(self):
        return list(self._file_ranges)

    def _active_segment_physical_indices(self) -> List[int]:
        if not self.has_open_segment or self.pending_segment_length <= 0:
            return []
        start_index = int(self._active_segment_start_index)
        length = int(self.pending_segment_length)
        if self.capacity is None:
            return list(range(start_index, start_index + length))
        end_index = (start_index + length - 1) % self.capacity
        if start_index <= end_index:
            return list(range(start_index, end_index + 1))
        return list(range(start_index, self.capacity)) + list(range(0, end_index + 1))

    def _requested_physical_indices(self, index: Union[IndexableType, BArrayType]) -> List[int]:
        limit = self.capacity if self.capacity is not None else self.length
        if limit == 0:
            return []
        if isinstance(index, int):
            return [int(index)]
        if index is Ellipsis:
            return list(range(limit))
        if isinstance(index, slice):
            return list(range(*index.indices(limit)))
        if self.backend.is_backendarray(index):
            if self.backend.dtype_is_boolean(index.dtype):
                index = self.backend.nonzero(index)[0]
            else:
                index = (index + limit) % limit
            return [int(x) for x in self.backend.to_numpy(index)]
        return []

    def _overlaps_active_segment(self, index: Union[IndexableType, BArrayType]) -> bool:
        if not self.has_open_segment or self.pending_segment_length <= 0:
            return False
        active_indices = set(self._active_segment_physical_indices())
        requested_indices = self._requested_physical_indices(index)
        return any(requested_index in active_indices for requested_index in requested_indices)

    def get(self, index: Union[IndexableType, BArrayType]):
        if self._overlaps_active_segment(index):
            raise RuntimeError("Cannot read from VideoStorage while a segment is open; finalize or abort it first")
        return super().get(index)

    # Auto-selected codec per requested `base` (None = any codec family).
    _auto_codec_cache : Dict[Optional[str], str] = {}

    @staticmethod
    def _encoder_available(codec_name : str) -> bool:
        """Whether `codec_name` resolves to an encoder in the linked FFmpeg build.

        `codecs_available` also lists decoder-only entries (e.g. `vp8_qsv`), which
        cannot be used to write video files.
        """
        if codec_name not in codecs_available:
            return False
        try:
            av.codec.Codec(codec_name, "w")
        except Exception:
            return False
        return True

    @staticmethod
    def get_auto_codec(
        base : Optional[str] = None
    ) -> str:
        cache = __class__._auto_codec_cache
        if base in cache:
            return cache[base]
        
        # --- PyAV Implementation ---
        preferred_codecs = ["av1", "hevc", "h264", "mpeg4", "vp9", "vp8"] if base is None else [base]
        preferred_suffixes = ["_nvenc", "_vaapi", "_amf", "_qsv", "_videotoolbox"]

        # --- TorchCodec Implementation ---
        # preferred_codecs = ["hevc", "av1", "h264", "mpeg4", "vp9", "vp8"] if base is None else [base]
        # preferred_suffixes = ["_nvenc"]
        
        # Codec preference comes first, hardware suffix preference second:
        # the first available hardware encoder in that order wins.
        target_codec = next(
            (
                codec + suffix
                for codec in preferred_codecs
                for suffix in preferred_suffixes
                if __class__._encoder_available(codec + suffix)
            ),
            None,
        )

        if target_codec is None:
            target_codec = next(
                (codec for codec in preferred_codecs if __class__._encoder_available(codec)),
                None,
            )
        
        if target_codec is None:
            raise RuntimeError("No suitable video codec found in available codecs.")

        cache[base] = target_codec
        return target_codec

    def __init__(
        self,
        single_instance_space: BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType],
        cache_filename : Union[str, os.PathLike],
        seek_mode : Literal['exact', 'approximate'] = 'exact',
        hardware_acceleration : Optional[Union[Any, Literal['auto']]] = 'auto',
        codec : Union[str, Literal['auto']] = 'auto',
        decode_backend : Literal['torchcodec', 'pyav', 'auto'] = 'auto',
        cache_decoders : bool = False,
        file_ext : str = "mp4",
        file_pixel_format : Optional[str] = None,
        buffer_pixel_format : str = "rgb24",
        fps : int = 15,
        encoder_options : Optional[Dict[str, str]] = None,
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
            # TorchCodecVideoReader only decodes rgb24; other buffer formats (e.g. gray16le depth) need PyAV.
            if buffer_pixel_format == 'rgb24' and importlib.util.find_spec("torchcodec"):
                decode_backend = 'torchcodec'
            else:
                decode_backend = 'pyav'
        self.decode_backend = decode_backend

        self.fps = fps
        self.encoder_options = None if encoder_options is None else {str(k): str(v) for k, v in encoder_options.items()}
        self.file_pixel_format = file_pixel_format
        self.buffer_pixel_format = buffer_pixel_format
        self.cache_decoders = cache_decoders
        self._decoder_cache = {}

    #: libx265 clips shorter than this are encoded without B-frames. With B-frames, x265 only
    #: initializes its dts offset (`Encoder::m_bframeDelayTime`) once it receives frame
    #: `m_bframeDelay` (1, or 2 with b-pyramid), so 1-2 frame clips get dts computed from
    #: uninitialized memory: the mp4 muxer then often fails with "pts/dts pair unsupported".
    X265_MIN_BFRAME_CLIP_LENGTH = 3

    @property
    def _min_clip_length_for_default_options(self) -> int:
        """Frames a streamed clip holds back before its encoder opens with the default options."""
        return self.X265_MIN_BFRAME_CLIP_LENGTH if self.codec == "libx265" else 1

    def _encoder_options_for_clip(self, num_frames: Optional[int]) -> Optional[Dict[str, str]]:
        if num_frames is None or num_frames >= self._min_clip_length_for_default_options:
            return self.encoder_options
        options = dict(self.encoder_options or {})
        x265_params = options.get("x265-params", "")
        options["x265-params"] = f"{x265_params}:bframes=0" if x265_params else "bframes=0"
        return options

    def get_cached_decoder(self, filename: str):
        if (not self.cache_decoders) or (filename not in self._decoder_cache):
            if self.decode_backend == 'pyav':
                reader_cls = PyAvVideoReader
            elif self.decode_backend == 'torchcodec':
                reader_cls = TorchCodecVideoReader
            else:
                raise ValueError(f"Unknown decode_backend {self.decode_backend}")
            video_reader = reader_cls(
                backend=self.backend,
                filename=filename,
                buffer_pixel_format=self.buffer_pixel_format,
                hwaccel=self.hwaccel,
                seek_mode=self.seek_mode,
                device=self.device,
            )
            if self.cache_decoders and self.decode_backend != 'pyav': 
                # PyAV decoders have issues with caching as they hold file handles, so we only cache non-PyAV decoders
                self._decoder_cache[filename] = video_reader
            return video_reader
        else:
            return self._decoder_cache[filename]
    
    def get_from_file(self, filename : str, index : Union[IndexableType, BArrayType], total_length : int) -> BArrayType:
        
        with self.get_cached_decoder(filename) as video_reader:
            data = video_reader.read(index, total_length)
        if not self.cache_decoders:
            video_reader.close()
        return data

    def remove_index_range(self, start_index: int, end_index: int):
        super().remove_index_range(start_index, end_index)
        self._invalidate_decoder_cache()

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
        value_np = self.backend.to_numpy(value)
        container, output_stream = self._open_encoder(filename, len(value_np))
        try:
            for frame_np in value_np:
                self._encode_frame(container, output_stream, frame_np)
        except Exception:
            self._close_encoder(container, output_stream, flush=False)
            raise
        self._close_encoder(container, output_stream)

    def _open_encoder(self, filename : str, num_frames : Optional[int]):
        """Open `filename` for writing a `num_frames`-frame clip (`None`: not known yet); return `(container, stream)`."""
        logging.getLogger("libav").setLevel(self.PyAV_LOG_LEVEL)
        container = av.open(filename, mode='w')
        try:
            output_stream = container.add_stream(
                self.codec, rate=self.fps, options=self._encoder_options_for_clip(num_frames)
            )
            # (H, W, C) or (H, W)
            output_stream.width = self.single_instance_space.shape[1]
            output_stream.height = self.single_instance_space.shape[0]
            if self.file_pixel_format is not None:
                output_stream.pix_fmt = self.file_pixel_format
            # output_stream.time_base = Fraction(1, self.fps).limit_denominator(int(2**16 - 1))
        except Exception:
            container.close()
            av.logging.restore_default_callback()
            raise
        return container, output_stream

    def _encode_frame(self, container, output_stream, frame_np : np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(frame_np, format=self.buffer_pixel_format, channel_last=True)
        packets = output_stream.encode(frame)
        if packets:
            container.mux(packets)

    def _close_encoder(self, container, output_stream, flush : bool = True) -> None:
        """Flush the encoder (unless `flush` is False, to drop the clip) and close the file."""
        try:
            if flush:
                packets = output_stream.encode()
                if packets:
                    container.mux(packets)
        finally:
            try:
                if hasattr(output_stream, 'close'):
                    output_stream.close()
            finally:
                container.close()
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
        if self.has_open_segment:
            raise RuntimeError("Cannot dump VideoStorage while a segment is open; finalize or abort it first")
        assert os.path.exists(path) and os.path.samefile(path, self.cache_filename), \
            f"Dump path {path} does not match cache filename {self.cache_filename}"
        metadata = {
            "storage_type": __class__.__name__,
            "codec_base": self.codec if "_" not in self.codec else self.codec.split("_")[0],
            "file_ext": self.file_ext,
            "fps": self.fps,
            "file_pixel_format": self.file_pixel_format,
            "buffer_pixel_format": self.buffer_pixel_format,
            "encoder_options": self.encoder_options,
            "capacity": self.capacity,
            "length": self.length,
        }
        with open(os.path.join(path, "video_metadata.json"), "w") as f:
            json.dump(metadata, f)
        super().dumps(path)

    def clear(self):
        if self.has_open_segment:
            self.abort_segment()
        super().clear()
        self._invalidate_decoder_cache()

    def close(self):
        if self.has_open_segment:
            self.abort_segment()
        for decoder in self._decoder_cache.values():
            decoder.close()
        self._decoder_cache.clear()


class _VideoSegmentWriter(SegmentWriter):
    """
    Encodes the appended frames as they arrive into a temporary file next to the segment files;
    the commit renames it to the file of its index range.
    The first frames are held back until the clip is long enough for the default encoder options
    (see `VideoStorage.X265_MIN_BFRAME_CLIP_LENGTH`).
    """

    def __init__(self, storage : VideoStorage):
        super().__init__(storage)
        self.temp_filename = storage._get_active_segment_temp_filename()
        self._container = None
        self._stream = None
        self._held_frames : List[np.ndarray] = []
        if storage._min_clip_length_for_default_options <= 1:
            # Nothing to hold back: open now, so a bad codec / format fails right away.
            try:
                self._open_encoder(None)
            except Exception:
                self.abort() # remove any partially written temp file
                raise

    def _append(self, frame) -> None:
        storage = self.storage
        frame_np = storage.backend.to_numpy(frame) if storage.backend.is_backendarray(frame) else np.asarray(frame)
        try:
            if self._stream is not None:
                storage._encode_frame(self._container, self._stream, frame_np)
                return
            self._held_frames.append(frame_np)
            if len(self._held_frames) >= storage._min_clip_length_for_default_options:
                self._open_encoder(None)
        except Exception:
            self.abort()
            raise

    def _open_encoder(self, num_frames : Optional[int]) -> None:
        storage = self.storage
        self._container, self._stream = storage._open_encoder(self.temp_filename, num_frames)
        held_frames, self._held_frames = self._held_frames, []
        for frame_np in held_frames:
            storage._encode_frame(self._container, self._stream, frame_np)

    def _finish(self) -> None:
        if self.length == 0:
            return
        try:
            if self._stream is None:
                self._open_encoder(len(self._held_frames))
            container, stream = self._container, self._stream
            self._container, self._stream = None, None
            self.storage._close_encoder(container, stream)
        except Exception:
            self.abort()
            raise

    def commit(self, start_index : int) -> None:
        self.finish()
        if self.length == 0:
            self.abort()
            return
        storage = self.storage
        start_index, end_index = int(start_index), int(start_index) + self.length - 1
        if storage.capacity is not None:
            start_index, end_index = start_index % storage.capacity, end_index % storage.capacity
        storage.remove_index_range(start_index, end_index)
        os.replace(self.temp_filename, storage._make_filename(start_index, end_index))
        storage._add_file_range(start_index, end_index)
        storage._invalidate_decoder_cache()

    def abort(self) -> None:
        container, stream = self._container, self._stream
        self._container, self._stream, self._held_frames = None, None, []
        if container is not None:
            try:
                self.storage._close_encoder(container, stream, flush=False)
            except Exception:
                pass
        if os.path.exists(self.temp_filename):
            try:
                os.remove(self.temp_filename)
            except FileNotFoundError:
                pass
