"""VideoStorage specifics: codec selection, decoding paths and segment handling."""

import gc
import json
import os

import numpy as np
import pytest

pytest.importorskip("av")

from unienv_data.storages import video_storage as video_storage_module
from unienv_data.storages.video_storage import VideoStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace

from tests_common.storage_video import (
    LOSSLESS_DEPTH_KWARGS,
    LOSSLESS_RGB_KWARGS,
    clean_video_codec_cache,  # noqa: F401  (fixture)
    reset_video_codec_cache,
)

B = NumpyComputeBackend

pytestmark = pytest.mark.usefixtures("clean_video_codec_cache")


def _rgb_space(h: int = 8, w: int = 8) -> BoxSpace:
    return BoxSpace(B, 0, 255, np.uint8, shape=(h, w, 3))


def _frames(n: int, seed: int = 0, h: int = 8, w: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


@pytest.fixture
def fake_encoders(monkeypatch):
    available = {"av1", "hevc", "h264", "mpeg4", "vp8", "h264_nvenc", "hevc_qsv", "vp8_qsv", "mpeg4_amf"}
    monkeypatch.setattr(VideoStorage, "_encoder_available", staticmethod(lambda name: name in available))
    return available


def test_auto_codec_prefers_codec_order_before_hardware_suffix(fake_encoders):
    # hevc ranks above h264 in the codec preference list, so hevc_qsv wins over h264_nvenc,
    # and the lowest-priority candidate (mpeg4_amf / vp8_qsv) must not win.
    assert VideoStorage.get_auto_codec() == "hevc_qsv"


def test_auto_codec_respects_base_and_caches_per_base(fake_encoders):
    assert VideoStorage.get_auto_codec(base="h264") == "h264_nvenc"
    assert VideoStorage.get_auto_codec(base="mpeg4") == "mpeg4_amf"
    assert VideoStorage.get_auto_codec(base="vp8") == "vp8_qsv"
    assert VideoStorage.get_auto_codec() == "hevc_qsv"
    # cached results stay per base
    assert VideoStorage.get_auto_codec(base="h264") == "h264_nvenc"


def test_auto_codec_falls_back_to_software_codec(monkeypatch):
    monkeypatch.setattr(VideoStorage, "_encoder_available", staticmethod(lambda name: name in {"h264", "vp9"}))
    assert VideoStorage.get_auto_codec() == "h264"
    assert VideoStorage.get_auto_codec(base="vp9") == "vp9"


def test_auto_codec_raises_without_any_encoder(monkeypatch):
    monkeypatch.setattr(VideoStorage, "_encoder_available", staticmethod(lambda name: False))
    with pytest.raises(RuntimeError, match="No suitable video codec"):
        VideoStorage.get_auto_codec()


def test_encoder_available_rejects_decoder_only_and_unknown_names():
    assert not VideoStorage._encoder_available("definitely-not-a-codec")
    if "ffv1" in video_storage_module.codecs_available:
        assert VideoStorage._encoder_available("ffv1")


def test_reset_helper_clears_cache(fake_encoders):
    VideoStorage.get_auto_codec()
    reset_video_codec_cache()
    fake_encoders.discard("hevc_qsv")
    assert VideoStorage.get_auto_codec() == "h264_nvenc"


@pytest.mark.parametrize("index", [
    np.array([2, 0, 2]),
    np.array([1, 1, 1, 1, 1, 1]),
    np.array([9, 0, 9, 0, 3, 3, 7, 7]),
], ids=["dup-sparse-path", "all-same-dense-path", "dup-dense-path"])
def test_get_with_duplicate_indices(tmp_path, index):
    frames = _frames(10)
    storage = VideoStorage.create(_rgb_space(), capacity=10, cache_path=str(tmp_path / "v"), **LOSSLESS_RGB_KWARGS)
    storage.set(slice(None), frames)
    np.testing.assert_array_equal(storage.get(index), frames[index])
    storage.close()


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

def _storage(tmp_path, capacity=6, name="v", **overrides):
    kwargs = dict(LOSSLESS_RGB_KWARGS)
    kwargs.update(overrides)
    return VideoStorage.create(_rgb_space(), capacity=capacity, cache_path=str(tmp_path / name), **kwargs)


def test_open_segment_blocks_overlapping_reads_dumps_and_is_aborted_on_close(tmp_path):
    frames = _frames(6)
    storage = _storage(tmp_path)
    storage.set(slice(None), frames)
    storage.mark_segment_start(2)
    storage.append(frames[0])
    storage.append(frames[1])
    assert storage.pending_segment_length == 2
    with pytest.raises(AssertionError):
        storage.mark_segment_start(0)
    with pytest.raises(RuntimeError, match="segment is open"):
        storage.get(3)
    with pytest.raises(RuntimeError, match="segment is open"):
        storage.get(np.array([True, False, True, False, False, False]))
    np.testing.assert_array_equal(storage.get(slice(4, 6)), frames[4:6])
    with pytest.raises(RuntimeError, match="segment is open"):
        storage.dumps(storage.cache_filename)
    temp_files = [f for f in os.listdir(storage.cache_filename) if f.startswith(".active-")]
    assert len(temp_files) == 1
    storage.close()
    assert not storage.has_open_segment
    assert not [f for f in os.listdir(storage.cache_filename) if f.startswith(".active-")]


def test_full_segment_rejects_more_frames(tmp_path):
    frames = _frames(3)
    storage = _storage(tmp_path, capacity=2)
    storage.mark_segment_start(0)
    storage.append(frames[0])
    storage.append(frames[1])
    with pytest.raises(RuntimeError, match="full"):
        storage.append(frames[2])
    storage.mark_segment_end()
    np.testing.assert_array_equal(storage.get(slice(None)), frames[:2])
    assert storage.get_segments() == [(0, 1)]


def test_empty_segment_leaves_no_files(tmp_path):
    storage = _storage(tmp_path, capacity=None)
    storage.mark_segment_start(0)
    storage.mark_segment_end()
    assert os.listdir(storage.cache_filename) == []
    assert storage.get_segments() == []


def test_read_only_storage_rejects_segments(tmp_path):
    storage = _storage(tmp_path)
    storage.set(slice(None), _frames(6))
    storage.dumps(storage.cache_filename)
    loaded = VideoStorage.load_from(storage.cache_filename, _rgb_space(), hardware_acceleration=None,
                                    decode_backend="pyav", codec="ffv1")
    with pytest.raises(AssertionError):
        loaded.mark_segment_start(0)
    with pytest.raises(RuntimeError):
        loaded.append(_frames(1)[0])


def test_invalid_codec_cleans_up_segment_temp_file(tmp_path):
    storage = _storage(tmp_path, codec="definitely-not-a-codec")
    with pytest.raises(Exception):
        storage.mark_segment_start(0)
    assert not storage.has_open_segment
    assert os.listdir(storage.cache_filename) == []


# ---------------------------------------------------------------------------
# Loading and metadata
# ---------------------------------------------------------------------------

def _dumped(tmp_path, capacity=6):
    frames = _frames(6)
    storage = _storage(tmp_path, capacity=capacity)
    if capacity is None:
        storage.extend_length(6)
    storage.set(slice(None), frames)
    storage.dumps(storage.cache_filename)
    path = storage.cache_filename
    storage.close()
    return path, frames


def test_load_from_legacy_metadata_name(tmp_path):
    path, frames = _dumped(tmp_path)
    os.rename(os.path.join(path, "video_metadata.json"), os.path.join(path, "image_metadata.json"))
    loaded = VideoStorage.load_from(path, _rgb_space(), hardware_acceleration=None, decode_backend="pyav")
    assert loaded.codec == "ffv1"
    np.testing.assert_array_equal(loaded.get(slice(None)), frames)


def test_load_from_missing_metadata_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        VideoStorage.load_from(str(tmp_path / "empty"), _rgb_space())


def test_load_with_auto_codec_and_hardware_uses_codec_base(tmp_path, monkeypatch):
    path, frames = _dumped(tmp_path)
    monkeypatch.setattr(VideoStorage, "_encoder_available", staticmethod(lambda name: name in {"ffv1", "h264"}))
    loaded = VideoStorage.load_from(path, _rgb_space(), hardware_acceleration="auto", decode_backend="pyav")
    assert loaded.codec == "ffv1"


def test_load_without_codec_base_requires_hardware_or_codec(tmp_path):
    path, _ = _dumped(tmp_path)
    meta_path = os.path.join(path, "video_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta.pop("codec_base")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    with pytest.raises(AssertionError, match="Codec base"):
        VideoStorage.load_from(path, _rgb_space(), hardware_acceleration=None, decode_backend="pyav")


def test_mutable_multiprocessing_is_rejected(tmp_path):
    with pytest.raises(AssertionError):
        _storage(tmp_path, multiprocessing=True)
    path, _ = _dumped(tmp_path, capacity=6)
    with pytest.raises(AssertionError):
        VideoStorage.load_from(path, _rgb_space(), read_only=False, multiprocessing=True,
                               hardware_acceleration=None, decode_backend="pyav", codec="ffv1")


def test_create_requires_new_cache_path(tmp_path):
    with pytest.raises(ValueError):
        VideoStorage.create(_rgb_space(), capacity=2, cache_path=None, **LOSSLESS_RGB_KWARGS)
    (tmp_path / "exists").mkdir()
    with pytest.raises(AssertionError):
        VideoStorage.create(_rgb_space(), capacity=2, cache_path=str(tmp_path / "exists"), **LOSSLESS_RGB_KWARGS)


def test_auto_decode_backend_falls_back_to_pyav(tmp_path, monkeypatch):
    import importlib.util
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "torchcodec" else real_find_spec(name, *a, **k))
    storage = _storage(tmp_path, decode_backend="auto")
    assert storage.decode_backend == "pyav"


def test_unknown_decode_backend_raises_on_read(tmp_path):
    storage = _storage(tmp_path, decode_backend="nope")
    storage.set(slice(None), _frames(6))
    with pytest.raises(ValueError, match="decode_backend"):
        storage.get(0)


# ---------------------------------------------------------------------------
# Decoding paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seek_mode", ["exact", "approximate"])
@pytest.mark.parametrize("cache_decoders", [False, True])
def test_seek_modes_and_decoder_cache(tmp_path, seek_mode, cache_decoders):
    frames = _frames(12, seed=3)
    storage = _storage(tmp_path, capacity=12, seek_mode=seek_mode, cache_decoders=cache_decoders)
    storage.set(slice(None), frames)
    for index in (0, 7, 11, 3):
        np.testing.assert_array_equal(storage.get(index), frames[index])
    np.testing.assert_array_equal(storage.get(np.array([10, 1])), frames[[10, 1]])
    np.testing.assert_array_equal(storage.get(slice(None, None, 3)), frames[::3])
    storage.close()


def test_depth_video_round_trip(tmp_path):
    space = BoxSpace(B, 0, 65535, np.uint16, shape=(4, 6))
    depth = np.random.default_rng(1).integers(0, 65536, size=(5, 4, 6), dtype=np.uint16)
    storage = VideoStorage.create(space, capacity=None, cache_path=str(tmp_path / "d"), **LOSSLESS_DEPTH_KWARGS)
    storage.extend_length(5)
    storage.set(slice(None), depth)
    np.testing.assert_array_equal(storage.get(slice(None)), depth)
    np.testing.assert_array_equal(storage.get(2), depth[2])
    storage.close()


def test_torch_backend_frames(tmp_path):
    import torch
    from unienv_interface.backends.pytorch import PyTorchComputeBackend

    space = BoxSpace(PyTorchComputeBackend, 0, 255, torch.uint8, shape=(8, 8, 3), device="cpu")
    frames = torch.from_numpy(_frames(4))
    storage = VideoStorage.create(space, capacity=4, cache_path=str(tmp_path / "t"), **LOSSLESS_RGB_KWARGS)
    storage.set(slice(None), frames)
    assert torch.equal(storage.get(slice(None)), frames)
    assert torch.equal(storage.get(torch.tensor([3, 0])), frames[[3, 0]])
    assert torch.equal(storage.get(1), frames[1])
    storage.mark_segment_start(1)
    storage.append(frames[0])
    storage.mark_segment_end()
    assert torch.equal(storage.get(1), frames[0])
    storage.close()


def test_hwaccel_selection(monkeypatch):
    from unienv_data.storages.video_storage import PyAvVideoReader

    for devices, expected in [(["d3d11va", "cuda"], "d3d11va"), (["cuda"], "cuda"), (["vaapi"], "vaapi"),
                              (["videotoolbox"], "videotoolbox"), (["vulkan"], "vulkan")]:
        monkeypatch.setattr(PyAvVideoReader, "available_hwdevices", staticmethod(lambda d=devices: d))
        accel = PyAvVideoReader.get_auto_hwaccel()
        assert accel is not None
    monkeypatch.setattr(PyAvVideoReader, "available_hwdevices", staticmethod(lambda: []))
    assert PyAvVideoReader.get_auto_hwaccel() is None
    assert isinstance(PyAvVideoReader.available_hwdevices(), list)


@pytest.fixture(autouse=True)
def isolated_failed_hwdevices(monkeypatch):
    """Each test starts with no remembered hardware-decoder failures."""
    from unienv_data.storages.video_storage import PyAvVideoReader
    monkeypatch.setattr(PyAvVideoReader, "_failed_hwaccels", set())
    monkeypatch.setattr(PyAvVideoReader, "_auto_hwaccels", {})


def _failing_hwaccel_open(monkeypatch):
    """Make ``av.open`` fail whenever a hardware accelerator is requested."""
    import av

    real_open, calls = av.open, []

    def fake_open(*args, hwaccel=None, **kwargs):
        calls.append(hwaccel)
        if hwaccel is not None:
            raise av.error.ExternalError(542398533, "hardware device init failed")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(av, "open", fake_open)
    return calls


def test_reader_falls_back_to_software_when_hwaccel_init_fails(tmp_path, monkeypatch):
    from av.codec.hwaccel import HWAccel
    from unienv_data.storages.video_storage import PyAvVideoReader

    frames = _frames(6)
    storage = _storage(tmp_path)
    storage.set(slice(None), frames)
    filename = storage._make_filename(0, 5)
    calls = _failing_hwaccel_open(monkeypatch)
    monkeypatch.setattr(PyAvVideoReader, "get_auto_hwaccel",
                        staticmethod(lambda: HWAccel(device_type="cuda", allow_software_fallback=True)))

    with PyAvVideoReader(B, filename, buffer_pixel_format="rgb24", hwaccel="auto") as reader:
        np.testing.assert_array_equal(next(reader), frames[0])
    assert calls[0] is not None and calls[1] is None


def test_failed_hwaccel_device_is_not_retried(tmp_path, monkeypatch):
    from av.codec.hwaccel import HWAccel
    from unienv_data.storages.video_storage import PyAvVideoReader

    storage = _storage(tmp_path)
    storage.set(slice(None), _frames(6))
    filename = storage._make_filename(0, 5)
    calls = _failing_hwaccel_open(monkeypatch)
    cuda = HWAccel(device_type="cuda", allow_software_fallback=True)

    PyAvVideoReader(B, filename, hwaccel=cuda).close()
    assert calls == [cuda, None]
    PyAvVideoReader(B, filename, hwaccel=cuda).close()
    assert calls == [cuda, None, None]  # straight to software decoding

    # ``hwaccel='auto'`` skips a failed device and tries the next available one
    monkeypatch.setattr(PyAvVideoReader, "available_hwdevices", staticmethod(lambda: ["cuda", "vaapi"]))
    auto_cuda = PyAvVideoReader.get_auto_hwaccel()
    assert PyAvVideoReader.get_auto_hwaccel() is auto_cuda
    PyAvVideoReader(B, filename, hwaccel="auto").close()
    auto_vaapi = PyAvVideoReader.get_auto_hwaccel()
    assert auto_vaapi is not None and auto_vaapi is not auto_cuda


def test_forked_child_does_not_retry_a_failed_hwaccel_device(tmp_path, monkeypatch):
    # Re-initialising a device whose initialisation failed before the fork can crash the
    # child inside FFmpeg (seen with CUDA hidden via CUDA_VISIBLE_DEVICES), so the child
    # must inherit the failure and decode in software.
    import multiprocessing as mp
    from av.codec.hwaccel import HWAccel
    from unienv_data.storages.video_storage import PyAvVideoReader

    frames = _frames(6)
    storage = _storage(tmp_path)
    storage.set(slice(None), frames)
    filename = storage._make_filename(0, 5)
    calls = _failing_hwaccel_open(monkeypatch)
    cuda = HWAccel(device_type="cuda", allow_software_fallback=True)
    PyAvVideoReader(B, filename, hwaccel=cuda).close()

    def child(queue):
        del calls[:]
        with PyAvVideoReader(B, filename, buffer_pixel_format="rgb24", hwaccel=cuda) as reader:
            frame = next(reader)
        queue.put(([c is None for c in calls], frame))

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=child, args=(queue,))
    proc.start()
    opened_in_software, frame = queue.get(timeout=60)
    proc.join(timeout=30)
    assert opened_in_software == [True]
    np.testing.assert_array_equal(frame, frames[0])


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_reader_without_software_fallback_raises_and_deletes_cleanly(tmp_path, monkeypatch):
    import av
    from av.codec.hwaccel import HWAccel
    from unienv_data.storages.video_storage import PyAvVideoReader

    storage = _storage(tmp_path)
    storage.set(slice(None), _frames(6))
    _failing_hwaccel_open(monkeypatch)
    strict = HWAccel(device_type="cuda", allow_software_fallback=False)
    with pytest.raises(av.error.FFmpegError):
        PyAvVideoReader(B, storage._make_filename(0, 5), hwaccel=strict)
    gc.collect()  # the half-built reader's __del__ must not raise


def test_reader_frame_index_conversions(tmp_path):
    from unienv_data.storages.video_storage import PyAvVideoReader

    frames = _frames(6)
    storage = _storage(tmp_path)
    storage.set(slice(None), frames)
    filename = storage._make_filename(0, 5)
    reader = PyAvVideoReader(B, filename, buffer_pixel_format="rgb24")
    with reader:
        for i in range(6):
            assert reader.pts_to_frame_index(reader.frame_index_to_pts(i)) == i
        with pytest.raises(IndexError):
            reader.frame_index_to_pts(-1)
        with pytest.raises(IndexError):
            reader.seek(-1)
        reader.seek(4)
        assert reader.tell() == 4
        np.testing.assert_array_equal(next(reader), frames[4])
        np.testing.assert_array_equal(np.stack(list(reader)), frames[5:])
    reader.close()
    reader.close()
    assert PyAvVideoReader._to_fraction(None) is None
    assert PyAvVideoReader._to_fraction("not-a-number") is None
    from fractions import Fraction
    assert PyAvVideoReader._to_fraction(Fraction(1, 3)) == Fraction(1, 3)
    assert PyAvVideoReader._to_fraction(0.5) == Fraction(1, 2)


def _pretend_torchcodec_installed(monkeypatch):
    import importlib.util
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: object() if name == "torchcodec" else real_find_spec(name, *a, **k))


def test_auto_decode_backend_uses_pyav_for_non_rgb_buffers(tmp_path, monkeypatch):
    # TorchCodecVideoReader only supports rgb24, so depth (gray16le) videos must not pick it.
    _pretend_torchcodec_installed(monkeypatch)
    space = BoxSpace(B, 0, 65535, np.uint16, shape=(4, 6))
    kwargs = dict(LOSSLESS_DEPTH_KWARGS, decode_backend="auto")
    depth = VideoStorage.create(space, capacity=2, cache_path=str(tmp_path / "d"), **kwargs)
    assert depth.decode_backend == "pyav"
    rgb = _storage(tmp_path, name="rgb", decode_backend="auto")
    assert rgb.decode_backend == "torchcodec"


# ---------------------------------------------------------------------------
# TorchCodec adapter (exercised with a minimal in-process stand-in for torchcodec)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_torchcodec(monkeypatch):
    import contextlib
    import sys
    import types

    import av
    import torch

    class _Frames:
        def __init__(self, data):
            self.data = data

    class VideoDecoder:
        instances = []

        def __init__(self, filename, device="cpu", seek_mode="exact"):
            with av.open(filename) as container:
                frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            self._frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # (N, C, H, W)
            self.device = device
            VideoDecoder.instances.append(self)

        def __len__(self):
            return self._frames.shape[0]

        def get_frame_at(self, index):
            return _Frames(self._frames[index])

        def get_frames_in_range(self, start, stop, step=1):
            return _Frames(self._frames[start:stop:step])

        def get_frames_at(self, indices):
            assert indices.dtype == torch.int64
            return _Frames(self._frames[indices])

    @contextlib.contextmanager
    def set_cuda_backend(name):
        yield

    module = types.ModuleType("torchcodec")
    decoders = types.ModuleType("torchcodec.decoders")
    decoders.VideoDecoder = VideoDecoder
    decoders.set_cuda_backend = set_cuda_backend
    module.decoders = decoders
    monkeypatch.setitem(sys.modules, "torchcodec", module)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    return VideoDecoder


@pytest.mark.parametrize("cache_decoders", [False, True])
def test_torchcodec_reader_index_types(tmp_path, fake_torchcodec, cache_decoders):
    frames = _frames(6)
    storage = _storage(tmp_path, decode_backend="torchcodec", cache_decoders=cache_decoders)
    storage.set(slice(None), frames)
    np.testing.assert_array_equal(storage.get(2), frames[2])
    np.testing.assert_array_equal(storage.get(slice(1, 5, 2)), frames[1:5:2])
    np.testing.assert_array_equal(storage.get(Ellipsis), frames)
    np.testing.assert_array_equal(storage.get(np.array([5, 0, 5])), frames[[5, 0, 5]])
    np.testing.assert_array_equal(storage.get(np.array([True, False, False, False, False, True])), frames[[0, 5]])
    if cache_decoders:
        assert len(fake_torchcodec.instances) == 1
    storage.close()


def test_torchcodec_reader_with_torch_backend_and_hwaccel(tmp_path, fake_torchcodec, monkeypatch):
    import torch
    from unienv_data.storages.video_storage import TorchCodecVideoReader
    from unienv_interface.backends.pytorch import PyTorchComputeBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert TorchCodecVideoReader.get_auto_hwaccel() is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert TorchCodecVideoReader.get_auto_hwaccel() == "beta"

    space = BoxSpace(PyTorchComputeBackend, 0, 255, torch.uint8, shape=(8, 8, 3), device="cpu")
    frames = torch.from_numpy(_frames(4))
    kwargs = dict(LOSSLESS_RGB_KWARGS, decode_backend="torchcodec", hardware_acceleration="auto")
    storage = VideoStorage.create(space, capacity=4, cache_path=str(tmp_path / "t"), **kwargs)
    storage.set(slice(None), frames)
    assert torch.equal(storage.get(torch.tensor([3, 1])).cpu(), frames[[3, 1]])
    assert fake_torchcodec.instances[-1].device == "cuda"  # hardware decoding was requested
    storage.close()


def test_torchcodec_reader_requires_rgb24(tmp_path, fake_torchcodec):
    from unienv_data.storages.video_storage import TorchCodecVideoReader

    with pytest.raises(AssertionError):
        TorchCodecVideoReader(B, "unused.mkv", buffer_pixel_format="gray16le")
