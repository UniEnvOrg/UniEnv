import numpy as np
import pytest
import torch

from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.flattened import FlattenedStorage
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace


def _pick_available_encoder(av_module, preferred_names):
    for codec_name in preferred_names:
        if codec_name not in av_module.codec.codecs_available:
            continue
        try:
            av_module.codec.Codec(codec_name, "w")
        except Exception:
            continue
        return codec_name
    return None


def test_legacy_storage_append_and_segment_end_no_segments():
    space = BoxSpace(
        PyTorchComputeBackend,
        -10,
        10,
        torch.int64,
        shape=(2,),
    )
    storage = PytorchTensorStorage.create(space, capacity=4, is_memmap=False)

    assert storage.get_segments() is None
    storage.mark_segment_start(0)
    sample = torch.tensor([1, 2], dtype=torch.int64)
    storage.append(sample)
    assert torch.equal(storage.get(0), sample)
    storage.mark_segment_end()
    assert storage.get_segments() is None
    assert not storage.has_open_segment


def test_replay_buffer_append_with_flattened_storage_wrapper():
    space = BoxSpace(
        PyTorchComputeBackend,
        -10,
        10,
        torch.int64,
        shape=(2,),
    )
    rb = ReplayBuffer.create(
        FlattenedStorage,
        space,
        inner_storage_cls=PytorchTensorStorage,
        capacity=4,
        is_memmap=False,
    )

    sample = torch.tensor([3, 4], dtype=torch.int64)
    rb.append(sample)
    assert rb.storage.has_open_segment
    rb.mark_segment_end()

    assert rb.get_segments() is None
    assert torch.equal(rb.get_at(0), sample)


def test_video_storage_segmented_append_and_logical_segments(tmp_path):
    av = pytest.importorskip("av")
    codec = _pick_available_encoder(av, ("libx264rgb", "libx264"))
    if codec is None:
        pytest.skip("No x264 encoder is available")

    from unienv_data.storages.video_storage import VideoStorage

    file_pixel_format = "rgb24" if codec == "libx264rgb" else "yuv420p"
    space = BoxSpace(
        NumpyComputeBackend,
        0,
        255,
        np.uint8,
        shape=(4, 4, 3),
    )
    rb = ReplayBuffer.create(
        VideoStorage,
        space,
        cache_path=str(tmp_path / "video"),
        capacity=None,
        codec=codec,
        file_ext="mkv",
        file_pixel_format=file_pixel_format,
        buffer_pixel_format="rgb24",
        decode_backend="pyav",
        hardware_acceleration=None,
    )

    seg1 = [np.full((4, 4, 3), fill_value=v, dtype=np.uint8) for v in (16, 32)]
    seg2 = [np.full((4, 4, 3), fill_value=v, dtype=np.uint8) for v in (48, 64)]

    for frame in seg1:
        rb.append(frame)
    rb.mark_segment_end()
    for frame in seg2:
        rb.append(frame)
    rb.mark_segment_end()

    assert len(rb) == 4
    assert rb.storage.get_segments() == [(0, 1), (2, 3)]
    assert rb.get_segments() == [(0, 2), (2, 4)]
    np.testing.assert_array_equal(rb.get_at(0), seg1[0])
    np.testing.assert_array_equal(rb.get_at(3), seg2[1])


def test_video_storage_fixed_capacity_wrap_segments(tmp_path):
    av = pytest.importorskip("av")
    if "ffv1" not in av.codec.codecs_available:
        pytest.skip("ffv1 codec is not available")

    from unienv_data.storages.video_storage import VideoStorage

    space = BoxSpace(
        NumpyComputeBackend,
        0,
        65535,
        np.uint16,
        shape=(4, 4),
    )
    rb = ReplayBuffer.create(
        VideoStorage,
        space,
        cache_path=str(tmp_path / "video_wrap"),
        capacity=4,
        codec="ffv1",
        file_ext="mkv",
        file_pixel_format="gray16le",
        buffer_pixel_format="gray16le",
        decode_backend="pyav",
        hardware_acceleration=None,
    )

    first_segment = [np.full((4, 4), fill_value=v, dtype=np.uint16) for v in (100, 200, 300, 400)]
    second_segment = [np.full((4, 4), fill_value=v, dtype=np.uint16) for v in (500, 600, 700)]

    for frame in first_segment:
        rb.append(frame)
    rb.mark_segment_end()
    for frame in second_segment:
        rb.append(frame)
    rb.mark_segment_end()

    segments = rb.get_segments()
    assert segments is not None
    assert len(segments) == 2
    assert any(start > end for start, end in segments)
    assert any(start == 0 and end == 1 for start, end in segments)
    assert any(start == 1 and end == 0 for start, end in segments)
    np.testing.assert_array_equal(rb.get_at(0), first_segment[-1])


def test_replay_buffer_export_no_trajectory_buffer():
    import unienv_data.replay_buffer as replay_buffer_module

    assert hasattr(replay_buffer_module, "ReplayBuffer")
    assert not hasattr(replay_buffer_module, "TrajectoryReplayBuffer")
