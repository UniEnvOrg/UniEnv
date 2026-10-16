import glob
import os

import numpy as np
import pytest
import torch

from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.backend_compat import ToBackendOrDeviceStorage
from unienv_data.storages.dict_storage import DictStorage
from unienv_data.storages.flattened import FlattenedStorage
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_data.storages.transformation import TransformedStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.transformations.chained_transform import ChainedTransformation
from unienv_interface.transformations.crop import CropTransformation
from unienv_interface.transformations.rescale import RescaleTransformation

av = pytest.importorskip("av")

DEPTH_SHAPE = (6, 8)


def _step_space():
    return DictSpace(
        PyTorchComputeBackend,
        {
            "state": BoxSpace(PyTorchComputeBackend, -10.0, 10.0, torch.float32, shape=(3,)),
            "depth": BoxSpace(PyTorchComputeBackend, 0.0, 65.535, torch.float32, shape=DEPTH_SHAPE),
            "episode_id": BoxSpace(PyTorchComputeBackend, 0, 1000, torch.int32, shape=(1,)),
        },
    )


def _make_buffer(path, capacity):
    """The collection layout in miniature: flattened memmap tensors + lossless FFV1 depth video."""
    if "ffv1" not in av.codec.codecs_available:
        pytest.skip("ffv1 codec is not available")
    from unienv_data.storages.video_storage import VideoStorage

    return ReplayBuffer.create(
        DictStorage,
        _step_space(),
        storage_cls_map={"*": FlattenedStorage, "depth": ToBackendOrDeviceStorage},
        type_kwargs={FlattenedStorage: {"inner_storage_cls": PytorchTensorStorage, "is_memmap": True}},
        key_kwargs={
            "depth": {
                "inner_storage_cls": TransformedStorage,
                "inner_storage_kwargs": {
                    "inner_storage_cls": VideoStorage,
                    "inner_storage_kwargs": {
                        "codec": "ffv1",
                        "file_ext": "mkv",
                        "file_pixel_format": "gray16le",
                        "buffer_pixel_format": "gray16le",
                        "decode_backend": "pyav",
                        "hardware_acceleration": None,
                    },
                    "data_transformation": ChainedTransformation([
                        CropTransformation(crop_low=0.0, crop_high=65.535),
                        RescaleTransformation(0, 65535, new_dtype=np.uint16),
                    ]),
                },
                "backend": NumpyComputeBackend,
            },
        },
        cache_path=str(path),
        capacity=capacity,
        multiprocessing=False,
    )


def _episodes(lengths, seed=0):
    rng = np.random.default_rng(seed)
    episodes = []
    for episode_id, length in enumerate(lengths):
        episodes.append([
            {
                "state": torch.as_tensor(rng.uniform(-1, 1, 3), dtype=torch.float32),
                # Multiples of 1 mm: exact through the uint16 depth encoding.
                "depth": torch.as_tensor(rng.integers(0, 65535, DEPTH_SHAPE) / 1000.0, dtype=torch.float32),
                "episode_id": torch.tensor([episode_id], dtype=torch.int32),
            }
            for _ in range(length)
        ])
    return episodes


def _stack(steps):
    return {key: torch.stack([step[key] for step in steps]) for key in steps[0]}


def _fill_with_writers(rb, episodes, commit_order):
    """Open one writer per episode, append round-robin (like a batched rollout), commit in `commit_order`."""
    writers = [rb.open_segment_writer() for _ in episodes]
    for t in range(max(len(episode) for episode in episodes)):
        for writer, episode in zip(writers, episodes):
            if t < len(episode):
                writer.append(episode[t])
    for index in commit_order:
        rb.commit_segment(writers[index])


def _assert_same_contents(rb_a, rb_b):
    assert len(rb_a) == len(rb_b)
    assert rb_a.offset == rb_b.offset
    assert rb_a.get_segments() == rb_b.get_segments()
    for i in range(len(rb_a)):
        a, b = rb_a.get_at(i), rb_b.get_at(i)
        for key in a:
            np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]), err_msg=f"{key} at {i}")


def _temp_files(path):
    return glob.glob(os.path.join(str(path), "**", ".active-*"), recursive=True)


def test_concurrent_segment_writers_match_extend(tmp_path):
    episodes = _episodes([5, 2, 1, 7])
    commit_order = [1, 3, 0, 2]
    rb_writers = _make_buffer(tmp_path / "writers", capacity=100)
    rb_extend = _make_buffer(tmp_path / "extend", capacity=100)

    _fill_with_writers(rb_writers, episodes, commit_order)
    for index in commit_order:
        rb_extend.extend(_stack(episodes[index]))

    _assert_same_contents(rb_writers, rb_extend)
    first = rb_writers.get_at(0)
    np.testing.assert_allclose(np.asarray(first["depth"]), np.asarray(episodes[1][0]["depth"]), atol=1e-6)
    assert _temp_files(tmp_path) == []

    # One video file per committed episode, named by its index range.
    depth_dir = os.path.dirname(glob.glob(str(tmp_path / "writers" / "**" / "*.mkv"), recursive=True)[0])
    assert sorted(os.path.basename(f) for f in glob.glob(os.path.join(depth_dir, "*.mkv"))) == sorted(
        ["0_1.mkv", "2_8.mkv", "9_13.mkv", "14_14.mkv"]
    )

    rb_writers.dumps(str(tmp_path / "writers"))
    loaded = ReplayBuffer.load_from(str(tmp_path / "writers"), backend=PyTorchComputeBackend, device="cpu")
    _assert_same_contents(loaded, rb_extend)


def test_segment_writers_wrap_fixed_capacity_like_extend(tmp_path):
    episodes = _episodes([6, 6, 3], seed=1)
    rb_writers = _make_buffer(tmp_path / "writers", capacity=10)
    rb_extend = _make_buffer(tmp_path / "extend", capacity=10)

    _fill_with_writers(rb_writers, episodes, [0, 1, 2])
    for episode in episodes:
        rb_extend.extend(_stack(episode))

    assert len(rb_writers) == 10
    _assert_same_contents(rb_writers, rb_extend)


def test_segment_writers_unbounded_capacity(tmp_path):
    if "ffv1" not in av.codec.codecs_available:
        pytest.skip("ffv1 codec is not available")
    from unienv_data.storages.video_storage import VideoStorage

    def make(path):
        return ReplayBuffer.create(
            VideoStorage,
            BoxSpace(NumpyComputeBackend, 0, 65535, np.uint16, shape=DEPTH_SHAPE),
            cache_path=str(path),
            capacity=None,
            codec="ffv1",
            file_ext="mkv",
            file_pixel_format="gray16le",
            buffer_pixel_format="gray16le",
            decode_backend="pyav",
            hardware_acceleration=None,
        )

    rng = np.random.default_rng(2)
    episodes = [list(rng.integers(0, 65535, (length, *DEPTH_SHAPE), dtype=np.uint16)) for length in (3, 4)]
    rb_writers, rb_extend = make(tmp_path / "writers"), make(tmp_path / "extend")

    _fill_with_writers(rb_writers, episodes, [1, 0])
    for index in (1, 0):
        rb_extend.extend(np.stack(episodes[index]))

    assert len(rb_writers) == 7
    assert rb_writers.get_segments() == rb_extend.get_segments()
    np.testing.assert_array_equal(rb_writers.get_at(slice(0, 7)), rb_extend.get_at(slice(0, 7)))
    np.testing.assert_array_equal(rb_writers.get_at(0), episodes[1][0])


def test_segment_writer_abort_and_early_finish(tmp_path):
    (aborted, kept) = _episodes([3, 4], seed=3)
    rb = _make_buffer(tmp_path / "rb", capacity=20)

    writer = rb.open_segment_writer()
    for step in aborted:
        writer.append(step)
    assert len(_temp_files(tmp_path)) == 1  # the depth clip being encoded
    writer.abort()
    assert _temp_files(tmp_path) == []

    writer = rb.open_segment_writer()
    for step in kept:
        writer.append(step)
    writer.finish()  # e.g. waiting for an earlier segment to be committed first
    with pytest.raises(RuntimeError):
        writer.append(kept[0])
    rb.commit_segment(writer)
    assert len(rb) == 4
    np.testing.assert_array_equal(np.asarray(rb.get_at(3)["state"]), np.asarray(kept[3]["state"]))
    assert _temp_files(tmp_path) == []

    # An empty writer commits nothing.
    rb.commit_segment(rb.open_segment_writer())
    assert len(rb) == 4


def test_segment_writer_rejects_foreign_writer_and_oversized_segment(tmp_path):
    rb = _make_buffer(tmp_path / "rb", capacity=3)
    other = _make_buffer(tmp_path / "other", capacity=3)
    (episode,) = _episodes([4], seed=4)

    writer = other.open_segment_writer()
    writer.append(episode[0])
    with pytest.raises(AssertionError):
        rb.commit_segment(writer)
    writer.abort()

    writer = rb.open_segment_writer()
    for step in episode:
        writer.append(step)
    with pytest.raises(ValueError):
        rb.commit_segment(writer)
    assert len(rb) == 0
    assert _temp_files(tmp_path) == []


def test_x265_segment_writer_holds_short_clips_for_bframe_free_encoding(tmp_path, monkeypatch):
    if "libx265" not in av.codec.codecs_available:
        pytest.skip("libx265 is not available")
    from unienv_data.storages.video_storage import VideoStorage

    space = BoxSpace(NumpyComputeBackend, 0, 255, np.uint8, shape=(64, 64, 3))
    storage = VideoStorage.create(
        space,
        cache_path=str(tmp_path / "rgb"),
        codec="libx265",
        file_pixel_format="yuv420p",
        buffer_pixel_format="rgb24",
        encoder_options={"preset": "fast", "x265-params": "keyint=15:log-level=error"},
        decode_backend="pyav",
        hardware_acceleration=None,
        capacity=100,
    )
    opened_with = []
    open_encoder = storage._open_encoder
    monkeypatch.setattr(
        storage, "_open_encoder", lambda filename, num_frames: opened_with.append(num_frames) or open_encoder(filename, num_frames)
    )
    # Smooth moving gradients: compressible, so the decoded frames stay close to them.
    y, x = np.mgrid[0:64, 0:64]
    frames = np.stack([
        np.stack([(3 * x + 5 * t) % 256, (2 * y + 3 * t) % 256, np.full_like(x, 128)], axis=-1)
        for t in range(20)
    ]).astype(np.uint8)

    start = 0
    for length in (1, 2, 20):
        writer = storage.open_segment_writer()
        for frame in frames[:length]:
            writer.append(frame)
        writer.commit(start)
        start += length
    # Short clips open their encoder at the end with their length (no B-frames);
    # longer clips stream once x265's B-frame delay is covered.
    assert opened_with == [1, 2, None]
    assert storage._encoder_options_for_clip(2)["x265-params"].endswith(":bframes=0")
    decoded = storage.get(slice(3, 23))
    assert decoded.shape == (20, 64, 64, 3)
    assert np.abs(decoded.astype(int) - frames.astype(int)).mean() < 10
