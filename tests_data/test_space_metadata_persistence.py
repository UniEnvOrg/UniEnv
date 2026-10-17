"""``Space.metadata`` survives storages, ReplayBuffer ``dumps`` / ``load_from`` and the batch views."""

from __future__ import annotations

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import BoxSpace, DictSpace, Space
from unienv_interface.space.space_utils.metadata_utils import spaces_equal_with_metadata
from unienv_data.replay_buffer import ReplayBuffer

from tests_common.storage_cases import STORAGE_CASES, StorageCase, random_batch
from tests_common.storage_video import clean_video_codec_cache  # noqa: F401  (fixture)

pytestmark = pytest.mark.usefixtures("clean_video_codec_cache")

N = 3


def annotate(space: Space, path: str = "root") -> Space:
    """Give ``space`` and each nested dict entry its own metadata (images get the image vocabulary)."""
    if isinstance(space, DictSpace):
        space = DictSpace(
            space.backend,
            {key: annotate(sub, f"{path}/{key}") for key, sub in space.spaces.items()},
            device=space.device,
        )
    if isinstance(space, BoxSpace) and len(space.shape) == 3 and space.shape[-1] == 3:
        return space.with_metadata(kind="image", format="rgb", layout="HWC", source=path, fps=15)
    return space.with_metadata(label=path, units=["m", "s"])


def _create_kwargs(case: StorageCase, capacity):
    kwargs = dict(case.create_kwargs() if callable(case.create_kwargs) else case.create_kwargs)
    if capacity is None:
        kwargs.update(case.unbounded_create_kwargs)
    return kwargs


@pytest.mark.parametrize("case", [pytest.param(c, id=c.id, marks=list(c.marks)) for c in STORAGE_CASES])
def test_replay_buffer_round_trip_keeps_metadata(case: StorageCase, tmp_path):
    space = annotate(case.make_space())
    capacity = case.capacities[-1]
    cache_path = str(tmp_path / "rb") if case.uses_cache_path else None
    rb = ReplayBuffer.create(
        case.storage_cls(), space,
        cache_path=cache_path, capacity=capacity,
        **_create_kwargs(case, capacity),
    )
    try:
        assert spaces_equal_with_metadata(rb.storage.single_instance_space, space)
        assert spaces_equal_with_metadata(rb.single_space, space)
        rb.extend(random_batch(space, N, seed=1))
        if case.buffered_segments:
            rb.mark_segment_end()
        if cache_path is None:
            return  # purely in-memory: nothing to persist
        rb.dumps(cache_path)
    finally:
        rb.close()

    restored_space = ReplayBuffer.get_space_from_path(cache_path, backend=space.backend, device=space.device)
    assert spaces_equal_with_metadata(restored_space, space)

    loaded = ReplayBuffer.load_from(
        cache_path, backend=space.backend, device=space.device, **case.load_kwargs,
    )
    try:
        assert len(loaded) == N
        assert spaces_equal_with_metadata(loaded.storage.single_instance_space, space)
        assert spaces_equal_with_metadata(loaded.single_space, space)
    finally:
        loaded.close()


def test_load_from_with_explicit_space_uses_that_space(tmp_path):
    """An explicitly passed space wins (equality ignores metadata, so differing descriptions are accepted)."""
    space = BoxSpace(NB, -1.0, 1.0, np.float32, shape=(2,), metadata={"kind": "action"})
    hdf5 = next(case for case in STORAGE_CASES if case.id == "hdf5-box")
    rb = ReplayBuffer.create(hdf5.storage_cls(), space, cache_path=str(tmp_path / "rb"), capacity=4)
    rb.extend(random_batch(space, 2))
    rb.dumps(str(tmp_path / "rb"))
    rb.close()
    relabelled = space.with_metadata(kind="proprio")
    loaded = ReplayBuffer.load_from(str(tmp_path / "rb"), backend=NB, single_instance_space=relabelled)
    try:
        assert loaded.single_space is relabelled
        assert loaded.single_space.metadata == {"kind": "proprio"}
    finally:
        loaded.close()


def _image_obs_space() -> DictSpace:
    return DictSpace(NB, {
        "image": BoxSpace(NB, 0, 255, np.uint8, shape=(4, 4, 3), metadata={"kind": "image", "format": "rgb", "layout": "HWC"}),
        "state": BoxSpace(NB, -1.0, 1.0, np.float32, shape=(2,), metadata={"kind": "proprio"}),
    }, metadata={"label": "obs"})


def test_batch_views_keep_metadata(tmp_path):
    from unienv_data.batches import FrameStackedBatch, SliceStackedBatch, ToBackendOrDeviceBatch, TransformedBatch
    from unienv_interface.transformations import DictTransformation, RescaleTransformation

    space = _image_obs_space()
    hdf5 = next(case for case in STORAGE_CASES if case.id == "hdf5-nested-dict")
    rb = ReplayBuffer.create(hdf5.storage_cls(), space, cache_path=str(tmp_path / "rb"), capacity=8)
    rb.extend(random_batch(space, 6))

    stacked = FrameStackedBatch(rb, prefetch_horizon=2)
    assert stacked.single_space.metadata == {"label": "obs"}
    assert stacked.single_space["image"].shape == (3, 4, 4, 3)
    assert stacked.single_space["image"].metadata == space["image"].metadata
    assert stacked.single_space["state"].metadata == {"kind": "proprio"}

    sliced = SliceStackedBatch(rb, np.array([-1, 0]))
    assert sliced.single_space["image"].metadata == space["image"].metadata

    transformed = TransformedBatch(rb, DictTransformation({"state": RescaleTransformation(0.0, 1.0)}))
    assert transformed.single_space["state"].metadata == {"kind": "proprio"}
    assert transformed.single_space["image"].metadata == space["image"].metadata

    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend
    moved = ToBackendOrDeviceBatch(rb, backend=PyTorchComputeBackend, device="cpu")
    assert spaces_equal_with_metadata(moved.single_space.to(NB), space)
    rb.close()
