"""File-per-sample (NPZ/Image) and file-per-episode (Video) storage internals."""

import json
import os

import numpy as np
import pytest

from unienv_data.storages.image_storage import ImageStorage
from unienv_data.storages.npz_storage import NPZStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace

B = NumpyComputeBackend

av = pytest.importorskip("av")
from unienv_data.storages.video_storage import VideoStorage  # noqa: E402

from tests_common.storage_video import LOSSLESS_RGB_KWARGS, clean_video_codec_cache  # noqa: E402,F401

pytestmark = pytest.mark.usefixtures("clean_video_codec_cache")


def _space() -> BoxSpace:
    return BoxSpace(B, 0, 255, np.uint8, shape=(4, 4, 3))


def _frames(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, 4, 4, 3), dtype=np.uint8)


FILE_BACKED = {
    "npz": (NPZStorage, {}),
    "image-png": (ImageStorage, {"format": "PNG"}),
    "video": (VideoStorage, LOSSLESS_RGB_KWARGS),
}


@pytest.mark.parametrize("name", list(FILE_BACKED))
@pytest.mark.parametrize("capacity", [None, 4], ids=["unbounded", "fixed"])
@pytest.mark.parametrize("index", [slice(0, 0), slice(3, 1), np.array([], dtype=np.int64)],
                         ids=["empty-slice", "reversed-slice", "empty-array"])
def test_empty_selection_returns_empty_batch(tmp_path, name, capacity, index):
    cls, kwargs = FILE_BACKED[name]
    storage = cls.create(_space(), capacity=capacity, cache_path=str(tmp_path / name), **kwargs)
    for _ in range(2):  # both before and after data exists
        result = storage.get(index)
        assert result.shape == (0, 4, 4, 3)
        assert result.dtype == np.uint8
        storage.set(index, np.zeros((0, 4, 4, 3), np.uint8))
        if capacity is None:
            storage.extend_length(4)
        storage.set(slice(0, 4), _frames(4))
    storage.close()
