"""ImageStorage specifics: encoding formats and Pillow compatibility."""

import warnings

import numpy as np
import pytest

from unienv_data.storages.image_storage import ImageStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace

B = NumpyComputeBackend


def _space() -> BoxSpace:
    return BoxSpace(B, 0, 255, np.uint8, shape=(6, 5, 3))


def _images(n: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, size=(n, 6, 5, 3), dtype=np.uint8)


def test_writing_does_not_use_deprecated_pillow_api(tmp_path):
    storage = ImageStorage.create(_space(), capacity=2, cache_path=str(tmp_path / "img"), format="PNG")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        storage.set(0, _images(1)[0])
