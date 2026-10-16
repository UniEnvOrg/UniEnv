"""``dict_storage_kwargs_from_space``: DictStorage layouts derived from ``Space.metadata``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import BinarySpace, BoxSpace, DictSpace
from unienv_interface.space.space_utils.metadata_utils import spaces_equal_with_metadata
from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages import (
    DictStorage,
    create_dict_storage_from_space,
    dict_storage_kwargs_from_space,
)
from unienv_data.storages.hdf5 import HDF5Storage
from unienv_data.storages.image_storage import ImageStorage
from unienv_data.storages.npz_storage import NPZStorage

from tests_common.storage_cases import random_batch
from tests_common.trees import assert_tree_equal

IMAGE_MD = {"kind": "image", "format": "rgb", "layout": "HWC", "fps": 12}


def obs_space() -> DictSpace:
    return DictSpace(NB, {
        "cameras": DictSpace(NB, {
            "front": BoxSpace(NB, 0, 255, np.uint8, shape=(8, 6, 3), metadata=dict(IMAGE_MD, source="front")),
            "wrist": BoxSpace(NB, 0, 255, np.uint8, shape=(8, 6, 3), metadata={"kind": "image", "format": "bgr"}),
            "chw": BoxSpace(NB, 0, 255, np.uint8, shape=(3, 8, 6), metadata={"kind": "image", "layout": "CHW"}),
            "float_image": BoxSpace(NB, 0.0, 1.0, np.float32, shape=(8, 6, 3), metadata={"kind": "image"}),
            "depth": BoxSpace(NB, 0.0, 10.0, np.float32, shape=(8, 6), metadata={"kind": "depth", "units": "m"}),
        }),
        "state": BoxSpace(NB, -1.0, 1.0, np.float32, shape=(7,), metadata={"kind": "proprio"}),
        "done": BinarySpace(NB, shape=()),
    }, metadata={"label": "obs"})


def test_image_leaves_go_to_image_storage_and_the_rest_to_hdf5():
    kwargs = dict_storage_kwargs_from_space(obs_space())
    assert kwargs["storage_cls_map"] == {
        "cameras/front": ImageStorage,
        "cameras/wrist": ImageStorage,
        "*": HDF5Storage,
    }
    assert kwargs["key_kwargs"] == {}
    assert kwargs["nested_separator"] == "/"


def test_overrides_per_kind_and_key():
    kwargs = dict_storage_kwargs_from_space(
        obs_space(),
        kind_storage_cls={"image": ImageStorage, "depth": NPZStorage},
        key_storage_cls={"cameras.wrist": None, "state": NPZStorage},
        default_storage_cls=NPZStorage,
        kind_kwargs={"image": {"format": "PNG"}, "depth": {"compressed": True}},
        key_kwargs={"cameras.front": {"quality": 90}, "state": {"compressed": False}},
        default_kwargs={"compressed": True},
        nested_separator=".",
    )
    assert kwargs["storage_cls_map"] == {
        "cameras.front": ImageStorage,
        "cameras.depth": NPZStorage,
        "state": NPZStorage,
        "*": NPZStorage,
    }
    assert kwargs["key_kwargs"] == {
        "cameras.front": {"format": "PNG", "quality": 90},
        "cameras.depth": {"compressed": True},
        "state": {"compressed": False},
        "*": {"compressed": True},
    }
    # No image routing at all
    assert dict_storage_kwargs_from_space(obs_space(), kind_storage_cls={})["storage_cls_map"] == {"*": HDF5Storage}


def test_video_storage_gets_the_declared_fps():
    from unienv_data.storages.video_storage import VideoStorage
    kwargs = dict_storage_kwargs_from_space(obs_space(), kind_storage_cls={"image": VideoStorage})
    assert kwargs["storage_cls_map"]["cameras/front"] is VideoStorage
    assert kwargs["key_kwargs"]["cameras/front"] == {"fps": 12}
    assert "cameras/wrist" not in kwargs["key_kwargs"]  # no fps declared
    explicit = dict_storage_kwargs_from_space(
        obs_space(), kind_storage_cls={"image": VideoStorage}, key_kwargs={"cameras/front": {"fps": 30}},
    )
    assert explicit["key_kwargs"]["cameras/front"] == {"fps": 30}


def test_requires_a_dict_space():
    with pytest.raises(TypeError, match="DictSpace"):
        dict_storage_kwargs_from_space(BoxSpace(NB, 0, 1, np.float32, shape=(2,)))


def _assert_layout(storage: DictStorage) -> None:
    assert isinstance(storage, DictStorage)
    assert isinstance(storage.storage_map["cameras/front"], ImageStorage)
    assert isinstance(storage.storage_map["cameras/wrist"], ImageStorage)
    residual = storage.storage_map["*"]
    assert isinstance(residual, HDF5Storage)
    assert set(residual.single_instance_space.keys()) == {"cameras", "state", "done"}
    assert set(residual.single_instance_space["cameras"].keys()) == {"chw", "float_image", "depth"}
    assert residual.single_instance_space.metadata == {"label": "obs"}
    assert residual.single_instance_space["cameras"]["depth"].metadata == {"kind": "depth", "units": "m"}


def test_replay_buffer_round_trip(tmp_path):
    space = obs_space()
    path = str(tmp_path / "rb")
    rb = ReplayBuffer.create(
        DictStorage, space, cache_path=path, capacity=None,
        # PNG: lossless, so the round trip can be compared exactly
        **dict_storage_kwargs_from_space(space, kind_kwargs={"image": {"format": "PNG"}}),
    )
    _assert_layout(rb.storage)
    data = random_batch(space, 4, seed=7)
    rb.extend(data)
    assert_tree_equal(rb.get_at(slice(None)), data)
    rb.dumps(path)
    rb.close()

    loaded = ReplayBuffer.load_from(path, backend=NB)
    try:
        _assert_layout(loaded.storage)
        assert len(loaded) == 4
        assert_tree_equal(loaded.get_at(slice(None)), data)
        assert spaces_equal_with_metadata(loaded.single_space, space)
    finally:
        loaded.close()


def test_default_jpeg_round_trip_is_close(tmp_path):
    space = DictSpace(NB, {
        "image": BoxSpace(NB, 0, 255, np.uint8, shape=(16, 16, 3), metadata={"kind": "image"}),
        "state": BoxSpace(NB, -1.0, 1.0, np.float32, shape=(2,)),
    })
    storage = create_dict_storage_from_space(space, cache_path=str(tmp_path / "s"), capacity=3)
    assert isinstance(storage.storage_map["image"], ImageStorage) and storage.storage_map["image"].format == "JPEG"
    smooth = np.broadcast_to(np.arange(16, dtype=np.uint8)[None, :, None] * 8, (3, 16, 16, 3)).copy()
    state = np.zeros((3, 2), np.float32)
    storage.set(slice(None), {"image": smooth, "state": state})
    got = storage.get(slice(None))
    np.testing.assert_array_equal(got["state"], state)
    assert np.abs(got["image"].astype(int) - smooth.astype(int)).max() <= 8
    storage.close()
