"""DictStorage routing, residual ('*') keys, separators and lifecycle."""

import numpy as np
import pytest

from unienv_data.storages.dict_storage import DictStorage
from unienv_data.storages.hdf5 import HDF5Storage
from unienv_data.storages.npz_storage import NPZStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace, DictSpace

from tests_common.trees import assert_tree_equal

B = NumpyComputeBackend


def _nested_space() -> DictSpace:
    return DictSpace(B, {
        "obs": DictSpace(B, {
            "pos": BoxSpace(B, -10.0, 10.0, np.float32, shape=(2,)),
            "vel": BoxSpace(B, -10.0, 10.0, np.float32, shape=(1,)),
        }),
        "reward": BoxSpace(B, -10.0, 10.0, np.float32, shape=()),
    })


def _sample(value: float) -> dict:
    return {
        "obs": {"pos": np.full((2,), value, np.float32), "vel": np.full((1,), value + 0.5, np.float32)},
        "reward": np.float32(value + 0.25),
    }


def _batch(values) -> dict:
    samples = [_sample(v) for v in values]
    return {
        "obs": {
            "pos": np.stack([s["obs"]["pos"] for s in samples]),
            "vel": np.stack([s["obs"]["vel"] for s in samples]),
        },
        "reward": np.stack([s["reward"] for s in samples]),
    }


@pytest.mark.parametrize("separator", ["/", "."])
def test_custom_nested_separator_with_residual_key_round_trip(tmp_path, separator):
    cache = tmp_path / "dict"
    storage = DictStorage.create(
        _nested_space(),
        {f"obs{separator}pos": NPZStorage, f"obs{separator}*": HDF5Storage, "*": HDF5Storage},
        capacity=3,
        cache_path=str(cache),
        nested_separator=separator,
    )
    assert set(storage.storage_map) == {f"obs{separator}pos", f"obs{separator}*", "*"}
    storage.set(0, _sample(1.0))
    storage.set(slice(1, 3), _batch([2.0, 3.0]))
    assert_tree_equal(storage.get(0), _sample(1.0))
    assert_tree_equal(storage.get(slice(None)), _batch([1.0, 2.0, 3.0]))
    storage.dumps(str(cache))
    storage.close()

    loaded = DictStorage.load_from(str(cache), _nested_space(), capacity=3)
    assert loaded.nested_separator == separator
    assert_tree_equal(loaded.get(np.array([2, 0])), _batch([3.0, 1.0]))
    assert_tree_equal(loaded.get(1), _sample(2.0))
    assert loaded.get_column(["obs", "pos"]) is loaded.storage_map[f"obs{separator}pos"]
    loaded.close()


def _flat_space() -> DictSpace:
    return DictSpace(B, {
        "a": BoxSpace(B, -10.0, 10.0, np.float32, shape=(2,)),
        "b": BoxSpace(B, -10.0, 10.0, np.float32, shape=()),
    })


def test_close_closes_child_storages(tmp_path):
    storage = DictStorage.create(_flat_space(), {"a": HDF5Storage, "b": HDF5Storage},
                                 capacity=2, cache_path=str(tmp_path / "dict"))
    children = list(storage.storage_map.values())
    assert all(child.root is not None for child in children)
    storage.close()
    assert all(child.root is None for child in children)


def test_garbage_collecting_parent_keeps_column_views_usable(tmp_path):
    import gc
    storage = DictStorage.create(_flat_space(), {"a": HDF5Storage, "b": HDF5Storage},
                                 capacity=2, cache_path=str(tmp_path / "dict"))
    storage.set(slice(None), {"a": np.ones((2, 2), np.float32), "b": np.array([1.0, 2.0], np.float32)})
    column = storage.get_column(["b"])
    del storage
    gc.collect()
    np.testing.assert_array_equal(column.get(slice(None)), np.array([1.0, 2.0], np.float32))
    column.close()


def _video_and_lowdim_space() -> DictSpace:
    return DictSpace(B, {
        "image": BoxSpace(B, 0, 255, np.uint8, shape=(8, 8, 3)),
        "reward": BoxSpace(B, -100.0, 100.0, np.float32, shape=()),
    })


def _video_and_lowdim_batch(start: int, n: int) -> dict:
    values = np.arange(start, start + n)
    return {
        "image": np.broadcast_to((values * 7 % 256).astype(np.uint8)[:, None, None, None], (n, 8, 8, 3)).copy(),
        "reward": values.astype(np.float32),
    }


@pytest.mark.parametrize("capacity", [None, 4])
def test_append_to_video_child_mixed_with_write_through_child(tmp_path, capacity):
    """
    A VideoStorage child holds appended frames back until the segment ends while an HDF5 child writes
    through, so the children report different pending-segment lengths; appending more than once used to
    trip DictStorage's "inconsistent pending-segment lengths" assertion.
    """
    from unienv_data.replay_buffer import ReplayBuffer
    from unienv_data.storages.video_storage import VideoStorage
    from tests_common.storage_video import LOSSLESS_RGB_KWARGS

    rb = ReplayBuffer.create(
        DictStorage, _video_and_lowdim_space(),
        storage_cls_map={"image": VideoStorage, "*": HDF5Storage},
        key_kwargs={"image": dict(LOSSLESS_RGB_KWARGS)},
        cache_path=str(tmp_path / "rb"), capacity=capacity,
    )
    data = _video_and_lowdim_batch(0, 7)
    for i in range(3):
        rb.append({key: value[i] for key, value in data.items()})
        # The buffered (video) child decides how much of the open segment is still pending
        assert rb.storage.pending_segment_length == i + 1
    rb.mark_segment_end()
    assert rb.storage.pending_segment_length == 0
    assert_tree_equal(rb.get_at(slice(None)), _video_and_lowdim_batch(0, 3))

    # One long segment: with a fixed capacity the buffer has to close it once it holds `capacity` samples
    for i in range(3, 7):
        rb.append({key: value[i] for key, value in data.items()})
    rb.mark_segment_end()
    expected_start = 0 if capacity is None else 7 - capacity
    assert_tree_equal(rb.get_at(slice(None)), _video_and_lowdim_batch(expected_start, 7 - expected_start))
    storage = rb.storage
    rb.close()
    storage.close()
