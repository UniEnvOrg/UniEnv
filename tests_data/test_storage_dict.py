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
