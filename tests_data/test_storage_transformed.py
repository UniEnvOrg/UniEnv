"""TransformedStorage behaviour beyond the generic storage contract."""

import numpy as np
import pytest

from unienv_data.storages.hdf5 import HDF5Storage
from unienv_data.storages.transformation import TransformedStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.transformations import IdentityTransformation

from tests_common.trees import assert_tree_equal

B = NumpyComputeBackend


def _dict_space() -> DictSpace:
    return DictSpace(B, {
        "a": BoxSpace(B, -10.0, 10.0, np.float32, shape=(2,)),
        "b": BoxSpace(B, -10.0, 10.0, np.float32, shape=()),
    })


@pytest.fixture
def dict_storage(tmp_path):
    storage = TransformedStorage.create(
        _dict_space(),
        HDF5Storage,
        data_transformation=IdentityTransformation(),
        capacity=4,
        cache_path=str(tmp_path / "transformed"),
    )
    storage.set(slice(None), {
        "a": np.arange(8, dtype=np.float32).reshape(4, 2),
        "b": np.arange(4, dtype=np.float32) * 10,
    })
    yield storage
    storage.close()


@pytest.mark.parametrize("index", [slice(1, 3), Ellipsis, np.array([3, 0]), np.array([True, False, True, False])],
                         ids=["slice", "ellipsis", "int-array", "bool-mask"])
def test_get_flattened_batched_with_dict_space(dict_storage, index):
    batched = dict_storage.get(index)
    n = sbu.batch_size_data(batched)
    expected = sfu.flatten_data(sbu.batch_space(dict_storage.single_instance_space, n), batched, start_dim=1)
    flat = dict_storage.get_flattened(index)
    assert flat.shape == (n, 3)
    np.testing.assert_array_equal(flat, expected)


def test_get_flattened_single_with_dict_space(dict_storage):
    np.testing.assert_array_equal(dict_storage.get_flattened(1), np.array([2.0, 3.0, 10.0], dtype=np.float32))


def test_set_flattened_batched_round_trip(dict_storage):
    new_rows = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    dict_storage.set_flattened(slice(0, 2), new_rows)
    np.testing.assert_array_equal(dict_storage.get_flattened(slice(0, 2)), new_rows)
    assert_tree_equal(dict_storage.get(0), {"a": np.array([1.0, 2.0], np.float32), "b": np.float32(3.0)})

    dict_storage.set_flattened(3, np.array([7.0, 8.0, 9.0], dtype=np.float32))
    np.testing.assert_array_equal(dict_storage.get_flattened(3), np.array([7.0, 8.0, 9.0], dtype=np.float32))


def test_close_closes_inner_storage(dict_storage):
    inner = dict_storage.inner_storage
    assert inner.root is not None
    dict_storage.close()
    assert inner.root is None


def test_to_backend_storage_close_closes_inner_storage(tmp_path):
    import torch
    from unienv_data.storages.backend_compat import ToBackendOrDeviceStorage
    from unienv_interface.backends.pytorch import PyTorchComputeBackend

    space = BoxSpace(PyTorchComputeBackend, -1.0, 1.0, torch.float32, shape=(2,), device="cpu")
    storage = ToBackendOrDeviceStorage.create(
        space, HDF5Storage, capacity=2, cache_path=str(tmp_path / "to_numpy"), backend=NumpyComputeBackend,
    )
    assert isinstance(storage, ToBackendOrDeviceStorage)
    inner = storage.inner_storage
    storage.close()
    assert inner.root is None


def test_cache_filename_reports_the_storage_directory(tmp_path, dict_storage):
    assert dict_storage.cache_filename == str(tmp_path / "transformed")
    dict_storage.dumps(dict_storage.cache_filename)
    loaded = TransformedStorage.load_from(dict_storage.cache_filename, _dict_space(), capacity=4)
    assert loaded.cache_filename == str(tmp_path / "transformed")
    loaded.close()
