"""ToBackendOrDeviceBatch: backend and/or device conversion views."""

import numpy as np
import pytest
import torch

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
from unienv_interface.space import BoxSpace, TupleSpace
from unienv_interface.space.space_utils import flatten_utils as sfu
from unienv_data.batches import ToBackendOrDeviceBatch
from unienv_data.batches.backend_compat import data_to

from tests_common.batch_helpers import ArrayBatch, make_rb, numpy_array_batch, rb_data
from tests_common.trees import assert_tree_equal, to_numpy


def _all_torch_on(tree, device):
    leaves = tree.values() if isinstance(tree, dict) else [tree]
    for leaf in leaves:
        assert isinstance(leaf, torch.Tensor)
        assert leaf.device == torch.device(device)


def test_device_only_conversion():
    # Regression: backend=None (device-only conversion) dereferenced target_backend.
    rb = make_rb(6)
    view = ToBackendOrDeviceBatch(rb, device="cpu")
    assert view.backend is TB and view.device == "cpu"
    idx = torch.tensor([1, 4])
    out = view.get_at(idx)
    _all_torch_on(out, "cpu")
    assert_tree_equal(out, rb.get_at(idx))
    assert_tree_equal(view.get_at(2), rb.get_at(2))
    assert_tree_equal(view.get_flattened_at(idx), rb.get_flattened_at(idx))
    data, meta = view.get_at_with_metadata(idx)
    assert_tree_equal(data, rb.get_at(idx)) and meta is None
    flat, meta = view.get_flattened_at_with_metadata(idx)
    assert_tree_equal(flat, rb.get_flattened_at(idx)) and meta is None

    view.set_at(torch.tensor([0]), rb_data(1, start=50))
    assert_tree_equal(rb.get_at(torch.tensor([0])), rb_data(1, start=50))
    view.set_flattened_at(torch.tensor([5]), rb.get_flattened_at(torch.tensor([0])))
    assert_tree_equal(rb.get_at(5), rb.get_at(0))


def test_torch_to_numpy_round_trip():
    rb = make_rb(6, capacity=10)
    view = ToBackendOrDeviceBatch(rb, backend=NB)
    assert view.backend is NB and view.is_mutable
    idx = np.array([0, 3, 5])
    out = view.get_at(idx)
    assert all(isinstance(v, np.ndarray) for v in out.values())
    assert_tree_equal(out, to_numpy_tree(rb.get_at(torch.tensor([0, 3, 5]))))
    assert view.single_space.contains(view.get_at(1))
    assert isinstance(view.get_flattened_at(idx), np.ndarray)

    new = {k: to_numpy(v) for k, v in rb_data(2, start=20).items()}
    view.set_at(np.array([1, 2]), new)
    assert_tree_equal(to_numpy_tree(rb.get_at(torch.tensor([1, 2]))), new)
    view.extend(new)
    assert len(rb) == len(view) == 8
    view.extend_flattened(view.get_flattened_at(np.array([0])))
    assert len(rb) == 9
    assert_tree_equal(view.get_at(8), view.get_at(0))


def to_numpy_tree(tree):
    return {k: to_numpy(v) for k, v in tree.items()}


def test_numpy_to_torch_with_device_and_metadata():
    # Converting backends *and* devices must move the converted (target) array.
    src = numpy_array_batch(6)
    view = ToBackendOrDeviceBatch(src, backend=TB, device="cpu")
    idx = torch.tensor([1, 4])
    data, meta = view.get_at_with_metadata(idx)
    _all_torch_on(data, "cpu")
    _all_torch_on(meta, "cpu")
    assert_tree_equal(data, src.get_at(np.array([1, 4])), check_dtype=False)
    assert_tree_equal(meta, {"step": np.array([10, 40])}, check_dtype=False)
    flat, meta = view.get_flattened_at_with_metadata(idx)
    assert isinstance(flat, torch.Tensor)
    assert_tree_equal(flat, src.get_flattened_at(np.array([1, 4])), check_dtype=False)
    view.remove_at(idx)
    assert len(src) == 4


def test_tuple_structure_is_preserved():
    space = TupleSpace(NB, (BoxSpace(NB, -1, 1, np.float32, shape=(2,)), BoxSpace(NB, 0, 9, np.int64, shape=())))
    src = ArrayBatch(space, (np.zeros((3, 2), np.float32), np.arange(3)))
    view = ToBackendOrDeviceBatch(src, backend=TB)
    out = view.get_at(torch.tensor([0, 2]))
    assert isinstance(out, tuple)
    assert_tree_equal(out[1], np.array([0, 2]), check_dtype=False)


def test_data_to_passes_non_array_values_through():
    assert data_to(None, NB, TB) is None
    assert data_to("text", NB, TB) == "text"
    out = data_to({"a": [np.zeros(1)], "b": 3}, NB, TB)
    assert isinstance(out["a"], list) and isinstance(out["a"][0], torch.Tensor) and out["b"] == 3


def test_flattened_access_is_flattened_get_at():
    src = numpy_array_batch(5)
    view = ToBackendOrDeviceBatch(src, backend=TB)
    idx = torch.tensor([0, 2])
    # Flattening promotes mixed dtypes differently per backend, so compare values only.
    assert_tree_equal(view.get_flattened_at(idx),
                      sfu.flatten_data(view._batched_space, view.get_at(idx), start_dim=1), check_dtype=False)


def test_torch_to_jax_with_device():
    # The device move must be done by the *target* backend after conversion.
    jax = pytest.importorskip("jax")
    from unienv_interface.backends.jax import JaxComputeBackend as JB
    rb = make_rb(4)
    device = jax.devices("cpu")[0]
    view = ToBackendOrDeviceBatch(rb, backend=JB, device=device)
    out = view.get_at(JB.asarray([0, 3]))
    assert all(isinstance(v, jax.Array) for v in out.values())
    assert_tree_equal(out, rb.get_at(torch.tensor([0, 3])), check_dtype=False)
