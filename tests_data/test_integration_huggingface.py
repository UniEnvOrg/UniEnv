"""HFAsUniEnvDataset over in-memory HuggingFace datasets."""

import numpy as np
import pytest

datasets = pytest.importorskip("datasets")

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_data.integrations.huggingface import HFAsUniEnvDataset

from tests_common.backends import cpu_backend_params
from tests_common.trees import assert_tree_equal, to_numpy

N = 8


def _hf_dataset():
    i = np.arange(N)
    return datasets.Dataset.from_dict({
        "obs": np.stack([i, -i], -1).astype(np.float32).tolist(),
        "reward": (i / 2).astype(np.float32).tolist(),
        "step": i.tolist(),
    })


def _expected(idx):
    i = np.arange(N)[idx]
    return {"obs": np.stack([i, -i], -1).astype(np.float32), "reward": i / 2, "step": i}


def _np(tree):
    return {k: to_numpy(v) for k, v in tree.items()}


@pytest.fixture
def wrapped(backend, device):
    return HFAsUniEnvDataset.create(_hf_dataset(), backend, device=device), backend


pytestmark = pytest.mark.parametrize("backend,device", cpu_backend_params())


def test_space_and_length(wrapped):
    ds, backend = wrapped
    assert len(ds) == N
    assert set(ds.single_space.keys()) == {"obs", "reward", "step"}
    assert ds.single_space["obs"].shape == (2,)
    assert ds.single_space["reward"].shape == ()
    assert ds.backend is backend and not ds.is_mutable


@pytest.mark.parametrize("kind", ["int", "negative-int", "slice", "ellipsis", "int-array", "negative-array", "bool-mask"])
def test_index_kinds(wrapped, kind):
    # Regression: numpy index arrays raised (dtype_is_boolean was given the array, not its dtype),
    # Ellipsis was rejected by HF, and numpy scalar columns broke space construction.
    ds, backend = wrapped
    picks = np.array([1, 6, 3])
    mask = np.zeros(N, dtype=bool)
    mask[[2, 5]] = True
    idx, ref = {
        "int": (4, 4),
        "negative-int": (-1, N - 1),
        "slice": (slice(1, 7, 2), slice(1, 7, 2)),
        "ellipsis": (..., slice(None)),
        "int-array": (backend.asarray(picks), picks),
        "negative-array": (backend.asarray(picks - N), picks),
        "bool-mask": (backend.asarray(mask), mask),
    }[kind]
    out = ds.get_at(idx)
    assert_tree_equal(_np(out), _expected(ref), check_dtype=False)
    space = ds.single_space if isinstance(idx, int) else sbu.batch_space(ds.single_space, len(_expected(ref)["step"]))
    assert space.contains(out)
    data, meta = ds.get_at_with_metadata(idx)
    assert meta == {}


def test_flattened_access(wrapped):
    ds, backend = wrapped
    idx = backend.asarray(np.array([0, 5]))
    flat = ds.get_flattened_at(idx)
    assert_tree_equal(flat, sfu.flatten_data(ds._batched_space, ds.get_at(idx), start_dim=1))
    assert_tree_equal(ds.get_flattened_at(2), sfu.flatten_data(ds.single_space, ds.get_at(2)))


def test_get_column(wrapped):
    # Regression: get_column returned rows like {"obs": ...} while advertising the column's space.
    ds, backend = wrapped
    col = ds.get_column(["obs"])
    assert col.single_space == ds.single_space["obs"]
    idx = backend.asarray(np.array([2, 7]))
    out = col.get_at(idx)
    assert col._batched_space.contains(out) or tuple(out.shape) == (2, 2)
    assert_tree_equal(to_numpy(out), _expected(np.array([2, 7]))["obs"])
    assert_tree_equal(to_numpy(col.get_at(3)), np.array([3, -3], dtype=np.float32))


def test_get_column_dotted_name(backend, device):
    if backend is not NB:
        pytest.skip("backend independent")
    hf = datasets.Dataset.from_dict({"observation.state": [[1.0], [2.0]], "action": [[0.0], [1.0]]})
    ds = HFAsUniEnvDataset.create(hf, NB)
    col = ds.get_column(["observation", "state"])
    assert_tree_equal(col.get_at(np.array([1])), np.array([[2.0]]), check_dtype=False)


def test_string_columns(backend, device):
    # Regression: with the numpy backend, HF's np.str_ values became 0-d "<U" arrays and
    # create() failed with "Box dtype must be a real floating or integer type".
    hf = datasets.Dataset.from_dict({"x": [0.0, 1.0, 2.0], "task": ["pick", "place", "push"]})
    ds = HFAsUniEnvDataset.create(hf, backend, device=device)
    one = ds.get_at(1)
    assert one["task"] == "place" and ds.single_space.contains(one)
    out = ds.get_at(backend.asarray(np.array([2, 0])))
    assert list(out["task"]) == ["push", "pick"]
    assert sbu.batch_space(ds.single_space, 2).contains(out)
    assert list(ds.get_at(slice(0, 2))["task"]) == ["pick", "place"]


def test_nested_struct_columns_are_batched_per_field(backend, device):
    # Regression: for index arrays / slices HF returns struct columns row by row (a list or
    # object array of dicts), which did not match the nested DictSpace of the batch.
    hf = datasets.Dataset.from_list([
        {"obs": {"pos": [float(i), -float(i)], "inner": {"vel": float(i)}}, "step": i} for i in range(N)
    ])
    ds = HFAsUniEnvDataset.create(hf, backend, device=device)
    picks = np.array([5, 1, 6])
    for idx, ref in ((backend.asarray(picks), picks), (slice(1, 7, 2), np.arange(N)[1:7:2]), (..., np.arange(N))):
        out = ds.get_at(idx)
        assert sbu.batch_space(ds.single_space, len(ref)).contains(out)
        np.testing.assert_array_equal(to_numpy(out["obs"]["pos"]), np.stack([ref, -ref], -1))
        np.testing.assert_array_equal(to_numpy(out["obs"]["inner"]["vel"]), ref)
        np.testing.assert_array_equal(to_numpy(out["step"]), ref)
    single = ds.get_at(3)
    assert ds.single_space.contains(single)
    np.testing.assert_array_equal(to_numpy(single["obs"]["pos"]), [3.0, -3.0])


def test_empty_index_gives_empty_batches_of_the_right_shape(wrapped):
    ds, backend = wrapped
    out = ds.get_at(backend.asarray(np.array([], dtype=np.int64)))
    assert tuple(out["obs"].shape) == (0, 2) and tuple(out["step"].shape) == (0,)
    assert sbu.batch_space(ds.single_space, 0).contains(out)
