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
