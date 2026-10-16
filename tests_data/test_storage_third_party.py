"""Vendored third-party helpers used by unienv_data (tensordict memmap, LeRobot metadata)."""

import json
import os
import pickle

import numpy as np
import pytest
import torch

from unienv_data.third_party.tensordict.memmap_tensor import MemoryMappedTensor


# ---------------------------------------------------------------------------
# MemoryMappedTensor (backs PytorchTensorStorage)
# ---------------------------------------------------------------------------

@pytest.fixture
def memmap_file(tmp_path):
    path = str(tmp_path / "data.memmap")
    MemoryMappedTensor.from_tensor(torch.arange(12.0).reshape(6, 2), filename=path)
    return path


def _read(path):
    return MemoryMappedTensor.from_filename(path, dtype=torch.float32, shape=(6, 2), readonly=True).clone()


@pytest.mark.parametrize("index", [slice(1, 3), 4, (slice(None), 1)], ids=["slice", "int", "column"])
def test_pickled_view_keeps_its_index(memmap_file, index):
    full = MemoryMappedTensor.from_filename(memmap_file, dtype=torch.float32, shape=(6, 2))
    view = full[index]
    assert isinstance(view, MemoryMappedTensor)
    clone = pickle.loads(pickle.dumps(view))
    assert clone.shape == view.shape
    torch.testing.assert_close(torch.as_tensor(clone), torch.as_tensor(view))


def test_pickled_read_only_tensor_stays_read_only(memmap_file):
    read_only = MemoryMappedTensor.from_filename(memmap_file, dtype=torch.float32, shape=(6, 2), readonly=True)
    clone = pickle.loads(pickle.dumps(read_only))
    clone[0] = 100.0  # private copy-on-write mapping
    torch.testing.assert_close(_read(memmap_file)[0], torch.tensor([0.0, 1.0]))


def test_pickled_writable_tensor_shares_the_file(memmap_file):
    writable = MemoryMappedTensor.from_filename(memmap_file, dtype=torch.float32, shape=(6, 2))
    clone = pickle.loads(pickle.dumps(writable))
    clone[5] = -1.0
    torch.testing.assert_close(_read(memmap_file)[5], torch.tensor([-1.0, -1.0]))
    state = writable.__getstate__()
    assert state["filename"] == os.path.abspath(memmap_file)
