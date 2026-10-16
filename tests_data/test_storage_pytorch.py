"""PytorchTensorStorage specifics."""

import numpy as np
import pytest
import torch

from unienv_data.base import SpaceStorage
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BinarySpace, BoxSpace


def _space(kind):
    if kind == "bool":
        return BinarySpace(PyTorchComputeBackend, shape=(3,), dtype=torch.bool, device="cpu")
    if kind == "image":
        return BoxSpace(PyTorchComputeBackend, 0, 255, torch.uint8, shape=(3, 4, 5), device="cpu")
    return BoxSpace(PyTorchComputeBackend, -10.0, 10.0, torch.float32, shape=(), device="cpu")


@pytest.mark.parametrize("kind", ["bool", "image", "scalar"])
@pytest.mark.parametrize("is_memmap", [False, True])
def test_append_matches_generic_append_with_wrap_around(tmp_path, kind, is_memmap):
    """The direct row write of `append` leaves the same data and cursor as the generic batched `set` path."""
    space = _space(kind)
    storages = [
        PytorchTensorStorage.create(space, capacity=5, is_memmap=is_memmap, cache_path=str(tmp_path / f"{i}.memmap"))
        for i in range(2)
    ]
    generator = torch.Generator().manual_seed(0)
    samples = [
        (torch.randint(0, 2, space.shape, generator=generator) * (1 if kind == "bool" else 7)).to(space.dtype)
        for _ in range(12)
    ]
    for storage in storages:
        storage.mark_segment_start(3)
    for sample in samples:
        storages[0].append(sample)
        SpaceStorage.append(storages[1], sample)
        assert storages[0]._active_segment_index == storages[1]._active_segment_index
    torch.testing.assert_close(storages[0].get(slice(None)), storages[1].get(slice(None)))
    # rows 3, 4, 0, 1, 2, 3, ... hold samples 0, 1, 2, ...: the last write to each row wins
    expected = torch.stack([samples[i] for i in (7, 8, 9, 10, 11)])
    rows = torch.tensor([(3 + i) % 5 for i in (7, 8, 9, 10, 11)])
    torch.testing.assert_close(storages[0].get(rows), expected)
    for storage in storages:
        storage.mark_segment_end()


def test_append_requires_an_open_segment():
    storage = PytorchTensorStorage.create(_space("scalar"), capacity=2, is_memmap=False)
    with pytest.raises(RuntimeError, match="without an open segment"):
        storage.append(torch.tensor(1.0))
