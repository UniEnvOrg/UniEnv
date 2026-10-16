"""Round-robin (offset != 0) consistency of every ReplayBuffer access path."""

import numpy as np
import pytest
import torch

from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.flattened import FlattenedStorage
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace


def _wrapped_flattened_rb(capacity: int = 4) -> ReplayBuffer:
    """Capacity-4 buffer holding rows 2..5 of ``arange(12).reshape(6, 2)`` with offset 2."""
    space = BoxSpace(PyTorchComputeBackend, -100.0, 100.0, torch.float32, shape=(2,), device="cpu")
    rb = ReplayBuffer.create(
        FlattenedStorage,
        space,
        inner_storage_cls=PytorchTensorStorage,
        capacity=capacity,
        is_memmap=False,
    )
    rb.extend(torch.arange(6, dtype=torch.float32).reshape(3, 2))
    rb.extend(torch.arange(6, 12, dtype=torch.float32).reshape(3, 2))
    assert rb.offset == 2 and rb.count == 4
    return rb


def test_get_flattened_at_respects_offset_after_wraparound():
    rb = _wrapped_flattened_rb()
    expected = torch.arange(4, 12, dtype=torch.float32).reshape(4, 2)
    for i in range(-4, 4):
        torch.testing.assert_close(rb.get_flattened_at(i), expected[i])
        torch.testing.assert_close(rb.get_flattened_at(i), rb.get_at(i))
        data, metadata = rb.get_flattened_at_with_metadata(i)
        torch.testing.assert_close(data, expected[i])
        assert metadata is None
    torch.testing.assert_close(rb.get_flattened_at(slice(None)), expected)
    torch.testing.assert_close(rb.get_flattened_at(Ellipsis), expected)
    torch.testing.assert_close(rb.get_flattened_at(torch.tensor([3, 0, -1])), expected[[3, 0, 3]])
