"""Small in-memory batches used by the batch-view / sampler / integration tests.

* :func:`make_rb` builds a CPU ``ReplayBuffer`` (FlattenedStorage over an
  in-memory PytorchTensorStorage) filled with deterministic data and no ring
  wrap-around, so ``rb.get_at(i)`` is sample ``i``.
* :class:`ArrayBatch` is a minimal backend-generic ``BatchBase`` that keeps its
  data (and optional per-sample metadata) as batched arrays. It exists so the
  metadata code paths of the views can be exercised (``ReplayBuffer`` always
  returns ``None`` metadata).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from unienv_interface.space import BinarySpace, BoxSpace, DictSpace, Space
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_data.base import BatchBase

EPISODE_LEN = 4


def rb_space(backend=None, device="cpu") -> DictSpace:
    """``{"obs": float32 (2,), "episode_id": int64 (), "episode_end": bool ()}``."""
    if backend is None:
        from unienv_interface.backends.pytorch import PyTorchComputeBackend as backend
    import torch
    return DictSpace(backend, {
        "obs": BoxSpace(backend, -1e6, 1e6, torch.float32, shape=(2,), device=device),
        "episode_id": BoxSpace(backend, 0, 1_000_000, torch.int64, shape=(), device=device),
        "episode_end": BinarySpace(backend, shape=(), device=device),
    }, device=device)


def rb_data(n: int, start: int = 0) -> Dict[str, Any]:
    """Torch data for :func:`rb_space`: ``obs[i] = [2i, 2i+1]``, episodes of ``EPISODE_LEN`` steps."""
    import torch
    i = torch.arange(start, start + n)
    return {
        "obs": torch.stack([2 * i, 2 * i + 1], dim=-1).to(torch.float32),
        "episode_id": i // EPISODE_LEN,
        "episode_end": (i % EPISODE_LEN) == EPISODE_LEN - 1,
    }


def make_rb(n: int = 10, capacity: Optional[int] = None, start: int = 0):
    """CPU ``ReplayBuffer`` holding ``rb_data(n, start)`` (``capacity`` defaults to ``n``)."""
    from unienv_data import ReplayBuffer
    from unienv_data.storages.flattened import FlattenedStorage
    from unienv_data.storages.pytorch import PytorchTensorStorage

    rb = ReplayBuffer.create(
        FlattenedStorage,
        rb_space(),
        inner_storage_cls=PytorchTensorStorage,
        capacity=capacity or max(n, 1),
        is_memmap=False,
    )
    if n > 0:
        rb.extend(rb_data(n, start))
    return rb


class ArrayBatch(BatchBase):
    """Mutable in-memory batch over batched arrays, with optional metadata."""

    def __init__(
        self,
        single_space: Space,
        data: Any,
        single_metadata_space: Optional[DictSpace] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(single_space, single_metadata_space)
        self.data = data
        self.metadata = metadata
        self.closed = 0

    def __len__(self) -> int:
        return int(sbu.batch_size_data(self.data))

    def _space(self, space: Space) -> Space:
        return sbu.batch_space(space, len(self))

    def _index(self, idx):
        if idx is Ellipsis:
            return slice(None)
        if isinstance(idx, int):
            assert -len(self) <= idx < len(self), f"index {idx} out of range"
        elif self.backend.is_backendarray(idx):
            assert len(idx.shape) == 1, "index arrays must be 1-D"
        return idx

    def get_at_with_metadata(self, idx):
        idx = self._index(idx)
        data = sbu.get_at(self._space(self.single_space), self.data, idx)
        if self.metadata is None:
            return data, None
        return data, sbu.get_at(self._space(self.single_metadata_space), self.metadata, idx)

    def get_at(self, idx):
        return self.get_at_with_metadata(idx)[0]

    def set_at(self, idx, value):
        self.data = sbu.set_at(self._space(self.single_space), self.data, self._index(idx), value)

    def remove_at(self, idx):
        keep = np.ones(len(self), dtype=bool)
        keep[self.backend.to_numpy(idx) if self.backend.is_backendarray(idx) else idx] = False
        keep = self.backend.from_numpy(keep)
        self.data = sbu.get_at(self._space(self.single_space), self.data, keep)
        if self.metadata is not None:
            self.metadata = sbu.get_at(self._space(self.single_metadata_space), self.metadata, keep)

    def extend(self, value):
        n = sbu.batch_size_data(value)
        self.data = sbu.concatenate(self._space(self.single_space), [self.data, value])
        if self.metadata is not None:
            extra = sbu.batch_space(self.single_metadata_space, n).create_empty()
            self.metadata = sbu.concatenate(self._space(self.single_metadata_space), [self.metadata, extra])

    def close(self) -> None:
        self.closed += 1


def numpy_array_batch(n: int = 10, with_metadata: bool = True, start: int = 0) -> ArrayBatch:
    """Numpy :class:`ArrayBatch` mirroring :func:`rb_data`, with metadata ``{"step": int64 ()}``."""
    from unienv_interface.backends.numpy import NumpyComputeBackend as NB
    space = DictSpace(NB, {
        "obs": BoxSpace(NB, -1e6, 1e6, np.float32, shape=(2,)),
        "episode_id": BoxSpace(NB, 0, 1_000_000, np.int64, shape=()),
        "episode_end": BinarySpace(NB, shape=()),
    })
    i = np.arange(start, start + n)
    data = {
        "obs": np.stack([2 * i, 2 * i + 1], axis=-1).astype(np.float32),
        "episode_id": (i // EPISODE_LEN).astype(np.int64),
        "episode_end": (i % EPISODE_LEN) == EPISODE_LEN - 1,
    }
    if not with_metadata:
        return ArrayBatch(space, data)
    meta_space = DictSpace(NB, {"step": BoxSpace(NB, 0, 1_000_000, np.int64, shape=())})
    return ArrayBatch(space, data, meta_space, {"step": (i * 10).astype(np.int64)})
