"""The shared IO thread pool of file-backed storages, and list storages (one file per sample) using it."""

import multiprocessing as mp
import threading

import numpy as np
import pytest
import torch

from unienv_data.storages import _io_pool
from unienv_data.storages.image_storage import ImageStorage
from unienv_data.storages.npz_storage import NPZStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace


def _space(backend=NumpyComputeBackend):
    if backend is PyTorchComputeBackend:
        return BoxSpace(backend, 0, 255, torch.uint8, shape=(6, 5, 3), device="cpu")
    return BoxSpace(backend, 0, 255, np.uint8, shape=(6, 5, 3))


def _images(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, 6, 5, 3), dtype=np.uint8)


def _create(kind, path, capacity, backend=NumpyComputeBackend):
    if kind == "png":
        return ImageStorage.create(_space(backend), capacity=capacity, cache_path=path, format="PNG")
    return NPZStorage.create(_space(backend), capacity=capacity, cache_path=path)


@pytest.mark.parametrize("kind", ["png", "npz"])
def test_batched_reads_and_writes_keep_order_and_last_write_wins(tmp_path, kind):
    storage = _create(kind, str(tmp_path / kind), capacity=32)
    images = _images(32)
    storage.set(slice(None), images)
    index = np.array([31, 0, 5, 5, 17, 2, 30, 9, 9, 1])
    np.testing.assert_array_equal(storage.get(index), images[index])
    np.testing.assert_array_equal(storage.get(np.arange(32) % 3 == 0), images[::3])

    # duplicated targets: the value at the last position wins, as with a serial loop
    update = _images(6, seed=1)
    storage.set(np.array([4, 7, 4, 8, 7, 4]), update)
    expected = images.copy()
    expected[4], expected[7], expected[8] = update[5], update[4], update[3]
    np.testing.assert_array_equal(storage.get(slice(None)), expected)
    storage.close()


def test_torch_index_arrays(tmp_path):
    storage = _create("npz", str(tmp_path / "t"), capacity=12, backend=PyTorchComputeBackend)
    images = torch.from_numpy(_images(12))
    storage.set(torch.arange(12), images)
    index = torch.tensor([11, 3, 3, 0, 7])
    torch.testing.assert_close(storage.get(index), images[index])
    storage.close()


def test_map_io_runs_on_the_pool_and_serially_inside_it():
    names = _io_pool.map_io(lambda _: threading.current_thread().name, range(16))
    assert all(name.startswith("unienv-storage-io") for name in names)
    assert _io_pool.map_io(lambda _: threading.current_thread().name, range(2)) == \
        [threading.current_thread().name] * 2
    # A pool worker that maps again must not wait on the (bounded) pool it occupies
    nested = _io_pool.map_io(
        lambda i: _io_pool.map_io(lambda j: (i, j, threading.current_thread().name), range(8)),
        range(_io_pool.IO_POOL_WORKERS * 2),
    )
    for i, results in enumerate(nested):
        assert [(a, b) for a, b, _ in results] == [(i, j) for j in range(8)]
        assert len({name for _, _, name in results}) == 1


def _child_reads(path, queue):
    storage = NPZStorage.load_from(path, _space(), capacity=16)
    queue.put(storage.get(np.arange(16)).sum())


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="needs fork")
def test_pool_is_recreated_after_fork(tmp_path):
    path = str(tmp_path / "f")
    storage = _create("npz", path, capacity=16)
    images = _images(16)
    storage.set(slice(None), images)
    storage.dumps(path)
    storage.get(np.arange(16))  # starts the parent's pool threads
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=_child_reads, args=(path, queue))
    child.start()
    assert queue.get(timeout=60) == images.sum()
    child.join(timeout=60)
    assert child.exitcode == 0
