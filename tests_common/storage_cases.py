"""Table of SpaceStorage backends for the storage contract suite.

Every backend is described by a :class:`StorageCase`. Legitimate per-backend
differences are encoded as explicit capability flags instead of ad-hoc
``if`` statements inside the tests.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BinarySpace, BoxSpace, DictSpace, Space
from unienv_interface.space.space_utils import batch_utils as sbu

from tests_common.storage_video import LOSSLESS_RGB_KWARGS

NB = NumpyComputeBackend


@dataclasses.dataclass
class StorageCase:
    id: str
    make_space: Callable[[], Space]
    storage_cls: Callable[[], type]
    # Extra ``create`` kwargs; a zero-argument callable is evaluated lazily (keeps optional imports out of collection).
    create_kwargs: Any = dataclasses.field(default_factory=dict)
    load_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Capacities to exercise (None = unbounded storage grown with extend_length).
    capacities: Tuple[Optional[int], ...] = (None, 5)
    # Extra create kwargs when the storage is unbounded.
    unbounded_create_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # create() takes a cache path (False only for purely in-memory storages).
    uses_cache_path: bool = True
    # dumps() can only target the storage's own cache path (NPZ / Image / Video / Episode).
    dumps_only_to_own_path: bool = False
    # get/set accept negative integers (List / Episode storages reject them).
    negative_indices: bool = True
    # set() accepts arbitrary (non-contiguous) index arrays and masks. Episode storages only
    # accept contiguous runs (or one run wrapping around a fixed capacity).
    scattered_set: bool = True
    # get() accepts slices with a negative step (torch tensors and h5py datasets do not).
    negative_step_slices: bool = True
    # Element-wise comparison tolerance for lossy encodings.
    atol: float = 0.0
    # Whether get() preserves the space dtype exactly.
    check_dtype: bool = True
    # create(..., multiprocessing=True) is supported for a mutable storage.
    multiprocessing_create: bool = True
    # load_from(..., read_only=True, multiprocessing=True) yields a multiprocessing-safe storage.
    multiprocessing_load: bool = True
    # get_segments() reports (start, end) segments instead of None.
    reports_segments: bool = False
    # Appends become visible only after mark_segment_end (and abort discards them).
    buffered_segments: bool = False
    marks: Tuple[Any, ...] = ()

    def create(self, tmp_path, capacity, *, name="storage", multiprocessing=False, **extra):
        storage_cls = self.storage_cls()
        kwargs = dict(self.create_kwargs() if callable(self.create_kwargs) else self.create_kwargs)
        if capacity is None:
            kwargs.update(self.unbounded_create_kwargs)
        kwargs.update(extra)
        cache_path = self.cache_path_for(tmp_path, name) if self.uses_cache_path else None
        return storage_cls.create(
            self.make_space(),
            capacity=capacity,
            cache_path=cache_path,
            multiprocessing=multiprocessing,
            **kwargs,
        )

    def cache_path_for(self, tmp_path, name="storage") -> str:
        ext = self.storage_cls().single_file_ext or ""
        return str(tmp_path / f"{name}{ext}")

    def load(self, path, capacity, **extra):
        kwargs = dict(self.load_kwargs)
        kwargs.update(extra)
        return self.storage_cls().load_from(path, self.make_space(), capacity=capacity, **kwargs)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def random_batch(space: Space, n: int, seed: int = 0) -> Any:
    """Deterministic batch of ``n`` samples of ``space`` (values inside the space bounds)."""
    rng = np.random.default_rng(seed)
    return _random_leaf(space, n, rng)


def _random_leaf(space: Space, n: int, rng: np.random.Generator) -> Any:
    if isinstance(space, DictSpace):
        return {key: _random_leaf(sub, n, rng) for key, sub in space.spaces.items()}
    backend = space.backend
    shape = (n, *space.shape)
    if isinstance(space, BinarySpace):
        arr = rng.integers(0, 2, size=shape).astype(bool)
    else:
        np_dtype = np.dtype(backend.to_numpy(backend.zeros((1,), dtype=space.dtype)).dtype)
        if np.issubdtype(np_dtype, np.integer):
            low = max(int(np.iinfo(np_dtype).min), -1000)
            high = min(int(np.iinfo(np_dtype).max), 1000)
            if np_dtype == np.uint8:
                low, high = 0, 255
            arr = rng.integers(low, high + 1, size=shape).astype(np_dtype)
        else:
            arr = rng.uniform(-5.0, 5.0, size=shape).astype(np_dtype)
    return backend.from_numpy(arr, dtype=space.dtype, device=space.device)


def get_rows(space: Space, batch: Any, rows) -> Any:
    """``batch[rows]`` for a batch of ``space`` samples (rows: int, slice or int array)."""
    batched_space = sbu.batch_space(space, sbu.batch_size_data(batch))
    if isinstance(rows, np.ndarray):
        rows = space.backend.from_numpy(rows.astype(np.int64), device=space.device) \
            if space.backend is not NB else rows
    return sbu.get_at(batched_space, batch, rows)


def to_backend_index(space: Space, index):
    """Convert a numpy index array to ``space``'s backend; leave ints/slices alone."""
    if isinstance(index, np.ndarray) and space.backend is not NB:
        return space.backend.from_numpy(index, device=space.device)
    return index


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------

def _np_box(shape=(3,), dtype=np.float32, low=-10.0, high=10.0) -> BoxSpace:
    return BoxSpace(NB, low, high, dtype, shape=shape)


def _np_nested_dict() -> DictSpace:
    return DictSpace(NB, {
        "obs": DictSpace(NB, {
            "pos": _np_box((2,)),
            "count": BoxSpace(NB, -1000, 1000, np.int64, shape=()),
        }),
        "reward": _np_box(()),
        "done": BinarySpace(NB, shape=(), dtype=np.bool_),
    })


def _np_flat_dict() -> DictSpace:
    return DictSpace(NB, {
        "obs": _np_box((2, 2)),
        "step": BoxSpace(NB, -1000, 1000, np.int64, shape=()),
        "done": BinarySpace(NB, shape=(), dtype=np.bool_),
    })


def _np_image() -> BoxSpace:
    return BoxSpace(NB, 0, 255, np.uint8, shape=(6, 8, 3))


def _torch_box() -> BoxSpace:
    import torch
    from unienv_interface.backends.pytorch import PyTorchComputeBackend
    return BoxSpace(PyTorchComputeBackend, -10.0, 10.0, torch.float32, shape=(2, 3), device="cpu")


def _torch_dict() -> DictSpace:
    import torch
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    return DictSpace(TB, {
        "a": BoxSpace(TB, -10.0, 10.0, torch.float32, shape=(2,), device="cpu"),
        "b": BoxSpace(TB, -1000, 1000, torch.int64, shape=(), device="cpu"),
    }, device="cpu")


# ---------------------------------------------------------------------------
# Lazy storage class getters (keep optional dependencies import-local)
# ---------------------------------------------------------------------------

def _cls(module: str, name: str) -> Callable[[], type]:
    def getter() -> type:
        import importlib
        return getattr(importlib.import_module(module), name)
    return getter


def _hdf5():
    return _cls("unienv_data.storages.hdf5", "HDF5Storage")()


def _npz():
    return _cls("unienv_data.storages.npz_storage", "NPZStorage")()


def _pytorch():
    return _cls("unienv_data.storages.pytorch", "PytorchTensorStorage")()


def _transformed_kwargs() -> Dict[str, Any]:
    from unienv_interface.transformations import RescaleTransformation
    return {
        "inner_storage_cls": _hdf5(),
        "data_transformation": RescaleTransformation(new_low=-1.0, new_high=1.0, new_dtype=np.float64),
    }


STORAGE_CASES: List[StorageCase] = [
    StorageCase(
        id="pytorch-memmap",
        negative_step_slices=False,
        make_space=_torch_box,
        storage_cls=_cls("unienv_data.storages.pytorch", "PytorchTensorStorage"),
        create_kwargs={"is_memmap": True},
        load_kwargs={"is_memmap": True},
        capacities=(5,),
    ),
    StorageCase(
        id="pytorch-ram",
        negative_step_slices=False,
        make_space=_torch_box,
        storage_cls=_cls("unienv_data.storages.pytorch", "PytorchTensorStorage"),
        create_kwargs={"is_memmap": False},
        load_kwargs={"is_memmap": False},
        capacities=(5,),
        uses_cache_path=False,
    ),
    StorageCase(
        id="flattened-pytorch",
        negative_step_slices=False,
        make_space=_torch_dict,
        storage_cls=_cls("unienv_data.storages.flattened", "FlattenedStorage"),
        create_kwargs=lambda: {"inner_storage_cls": _pytorch(), "is_memmap": True},
        load_kwargs={"is_memmap": True},
        capacities=(5,),
        # FlattenedStorage stores everything in one tensor of the promoted dtype.
        check_dtype=False,
    ),
    StorageCase(
        id="npz",
        make_space=lambda: _np_box((2, 3)),
        storage_cls=_cls("unienv_data.storages.npz_storage", "NPZStorage"),
        create_kwargs={"compressed": True},
        dumps_only_to_own_path=True,
        negative_indices=False,
    ),
    StorageCase(
        id="hdf5-box",
        negative_step_slices=False,
        make_space=lambda: _np_box((2, 3)),
        storage_cls=_cls("unienv_data.storages.hdf5", "HDF5Storage"),
        multiprocessing_create=False,
    ),
    StorageCase(
        id="hdf5-nested-dict",
        negative_step_slices=False,
        make_space=_np_nested_dict,
        storage_cls=_cls("unienv_data.storages.hdf5", "HDF5Storage"),
        create_kwargs={"compression": "gzip", "chunks": True},
        multiprocessing_create=False,
    ),
    StorageCase(
        id="parquet-flat-dict",
        make_space=_np_flat_dict,
        storage_cls=_cls("unienv_data.storages.parquet", "ParquetStorage"),
        create_kwargs={"piece_size": 2},
        multiprocessing_create=False,
    ),
    StorageCase(
        id="parquet-box",
        make_space=lambda: _np_box((3,)),
        storage_cls=_cls("unienv_data.storages.parquet", "ParquetStorage"),
        create_kwargs={"piece_size": 3, "compression": None},
        multiprocessing_create=False,
    ),
    StorageCase(
        id="image-png",
        make_space=_np_image,
        storage_cls=_cls("unienv_data.storages.image_storage", "ImageStorage"),
        create_kwargs={"format": "PNG"},
        dumps_only_to_own_path=True,
        negative_indices=False,
    ),
    StorageCase(
        id="video-ffv1",
        make_space=_np_image,
        storage_cls=_cls("unienv_data.storages.video_storage", "VideoStorage"),
        create_kwargs=dict(LOSSLESS_RGB_KWARGS),
        load_kwargs={"hardware_acceleration": None, "decode_backend": "pyav", "codec": "ffv1"},
        dumps_only_to_own_path=True,
        negative_indices=False,
        scattered_set=False,
        multiprocessing_create=False,
        reports_segments=True,
        buffered_segments=True,
    ),
    StorageCase(
        id="dict-npz-hdf5",
        negative_step_slices=False,
        make_space=_np_nested_dict,
        storage_cls=_cls("unienv_data.storages.dict_storage", "DictStorage"),
        create_kwargs=lambda: {"storage_cls_map": {"obs/pos": _npz(), "*": _hdf5()}},
        dumps_only_to_own_path=True,  # the NPZ child can only dump in place
        negative_indices=False,  # the NPZ child rejects negative indices
        multiprocessing_create=False,  # the HDF5 child cannot be created in multiprocessing mode
    ),
    StorageCase(
        id="dict-hdf5",
        negative_step_slices=False,
        make_space=_np_flat_dict,
        storage_cls=_cls("unienv_data.storages.dict_storage", "DictStorage"),
        create_kwargs=lambda: {"storage_cls_map": {"obs": _hdf5(), "*": _hdf5()}},
        multiprocessing_create=False,
    ),
    StorageCase(
        id="transformed-rescale-hdf5",
        negative_step_slices=False,
        make_space=lambda: _np_box((2, 3)),
        storage_cls=_cls("unienv_data.storages.transformation", "TransformedStorage"),
        create_kwargs=_transformed_kwargs,
        atol=1e-5,
        multiprocessing_create=False,
    ),
    StorageCase(
        id="to-numpy-hdf5",
        negative_step_slices=False,
        make_space=_torch_box,
        storage_cls=_cls("unienv_data.storages.backend_compat", "ToBackendOrDeviceStorage"),
        create_kwargs=lambda: {"inner_storage_cls": _hdf5(), "backend": NB},
        multiprocessing_create=False,
    ),
]


def storage_case_params(*, capacities: bool = True) -> list:
    """``pytest.param`` list over ``(case, capacity)`` (or just ``case``)."""
    params = []
    for case in STORAGE_CASES:
        if capacities:
            for capacity in case.capacities:
                cap_id = "unbounded" if capacity is None else f"cap{capacity}"
                params.append(pytest.param(case, capacity, id=f"{case.id}-{cap_id}", marks=list(case.marks)))
        else:
            params.append(pytest.param(case, id=case.id, marks=list(case.marks)))
    return params
