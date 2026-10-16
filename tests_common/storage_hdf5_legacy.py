"""Reproduces the on-disk layout that HDF5Storage wrote before it chose dataset layouts itself."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import h5py
import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BinarySpace, BoxSpace, DictSpace, TextSpace


def write_legacy_layout(
    root: h5py.Group,
    space,
    capacity: Optional[int],
    compression: Any = None,
    chunks: Any = None,
    initial_capacity: Optional[int] = None,
) -> None:
    """
    Create the datasets for `space` exactly as `HDF5Storage._construct_hdf5_file` did before it picked dataset
    layouts itself: every dataset gets `maxshape=(capacity, ...)`, hence a chunked layout with h5py's own
    chunk guess whenever `chunks` is None.
    """
    if not isinstance(space, DictSpace):
        space = DictSpace(space.backend, {"data": space}, device=space.device)
    initial_capacity = capacity if initial_capacity is None else initial_capacity
    for key, sub_space in space.spaces.items():
        if isinstance(sub_space, DictSpace):
            write_legacy_layout(
                root.create_group(key), sub_space, capacity,
                compression=compression.get(key) if isinstance(compression, Mapping) else compression,
                chunks=chunks.get(key, chunks.get("*")) if isinstance(chunks, Mapping) else chunks,
                initial_capacity=initial_capacity,
            )
            continue
        if isinstance(sub_space, (BoxSpace, BinarySpace)):
            shape = (initial_capacity, *sub_space.shape)
            maxshape = (capacity, *sub_space.shape)
            dtype = sub_space.dtype or NumpyComputeBackend.default_boolean_dtype
        elif isinstance(sub_space, TextSpace):
            shape = (initial_capacity,)
            maxshape = (capacity,)
            dtype = h5py.string_dtype(encoding="utf-8", length=sub_space.max_length)
        else:
            raise ValueError(f"Unsupported space type: {type(sub_space)}")
        current_chunks = chunks.get(key, chunks.get("*")) if isinstance(chunks, Mapping) else chunks
        if current_chunks is not None and isinstance(current_chunks, Sequence) and len(current_chunks) < len(shape):
            current_chunks = tuple(current_chunks) + tuple(shape[len(current_chunks):])
        root.create_dataset(
            key, shape=shape, maxshape=maxshape, dtype=dtype,
            compression=compression.get(key) if isinstance(compression, Mapping) else compression,
            chunks=current_chunks,
        )


def legacy_layout_is_loadable(root: h5py.Group, space, capacity: Optional[int]) -> None:
    """The layout checks the previous `HDF5Storage.load_from` ran on a file (raises AssertionError otherwise)."""
    if not isinstance(space, DictSpace) and isinstance(root, h5py.Group):
        root = root["data"]
    if isinstance(space, DictSpace):
        for key, sub_space in space.spaces.items():
            legacy_layout_is_loadable(root[key], sub_space, capacity)
        return
    assert isinstance(root, h5py.Dataset)
    assert root.maxshape[0] == capacity
    if isinstance(space, (BoxSpace, BinarySpace)):
        assert root.shape[1:] == space.shape
        assert root.dtype == np.dtype(space.dtype or NumpyComputeBackend.default_boolean_dtype)
    else:
        assert root.dtype.kind in ("O", "S")
