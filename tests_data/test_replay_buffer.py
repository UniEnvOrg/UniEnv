from typing import Generic, TypeVar, Generic, Optional, Any, Dict as DictT, Tuple as TupleT, Sequence as SequenceT, Union as UnionT, List
from unienv_data import *
from unienv_data.storages.flattened import FlattenedStorage
from unienv_data.storages.hdf5 import HDF5Storage
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_data.storages.parquet import ParquetStorage
from unienv_data.samplers import *
from unienv_data.batches import *
from unienv_interface.space import *
from unienv_interface.space.space_utils import flatten_utils as sfu
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
import torch
import numpy as np
import tempfile
import os
import pytest
from unienv_interface.space.space_utils import batch_utils as sbu

def perform_rb_fill_test(
    replay_buffer : ReplayBuffer[
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ],
    space : Space[Any, BDeviceType, BDtypeType, BRNGType],
    fill_length : int,
    rng : BRNGType,
) -> None:
    prev_len = len(replay_buffer)

    batched_space = sbu.batch_space(space, fill_length)
    rng, datas = batched_space.sample(rng)
    replay_buffer.extend(datas)

    assert len(replay_buffer) == fill_length + prev_len if replay_buffer.capacity is None else min(replay_buffer.capacity, fill_length + prev_len)

    check_size = fill_length if replay_buffer.capacity is None else min(fill_length, replay_buffer.capacity)
    index_check_mask = space.backend.zeros((fill_length,), dtype=space.backend.default_boolean_dtype, device=space.device)
    index_check_mask = space.backend.at(index_check_mask)[-check_size:].set(True)
    ref_slice = sbu.get_at(batched_space, datas, index_check_mask)
    for i in range(check_size):
        sampled = replay_buffer.get_at(-(check_size - i))
        sampled_p = replay_buffer.get_at(len(replay_buffer) - check_size + i)
        ref_data = sbu.get_at(batched_space, ref_slice, i)
        flat_sampled = sfu.flatten_data(space, sampled)
        flat_sampled_p = sfu.flatten_data(space, sampled_p)
        flat_data = sfu.flatten_data(space, ref_data)
        assert space.backend.all(flat_sampled == flat_data)
        assert space.backend.all(flat_sampled_p == flat_data)
    
    batched_check_space = sbu.batch_space(space, check_size)
    sampled_slice = replay_buffer.get_at(slice(-check_size, None))
    flat_sampled_slice = sfu.flatten_data(batched_check_space, sampled_slice)
    flat_ref_slice = sfu.flatten_data(batched_check_space, ref_slice)
    assert space.backend.all(flat_sampled_slice == flat_ref_slice)
    return slice(-check_size, None), flat_ref_slice

def check_fixed_capacity_replay_buffer(
    rb : ReplayBuffer[Any, BArrayType, BDeviceType, BDtypeType, BRNGType],
    seed : Optional[int] = None,
    load_kwargs : DictT[str, Any] = {},
):
    tempdumpdir = rb.cache_path if rb.cache_path is not None else tempfile.mkdtemp()
    rng = rb.backend.random.random_number_generator(seed, device=rb.device)
    perform_rb_fill_test(
        rb,
        rb.single_space,
        rb.capacity - 1,
        rng
    )
    assert len(rb) == rb.capacity - 1
    perform_rb_fill_test(
        rb,
        rb.single_space,
        1,
        rng
    )
    assert rb.offset == 0
    rb.clear()

    perform_rb_fill_test(
        rb,
        rb.single_space,
        rb.capacity - 2,
        rng
    )
    assert len(rb) == rb.capacity - 2
    perform_rb_fill_test(
        rb,
        rb.single_space,
        5,
        rng
    )
    assert rb.offset == 3

    ref_idx, flat_ref_slice = perform_rb_fill_test(
        rb,
        rb.single_space,
        rb.capacity,
        rng
    )
    rb.dumps(tempdumpdir)
    new_rb = ReplayBuffer.load_from(
        tempdumpdir,
        backend=rb.backend,
        device=rb.device,
        **load_kwargs
    )
    assert new_rb.capacity == rb.capacity
    assert new_rb.offset == rb.offset
    assert new_rb.count == rb.count
    assert new_rb.single_space == rb.single_space

    slice_space = sbu.batch_space(new_rb.single_space, new_rb.capacity)
    new_slice = new_rb.get_at(ref_idx)
    new_flat_slice = sfu.flatten_data(slice_space, new_slice)
    assert new_rb.backend.all(flat_ref_slice==new_flat_slice)

    perform_rb_fill_test(
        rb,
        rb.single_space,
        rb.capacity * 3,
        rng
    )
    perform_rb_fill_test(
        rb,
        rb.single_space,
        rb.capacity // 2,
        rng
    )

    rb.clear()

    perform_rb_fill_test(
        rb,
        rb.single_space,
        rb.capacity * 2,
        rng
    )

    return rb

@pytest.mark.parametrize("capacity", [10, 50])
@pytest.mark.parametrize("use_mmap", [False, True])
@pytest.mark.parametrize("seed", [0, 1024, 2048])
def test_torch_replay_buffer(
    capacity : int,
    use_mmap : bool,
    seed : int
):
    if not use_mmap:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    space = BoxSpace(
        PyTorchComputeBackend,
        0.0,
        100.0,
        torch.float32,
        device=device,
        shape=(3, 5, 2)
    )
    tempdumpdir = tempfile.mkdtemp()
    rb = ReplayBuffer.create(
        FlattenedStorage,
        space,
        inner_storage_cls=PytorchTensorStorage,
        cache_path=tempdumpdir if use_mmap else None,
        capacity=capacity,
        is_memmap=use_mmap,
    )
    check_fixed_capacity_replay_buffer(
        rb,
        seed=seed,
        load_kwargs=dict(
            is_memmap=use_mmap,
        )
    )

@pytest.mark.parametrize("capacity", [10, 50])
@pytest.mark.parametrize("seed", [0, 1024, 2048])
def test_hdf5_replay_buffer(
    capacity : int,
    seed : int
):
    space = BoxSpace(
        NumpyComputeBackend,
        0.0,
        100.0,
        np.float32,
        shape=(3, 5, 2)
    )
    tempdumpdir = tempfile.mkdtemp()
    rb = ReplayBuffer.create(
        HDF5Storage,
        space,
        cache_path=tempdumpdir,
        capacity=capacity,
    )
    check_fixed_capacity_replay_buffer(
        rb,
        seed=seed,
        load_kwargs={}
    )

@pytest.mark.parametrize("capacity", [10, 50])
@pytest.mark.parametrize("seed", [0, 1024, 2048])
def test_parquet_replay_buffer(
    capacity : int,
    seed : int
):
    space = BoxSpace(
        NumpyComputeBackend,
        0.0,
        100.0,
        np.float32,
        shape=(3, 5, 2)
    )
    tempdumpdir = tempfile.mkdtemp()
    rb = ReplayBuffer.create(
        ParquetStorage,
        space,
        cache_path=tempdumpdir,
        capacity=capacity,
    )
    check_fixed_capacity_replay_buffer(
        rb,
        seed=seed,
        load_kwargs={}
    )

@pytest.mark.parametrize("capacity", [10, 50])
@pytest.mark.parametrize("seed", [0, 1024])
def test_parquet_dict_replay_buffer(
    capacity : int,
    seed : int
):
    space = DictSpace(
        NumpyComputeBackend,
        {
            "obs": BoxSpace(NumpyComputeBackend, 0.0, 1.0, np.float32, shape=(4,)),
            "action": BoxSpace(NumpyComputeBackend, -1.0, 1.0, np.float32, shape=(2,)),
        },
    )
    tempdumpdir = tempfile.mkdtemp()
    rb = ReplayBuffer.create(
        ParquetStorage,
        space,
        cache_path=tempdumpdir,
        capacity=capacity,
    )
    check_fixed_capacity_replay_buffer(
        rb,
        seed=seed,
        load_kwargs={}
    )
