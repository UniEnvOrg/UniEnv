from typing import Generic, TypeVar, Generic, Optional, Any, Dict as DictT, Tuple as TupleT, Sequence as SequenceT, Union as UnionT, List
from unienv_data import *
from unienv_data.storages.common import FlattenedStorage
from unienv_data.storages.pytorch import PytorchTensorStorage
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


def perform_torch_replay_buffer_with_space_test(
    space: Space[Any, BDeviceType, BDtypeType, BRNGType],
    capacity: int,
    use_mmap : bool = False,
    seed : Optional[int] = None
):
    tempdumpdir = tempfile.mkdtemp()
    rb = ReplayBuffer.create(
        FlattenedStorage,
        space,
        inner_storage_cls=PytorchTensorStorage,
        cache_path=tempdumpdir if use_mmap else None,
        capacity=capacity,
        is_memmap=use_mmap,
    )
    
    rng = torch.Generator(space.device)
    if seed is not None:
        rng.manual_seed(seed)
    
    perform_rb_fill_test(
        rb,
        space,
        capacity - 1,
        rng
    )
    assert len(rb) == capacity - 1
    perform_rb_fill_test(
        rb,
        space,
        1,
        rng
    )
    assert rb.offset == 0
    rb.clear()

    ref_idx, flat_ref_slice = perform_rb_fill_test(
        rb,
        space,
        capacity,
        rng
    )
    rb.dumps(tempdumpdir)
    new_rb = ReplayBuffer.load_from(
        tempdumpdir, 
        backend=space.backend,
        device=space.device,
        is_memmap=use_mmap
    )
    slice_space = sbu.batch_space(space, capacity)
    new_slice = new_rb.get_at(ref_idx)
    new_flat_slice = sfu.flatten_data(slice_space, new_slice)
    assert torch.allclose(flat_ref_slice, new_flat_slice)

    perform_rb_fill_test(
        rb,
        space,
        capacity * 3,
        rng
    )
    perform_rb_fill_test(
        rb,
        space,
        capacity // 2,
        rng
    )

    rb.clear()

    perform_rb_fill_test(
        rb,
        space,
        capacity*2,
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
    perform_torch_replay_buffer_with_space_test(
        space,
        capacity,
        use_mmap,
        seed
    )

