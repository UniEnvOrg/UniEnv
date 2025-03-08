from typing import Generic, TypeVar, Generic, Optional, Any, Dict as DictT, Tuple as TupleT, Sequence as SequenceT, Union as UnionT, List
from unienv_data import *
from unienv_data.storages.common import ListStorage
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_data.samplers import *
from unienv_data.batches import *
from unienv_interface.space import *
from unienv_interface.space import flatten_utils as sfu, batch_utils as sbu
from unienv_interface.backends import NumpyComputeBackend, ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.pytorch import PyTorchComputeBackend
import torch
import numpy as np
import tempfile
import os
import pytest

def perform_rb_fill_test(
    replay_buffer : ReplayBuffer[
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ],
    space : Space[Any, Any, BDeviceType, BDtypeType, BRNGType],
    fill_length : int,
    rng : BRNGType,
) -> None:
    prev_len = len(replay_buffer)

    batched_space = sbu.batch_space(space, fill_length)
    rng, datas = batched_space.sample(rng)
    replay_buffer.extend(datas)

    assert len(replay_buffer) == fill_length + prev_len if replay_buffer.capacity is None else min(replay_buffer.capacity, fill_length + prev_len)

    check_size = fill_length if replay_buffer.capacity is None else min(fill_length, replay_buffer.capacity)
    index_check_mask = space.backend.array_api_namespace.zeros((fill_length,), dtype=space.backend.default_boolean_dtype, device=space.device)
    index_check_mask[-check_size:] = True
    ref_slice = sbu.read_batched_data_with_mask(batched_space, datas, index_check_mask)
    for i in range(check_size):
        sampled = replay_buffer.get_at(-(check_size - i))
        sampled_p = replay_buffer.get_at(len(replay_buffer) - check_size + i)
        ref_data = sbu.get_at(batched_space, ref_slice, i)
        flat_sampled = sfu.flatten_data(space, sampled)
        flat_sampled_p = sfu.flatten_data(space, sampled_p)
        flat_data = sfu.flatten_data(space, ref_data)
        assert space.backend.array_api_namespace.all(flat_sampled == flat_data)
        assert space.backend.array_api_namespace.all(flat_sampled_p == flat_data)
    
    sampled_slice = replay_buffer.get_at(slice(-check_size, None))
    flat_sampled_slice = sfu.flatten_data(batched_space, sampled_slice)
    flat_ref_slice = sfu.flatten_data(batched_space, ref_slice)
    assert space.backend.array_api_namespace.all(flat_sampled_slice == flat_ref_slice)
    return slice(-check_size, None), flat_ref_slice


def perform_torch_replay_buffer_with_space_test(
    space: Space[Any, Any, BDeviceType, BDtypeType, BRNGType],
    capacity: int,
    use_mmap : bool = False,
    seed : Optional[int] = None
):
    flattened_space = sfu.flatten_space(space)
    single_instance_shape = flattened_space.shape
    tempdumpdir = tempfile.mkdtemp()
    rb = ReplayBuffer.create(
        space,
        PytorchTensorStorage,
        memmap=use_mmap,
        memmap_path=None if not use_mmap else os.path.join(tempdumpdir, "storage.data"),
        capacity=capacity
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
    new_rb = ReplayBuffer.load_from(tempdumpdir, memmap=use_mmap)
    slice_space = sbu.batch_space(space, capacity)
    new_slice = new_rb.get_at(ref_idx)
    new_flat_slice = sfu.flatten_data(slice_space, new_slice)
    assert torch.allclose(rb.storage.data, new_rb.storage.data)
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

def perform_list_replay_buffer_with_space_test(
    space: Space[Any, Any, BDeviceType, BDtypeType, BRNGType],
    capacity: Optional[int],
    seed : Optional[int] = None
):
    rb = ReplayBuffer.create(
        space,
        ListStorage,
        capacity=capacity,
    )
    
    rng = np.random.default_rng(seed)
    
    if capacity is None:
        perform_rb_fill_test(rb, space, 10, rng)
        rb.clear()
        perform_rb_fill_test(rb, space, 5, rng)
        return rb
    
    perform_rb_fill_test(
        rb,
        space,
        capacity - 1,
        rng
    )
    perform_rb_fill_test(
        rb,
        space,
        1,
        rng
    )
    assert rb.offset == 0
    rb.clear()


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
    space = Box(
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

@pytest.mark.parametrize("capacity", [None, 10])
@pytest.mark.parametrize("seed", [0, 1024])
def test_list_replay_buffer(
    capacity : Optional[int],
    seed : int
):
    space = Box(
        NumpyComputeBackend,
        0.0,
        100.0,
        np.float32,
        shape=(3, 5, 2)
    )
    perform_list_replay_buffer_with_space_test(
        space,
        capacity,
        seed
    )
