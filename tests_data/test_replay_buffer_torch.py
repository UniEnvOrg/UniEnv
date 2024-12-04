from typing import Generic, TypeVar, Generic, Optional, Any, Dict as DictT, Tuple as TupleT, Sequence as SequenceT, Union as UnionT, List
from unienv_data import *
from unienv_data.storages.common import ListStorage
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_interface.space import *
from unienv_interface.space import flatten_utils as sfu, batch_utils as sbu
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.pytorch import PyTorchComputeBackend
import torch
import tempfile
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
    for i in range(check_size):
        sampled = replay_buffer.get_at(-(check_size - i))
        ref_data = datas[-check_size:][i]
        flat_sampled = sfu.flatten_data(space, sampled)
        flat_data = sfu.flatten_data(space, ref_data)
        assert torch.allclose(flat_sampled, flat_data)

def perform_torch_replay_buffer_with_space_test(
    space: Space[Any, Any, BDeviceType, BDtypeType, BRNGType],
    capacity: int,
    use_mmap : bool = False,
    device : Optional[BDeviceType] = None,
    seed : Optional[int] = None
):
    flattened_space = sfu.flatten_space(space)
    single_instance_shape = flattened_space.shape
    storage = PytorchTensorStorage(
        device,
        flattened_space.dtype,
        single_instance_shape,
        capacity,
        use_mmap=use_mmap,
        mmap_location=None if not use_mmap else tempfile.mktemp(),
    )
    rb = ReplayBuffer(
        storage,
        flattened_space,
    )
    
    rng = torch.Generator(device)
    if seed is not None:
        rng.manual_seed(seed)
    
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
        capacity,
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

@pytest.mark.parametrize("capacity", [10, 50])
@pytest.mark.parametrize("use_mmap", [False, True])
@pytest.mark.parametrize("seed", [0, 1024])
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
        1.0,
        torch.float32,
        device=device,
        shape=(3, 5, 2)
    )
    perform_torch_replay_buffer_with_space_test(
        space,
        capacity,
        use_mmap,
        device,
        seed
    )