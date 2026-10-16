"""Equivalence tests for the ``FuncSpaceDataQueue`` numpy / torch fast path.

The fast path (``empty_like`` + two slice copies per leaf, plain transposes for the
output) must reproduce the generic ``roll`` + ``sbu.set_at`` path exactly, including
masked resets, aliasing guarantees and pickling.
"""

import pickle

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.space import BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, TupleSpace
from unienv_interface.space import batch_utils as sbu
from unienv_interface.utils.framestack_queue import FuncSpaceDataQueue, SpaceDataQueue
from unienv_interface.wrapper import FrameStackWrapper
from tests_common.backends import backend_params
from tests_common.envs import StubEnv


def _dtype(backend, np_dtype):
    return backend.from_numpy(np.zeros(1, np_dtype)).dtype


def _nested_space(backend, device, batch_size):
    bs = () if batch_size is None else (batch_size,)
    return DictSpace(backend, {
        "pos": BoxSpace(backend, -5.0, 5.0, _dtype(backend, np.float32), device=device, shape=bs + (3,)),
        "flag": BinarySpace(backend, bs + (2,), device=device),
        "tup": TupleSpace(backend, [
            BoxSpace(backend, -100, 100, _dtype(backend, np.int64), device=device, shape=bs),
            BoxSpace(backend, 0, 255, _dtype(backend, np.uint8), device=device, shape=bs + (4, 4)),
        ]),
    })


def _to_numpy(backend, data):
    if isinstance(data, dict):
        return {k: _to_numpy(backend, v) for k, v in data.items()}
    if isinstance(data, tuple):
        return tuple(_to_numpy(backend, v) for v in data)
    return np.array(backend.to_numpy(data), copy=True)


def _assert_tree_equal(a, b):
    assert type(a) is type(b)
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_tree_equal(a[key], b[key])
    elif isinstance(a, tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_tree_equal(x, y)
    else:
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)


def _queues(space, batch_size, maxlen=3):
    fast = FuncSpaceDataQueue(space, batch_size, maxlen)
    ref = FuncSpaceDataQueue(space, batch_size, maxlen)
    ref._fast_plan = None  # force the generic path
    return fast, ref


@pytest.mark.parametrize("backend,device", backend_params())
@pytest.mark.parametrize("batch_size", [None, 4])
def test_fast_path_matches_generic_path(backend, device, batch_size):
    space = _nested_space(backend, device, batch_size)
    fast, ref = _queues(space, batch_size)
    assert (fast._fast_plan is not None) == (backend.simplified_name in ("numpy", "pytorch"))

    rng = backend.random.random_number_generator(0, device=device)
    rng, first = space.sample(rng)
    fast_state, ref_state = fast.init(first), ref.init(first)
    held = []  # (returned output, numpy snapshot) pairs that must never change
    mask_rng = np.random.default_rng(1)
    for t in range(12):
        rng, value = space.sample(rng)
        if batch_size is not None and t % 4 == 3:
            mask_np = mask_rng.random(batch_size) < 0.5
            mask = backend.from_numpy(mask_np, device=device)
            value = sbu.get_at(space, value, mask)
            fast_state = fast.reset(fast_state, value, mask)
            ref_state = ref.reset(ref_state, value, mask)
        elif t == 6:
            fast_state, ref_state = fast.reset(fast_state, value), ref.reset(ref_state, value)
        else:
            fast_state, ref_state = fast.add(fast_state, value), ref.add(ref_state, value)
        out = fast.get_output_data(fast_state)
        out_np = _to_numpy(backend, out)
        _assert_tree_equal(out_np, _to_numpy(backend, ref.get_output_data(ref_state)))
        assert fast.output_space.contains(out)
        held.append((out, out_np))
    for out, snapshot in held:
        _assert_tree_equal(_to_numpy(backend, out), snapshot)


@pytest.mark.parametrize("backend,device", backend_params())
def test_fast_path_add_does_not_alias_input(backend, device):
    space = BoxSpace(backend, -5.0, 5.0, _dtype(backend, np.float32), device=device, shape=(2, 3))
    queue = FuncSpaceDataQueue(space, 2, maxlen=2)
    if backend.simplified_name not in ("numpy", "pytorch"):
        pytest.skip("immutable backend")
    value = backend.from_numpy(np.ones((2, 3), np.float32), device=device)
    state = queue.init(value * 0)
    before = state.data
    before_np = _to_numpy(backend, before)
    state = queue.add(state, value)
    value[...] = 7.0  # mutate the caller's array after handing it over
    out = _to_numpy(backend, queue.get_output_data(state))
    np.testing.assert_array_equal(out[:, -1], 1.0)
    np.testing.assert_array_equal(out[:, 0], 0.0)
    # the previous state is untouched (functional semantics)
    _assert_tree_equal(_to_numpy(backend, before), before_np)


def test_fast_path_not_used_for_other_spaces():
    dyn = DynamicBoxSpace(NB, 0.0, 1.0, (1,), (3,), np.float32)
    assert FuncSpaceDataQueue(dyn, None, 2)._fast_plan is None
    nested = DictSpace(NB, {"ok": BoxSpace(NB, 0.0, 1.0, np.float32, shape=(2,)), "dyn": dyn})
    assert FuncSpaceDataQueue(nested, None, 2)._fast_plan is None
    plain = DictSpace(NB, {"ok": BoxSpace(NB, 0.0, 1.0, np.float32, shape=(2,))})
    assert FuncSpaceDataQueue(plain, None, 2)._fast_plan == {"ok": "leaf"}


def test_queue_pickles_and_keeps_fast_path():
    space = _nested_space(NB, None, 3)
    queue = SpaceDataQueue(space, 3, maxlen=3)
    rng = NB.random.random_number_generator(0)
    rng, value = space.sample(rng)
    queue.reset(value)
    rng, value = space.sample(rng)
    queue.add(value)
    clone = pickle.loads(pickle.dumps(queue))
    assert clone.func_queue._fast_plan is not None
    rng, value = space.sample(rng)
    queue.add(value)
    clone.add(value)
    _assert_tree_equal(_to_numpy(NB, clone.get_output_data()), _to_numpy(NB, queue.get_output_data()))


def test_frame_stack_wrapper_pickles_mid_episode():
    zero = np.zeros(2, np.float32)
    wrapper = FrameStackWrapper(StubEnv(batch_size=3), obs_stack_size=2, action_stack_size=2, action_default_value=zero)
    wrapper.reset(seed=0)
    action = np.full((3, 2), 0.1, np.float32)
    wrapper.step(action)
    clone = pickle.loads(pickle.dumps(wrapper))
    for _ in range(3):
        obs, *_ = wrapper.step(action)
        clone_obs, *_ = clone.step(action)
        _assert_tree_equal(clone_obs, obs)
