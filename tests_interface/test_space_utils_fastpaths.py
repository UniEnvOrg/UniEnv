"""Fast paths in ``flatten_utils`` / ``batch_utils`` must match the original implementations."""
import numpy as np
import pytest

from unienv_interface.space import (
    BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, TupleSpace, UnionSpace,
)
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.backends.numpy import NumpyComputeBackend
from tests_common.backends import cpu_backend_params

CPU_BACKENDS = cpu_backend_params()


def _np(backend, x):
    return np.asarray(backend.to_numpy(x))


def _reference_unflatten(space, data, start_dim=0):
    """The original Dict/Tuple unflatten: gather each part with ``take(arange)``."""
    if isinstance(space, (DictSpace, TupleSpace)):
        items = sfu._dict_items(space) if isinstance(space, DictSpace) else list(enumerate(space.spaces))
        result, start = {}, 0
        for key, subspace in items:
            end = start + sfu.flat_dim(subspace, start_dim)
            idx = space.backend.arange(start, end, dtype=space.backend.default_integer_dtype, device=space.backend.device(data))
            result[key] = _reference_unflatten(subspace, space.backend.take(data, idx, axis=start_dim), start_dim)
            start = end
        return result if isinstance(space, DictSpace) else tuple(result[i] for i in range(len(items)))
    return sfu.unflatten_data(space, data, start_dim)


def _leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, tuple):
        for v in x:
            yield from _leaves(v)
    else:
        yield x


def _nested_space(backend, device, batch=()):
    f32 = backend.float32
    box = lambda lo, hi, shape, dt=f32: BoxSpace(backend, lo, hi, dt, device=device, shape=batch + shape)
    return DictSpace(backend, {
        "pos": box(-1.0, 1.0, (3,)),
        "img": box(0, 255, (2, 2, 3), backend.uint8),
        "flag": BinarySpace(backend, batch + (2,), device=device),
        "nested": DictSpace(backend, {
            "a": box(-5.0, 5.0, (1,)),
            "t": TupleSpace(backend, [box(0.0, 1.0, (2,)), box(-2, 2, (4,), backend.int32)], device=device),
        }, device=device),
    }, device=device)


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
@pytest.mark.parametrize("batch,start_dim", [((), 0), ((4,), 1), ((4,), -1), ((2, 3), 2)])
def test_unflatten_slicing_matches_take(backend, device, batch, start_dim):
    space = _nested_space(backend, device, batch)
    if start_dim < 0:
        # start_dim=-1 flattens only the last axis of every leaf; use a space where that is consistent
        space = TupleSpace(backend, [
            BoxSpace(backend, -1.0, 1.0, backend.float32, device=device, shape=batch + (3,)),
            BoxSpace(backend, 0, 9, backend.int32, device=device, shape=batch + (2,)),
        ], device=device)
    rng = backend.random.random_number_generator(0, device=device)
    _, x = space.sample(rng)
    flat = sfu.flatten_data(space, x, start_dim)
    got = sfu.unflatten_data(space, flat, start_dim)
    expected = _reference_unflatten(space, flat, start_dim)
    assert space.contains(got)
    for g, e, orig in zip(_leaves(got), _leaves(expected), _leaves(x)):
        assert g.dtype == e.dtype and tuple(g.shape) == tuple(e.shape)
        np.testing.assert_array_equal(_np(backend, g), _np(backend, e))
        np.testing.assert_array_equal(_np(backend, g), _np(backend, orig))


@pytest.mark.parametrize("backend,device", [p for p in CPU_BACKENDS if p.values[0].__name__ != "JaxComputeBackend"])
def test_unflatten_results_do_not_alias_flat_input(backend, device):
    """Mutable backends: writing into the flat array must not change unflattened data."""
    spaces = [
        _nested_space(backend, device),
        UnionSpace(backend, [BoxSpace(backend, -1.0, 1.0, backend.float32, device=device, shape=(3,))], device=device),
        TupleSpace(backend, [DynamicBoxSpace(backend, -1.0, 1.0, (1, 2), (3, 2), backend.float32, device=device)], device=device),
    ]
    rng = backend.random.random_number_generator(1, device=device)
    for space in spaces:
        rng, x = space.sample(rng)
        flat = sfu.flatten_data(space, x)
        got = sfu.unflatten_data(space, flat)
        before = [_np(backend, leaf).copy() for leaf in _leaves(got) if backend.is_backendarray(leaf)]
        flat[...] = 0  # numpy and torch both support in-place assignment
        after = [_np(backend, leaf) for leaf in _leaves(got) if backend.is_backendarray(leaf)]
        for b, a in zip(before, after):
            np.testing.assert_array_equal(a, b)


def test_flat_dim_returns_python_ints():
    B = NumpyComputeBackend
    space = _nested_space(B, None, (4,))
    for start_dim in (0, 1):
        assert type(sfu.flat_dim(space, start_dim)) is int
        assert type(sfu.flat_dim(space["pos"], start_dim)) is int
    assert sfu.flat_dim(BoxSpace(B, 0.0, 1.0, B.float32, shape=()), 0) == 1
    assert sfu.flat_dim(DynamicBoxSpace(B, 0.0, 1.0, (1, 2), (3, 2), B.float32), 0) == 6


SET_AT_INDICES = [
    0, -1, np.int64(2), slice(1, 3), (1, 0), (slice(None), 1), [0, 2],
    np.array([3, 1]), np.array([True, False, True, False]), Ellipsis,
]


@pytest.mark.parametrize("index", SET_AT_INDICES, ids=repr)
@pytest.mark.parametrize("value_kind", ["array", "scalar", "np_scalar"])
def test_set_at_numpy_fast_path_matches_array_api_at(index, value_kind):
    import array_api_extra as xpx
    B = NumpyComputeBackend
    space = BoxSpace(B, -10.0, 10.0, B.float32, shape=(4, 3))
    items = np.arange(12, dtype=np.float32).reshape(4, 3)
    target_shape = items[index].shape
    value = {
        "array": np.full(target_shape, 7.5, dtype=np.float64),
        "scalar": 5,
        "np_scalar": np.float32(-2.5),
    }[value_kind]
    expected = xpx.at(items.copy())[index].set(value)
    got = sbu.set_at(space, items, index, value)
    assert got is items  # writeable numpy arrays are updated in place, as before
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


def test_set_at_numpy_read_only_array_is_copied():
    B = NumpyComputeBackend
    space = BoxSpace(B, -10.0, 10.0, B.float32, shape=(4,))
    items = np.zeros(4, np.float32)
    items.flags.writeable = False
    got = sbu.set_at(space, items, 1, 3.0)
    assert got is not items
    np.testing.assert_array_equal(items, np.zeros(4))
    np.testing.assert_array_equal(got, [0, 3, 0, 0])


def test_set_at_numpy_with_foreign_value_still_goes_through_array_api_at():
    torch = pytest.importorskip("torch")
    B = NumpyComputeBackend
    space = BoxSpace(B, -10.0, 10.0, B.float32, shape=(4,))
    with pytest.raises(TypeError):
        sbu.set_at(space, np.zeros(4, np.float32), 1, torch.ones(()))


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_set_at_dict_matches_per_leaf_at(backend, device):
    space = sbu.batch_space(_nested_space(backend, device), 5)
    rng = backend.random.random_number_generator(2, device=device)
    rng, items = space.sample(rng)
    _, value = sbu.batch_space(_nested_space(backend, device), 2).sample(rng)
    expected = [backend.at(backend.asarray(leaf, copy=True))[1:3].set(v) for leaf, v in zip(_leaves(items), _leaves(value))]
    got = sbu.set_at(space, items, slice(1, 3), value)
    for g, e in zip(_leaves(got), expected):
        np.testing.assert_array_equal(_np(backend, g), _np(backend, e))
