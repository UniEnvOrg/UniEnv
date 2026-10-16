"""Fast paths and caches of ``BoxSpace`` must behave exactly like the generic code paths."""
import copy
import math
import pickle

import numpy as np
import pytest

from unienv_interface.space import BoxSpace
from unienv_interface.space.space_utils import batch_utils as sbu
from tests_common.backends import cpu_backend_params

CPU_BACKENDS = cpu_backend_params()


def _dtype(backend, name):
    return getattr(backend, name)


def _np(backend, x):
    return np.asarray(backend.to_numpy(x))


def _generic_box(monkeypatch, *args, **kwargs):
    """Build a box with the scalar-bounds fast path disabled (the original code path)."""
    with monkeypatch.context() as m:
        m.setattr(BoxSpace, "_scalar_bounds_ok", staticmethod(lambda *a: False))
        return BoxSpace(*args, **kwargs)


def _assert_same_bounds(backend, a: BoxSpace, b: BoxSpace):
    assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
    for x, y in ((a._low, b._low), (a._high, b._high)):
        assert tuple(x.shape) == tuple(y.shape)
        assert x.dtype == y.dtype
        assert backend.device(x) == backend.device(y)
        np.testing.assert_array_equal(_np(backend, x), _np(backend, y))
    np.testing.assert_array_equal(_np(backend, a.low), _np(backend, b.low))
    np.testing.assert_array_equal(_np(backend, a.high), _np(backend, b.high))


SCALAR_BOUND_CASES = [
    # (low, high, dtype name, shape)
    (-1.0, 1.0, "float32", (7,)),
    (-1.0, 1.0, "float64", ()),
    (0.0, 0.0, "float32", (2, 3)),
    (-math.inf, math.inf, "float32", (4,)),
    (-math.inf, 2.0, "float32", (2, 2)),
    (0, 255, "uint8", (4, 4, 3)),
    (-math.inf, math.inf, "uint8", (3,)),   # -> dtype min/max
    (-math.inf, math.inf, "int32", ()),
    (-128, 127, "int8", (5,)),
    (0.7, 3.9, "int16", (2,)),              # truncation
    (-3, 2**40, "int64", (2,)),
    (0, 0, "int32", (2, 0)),
    (0.0, 1.0, "float32", (0,)),
]

# Invalid or backend-dependent bounds: both paths must accept/reject them alike
EDGE_SCALAR_BOUND_CASES = [
    (1e39, 1e40, "float32", (2,)),          # overflow when cast to float32 (an error on torch)
    (1.00000001, 1.0, "float32", (2,)),     # low > high in Python, equal in float32
    (1.0000001, 1.0, "float32", (2,)),      # low > high in Python and in float32
    (2.0, 1.0, "float32", (2,)),
    (math.nan, 1.0, "float32", (2,)),
    (0.0, math.nan, "float32", (2,)),
    (5, 4, "int32", (2,)),
    (0.9, 0.2, "int32", (2,)),              # truncates to 0 <= 0 -> valid in both paths
    (300, 400, "uint8", (2,)),
    (-1, 4, "uint8", (2,)),
    (math.nan, 3, "int32", (2,)),
]


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
@pytest.mark.parametrize("low,high,dtype_name,shape", SCALAR_BOUND_CASES)
def test_scalar_bounds_fast_path_matches_generic_path(monkeypatch, backend, device, low, high, dtype_name, shape):
    dtype = _dtype(backend, dtype_name)
    fast = BoxSpace(backend, low, high, dtype, device=device, shape=shape)
    generic = _generic_box(monkeypatch, backend, low, high, dtype, device=device, shape=shape)
    _assert_same_bounds(backend, fast, generic)
    assert repr(fast) == repr(generic)
    assert fast == generic


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
@pytest.mark.parametrize("low,high,dtype_name,shape", EDGE_SCALAR_BOUND_CASES)
def test_scalar_bounds_fast_path_edge_cases_match_generic_path(monkeypatch, backend, device, low, high, dtype_name, shape):
    dtype = _dtype(backend, dtype_name)

    def build(make):
        try:
            return make(), None
        except Exception as e:  # noqa: BLE001 - compare whatever the generic path raises
            return None, type(e)

    fast, fast_err = build(lambda: BoxSpace(backend, low, high, dtype, device=device, shape=shape))
    generic, generic_err = build(lambda: _generic_box(monkeypatch, backend, low, high, dtype, device=device, shape=shape))
    assert fast_err == generic_err
    if fast is not None:
        _assert_same_bounds(backend, fast, generic)


def test_scalar_bounds_ok_rejects_unrepresentable_bounds():
    from unienv_interface.backends.numpy import NumpyComputeBackend as B
    ok = BoxSpace._scalar_bounds_ok
    assert ok(B, 0, 255, B.uint8, None)
    assert not ok(B, 0, 256, B.uint8, None)
    assert not ok(B, -1, 3, B.uint8, None)
    assert not ok(B, 3, 2, B.int32, None)
    assert not ok(B, math.nan, 2.0, B.float32, None)
    assert not ok(B, 0, math.inf, B.int32, None)  # ints map inf to the dtype limits before this check
    assert ok(B, -math.inf, math.inf, B.float32, None)


def _array_bound_boxes(backend, device):
    f32 = backend.float32
    rng = np.random.default_rng(0)
    lo = rng.uniform(-2, -1, size=(4, 3)).astype(np.float32)
    hi = rng.uniform(1, 2, size=(4, 3)).astype(np.float32)
    row_const = np.repeat(np.arange(4, dtype=np.float32)[:, None], 3, axis=1)  # constant within each row
    asarr = lambda x: backend.from_numpy(x, device=device)
    return [
        BoxSpace(backend, -1.0, 1.0, f32, device=device, shape=(4, 3)),
        BoxSpace(backend, asarr(lo), asarr(hi), f32, device=device),
        BoxSpace(backend, asarr(lo[:1]), 5.0, f32, device=device, shape=(4, 3)),
        BoxSpace(backend, -asarr(row_const) - 1, asarr(row_const), f32, device=device),
        BoxSpace(backend, 0, 255, backend.uint8, device=device, shape=(4, 2, 2)),
        BoxSpace(backend, 0.0, 1.0, f32, device=device, shape=(4, 0)),
    ]


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_batch_and_unbatch_match_validating_constructor(backend, device):
    for space in _array_bound_boxes(backend, device):
        batched = sbu.batch_space(space, 5)
        reference = BoxSpace(
            backend, space._low[None], space._high[None], space.dtype,
            device=space.device, shape=(5,) + space.shape,
        )
        _assert_same_bounds(backend, batched, reference)
        unbatched = list(sbu.unbatch_spaces(space))
        assert len(unbatched) == space.shape[0]
        for i, sub in enumerate(unbatched):
            low, high = space._low, space._high
            reference = BoxSpace(
                backend,
                low[i] if low.shape[0] > i else low[0],
                high[i] if high.shape[0] > i else high[0],
                space.dtype, device=space.device, shape=space.shape[1:],
            )
            _assert_same_bounds(backend, sub, reference)
            assert repr(sub) == repr(reference)


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_low_high_are_cached_and_correct(backend, device):
    for space in _array_bound_boxes(backend, device):
        assert space.low is space.low and space.high is space.high
        expected_low = np.broadcast_to(_np(backend, space._low), space.shape)
        np.testing.assert_array_equal(_np(backend, space.low), expected_low)
        assert tuple(space.high.shape) == space.shape


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
@pytest.mark.parametrize("clone", [pickle.loads, copy.deepcopy, copy.copy], ids=["pickle", "deepcopy", "copy"])
def test_box_caches_are_not_pickled(backend, device, clone):
    # device=None: jax device objects themselves cannot be pickled
    space = BoxSpace(backend, -1.0, 1.0, backend.float32, shape=(3,))
    space.low, space.high  # fill the caches
    state = space.__getstate__()
    assert not any(name in state for name in BoxSpace._CACHE_ATTRS)
    restored = clone(pickle.dumps(space)) if clone is pickle.loads else clone(space)
    assert restored == space
    for name in BoxSpace._CACHE_ATTRS:
        assert name in restored.__dict__
    np.testing.assert_array_equal(_np(backend, restored.low), _np(backend, space.low))


def test_box_unpickles_state_without_caches():
    """Pickles written before the caches existed (no cache attributes in the state) still load."""
    from unienv_interface.backends.numpy import NumpyComputeBackend as B
    space = BoxSpace(B, -1.0, 1.0, B.float32, shape=(3,))
    old_state = {k: v for k, v in space.__dict__.items() if k not in BoxSpace._CACHE_ATTRS}
    restored = BoxSpace.__new__(BoxSpace)
    restored.__setstate__(old_state)
    assert restored == space
    np.testing.assert_array_equal(restored.high, np.ones(3, np.float32))


def test_box_low_inside_jit_does_not_leak_tracers():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from unienv_interface.backends.jax import JaxComputeBackend as B
    space = BoxSpace(B, -1.0, 1.0, B.float32, shape=(3,))
    out = jax.jit(lambda x: x + space.low + space.high)(jnp.zeros(3))
    np.testing.assert_allclose(np.asarray(out), np.zeros(3))
    # Values computed inside the trace must not have been cached
    np.testing.assert_allclose(np.asarray(space.low + 0), -np.ones(3))
    assert not isinstance(space.low, jax.core.Tracer)
