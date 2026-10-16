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


def _with_plan(space: BoxSpace, attr: str, plan) -> BoxSpace:
    """A copy of ``space`` (fresh caches) whose cached plan ``attr`` is forced to ``plan``."""
    clone = copy.copy(space)
    setattr(clone, attr, plan)
    return clone


def _contains_cases(backend, device):
    """(space, candidate) pairs covering every contains plan and the dtype/NaN corner cases."""
    asarr = lambda x, dt=None: backend.from_numpy(np.asarray(x, dtype=dt), device=device)
    f32 = backend.float32
    cases = []
    box = BoxSpace(backend, -3.14, 3.14, f32, device=device, shape=(2, 3))
    low = float(_np(backend, box._low).reshape(-1)[0])
    for values in (
        np.zeros((2, 3)), np.full((2, 3), low), np.full((2, 3), -low),
        np.full((2, 3), 3.2), np.full((2, 3), -3.2), [[0, 0, 0], [0, np.nan, 0]],
        [[0, 0, 0], [0, np.inf, 0]], np.full((2, 3), -np.inf),
    ):
        cases.append((box, asarr(values, np.float32)))
    cases.append((box, asarr(np.zeros(6), np.float32)))  # wrong shape
    cases.append((box, asarr(np.zeros((2, 3)), np.int8)))  # int castable to float32
    # float16 just below the float32 low bound: rounding the bound to float16 would accept it
    f16_low = np.float16(low)
    if float(f16_low) < low:
        cases.append((box, asarr(np.full((2, 3), f16_low), np.float16)))
    if backend.__name__ != "JaxComputeBackend":  # jax without x64 has no float64 arrays
        cases.append((box, asarr(np.zeros((2, 3)), np.float64)))
    cases.append((BoxSpace(backend, -1.0, 1.0, f32, device=device, shape=()), asarr(0.5, np.float32)))
    cases.append((BoxSpace(backend, -1.0, 1.0, f32, device=device, shape=()), asarr(1.5, np.float32)))
    cases.append((BoxSpace(backend, -1.0, 1.0, f32, device=device, shape=(0, 3)), asarr(np.zeros((0, 3)), np.float32)))
    # array bounds
    arr_box = BoxSpace(backend, asarr([-1.0, 0.0, 2.0], np.float32), asarr([1.0, 0.5, 3.0], np.float32), f32, device=device)
    for values in ([0, 0.25, 2.5], [0, 0.75, 2.5], [-1, 0, 2], [0, np.nan, 2.5]):
        cases.append((arr_box, asarr(values, np.float32)))
    # integer boxes
    img = BoxSpace(backend, 0, 255, backend.uint8, device=device, shape=(4, 3))
    cases.append((img, asarr(np.arange(12).reshape(4, 3) * 20, np.uint8)))
    cases.append((img, asarr(np.full((4, 3), 7), np.int8)))
    cases.append((img, asarr(np.full((4, 3), True), np.bool_)))
    cases.append((img, asarr(np.full((4, 3), 2), np.int32)))  # lenient can_cast on torch: values decide
    cases.append((img, asarr(np.full((4, 3), 300), np.int32)))
    int_box = BoxSpace(backend, -5, 10, backend.int32, device=device, shape=(3,))
    for values in ([-5, 0, 10], [-6, 0, 0], [0, 0, 11]):
        cases.append((int_box, asarr(values, np.int32)))
    cases.append((int_box, asarr([0, 1, 0], np.bool_)))
    full_i8 = BoxSpace(backend, -np.inf, np.inf, backend.int8, device=device, shape=(2,))
    cases.append((full_i8, asarr([-128, 127], np.int8)))
    cases.append((full_i8, asarr([-128, 127], np.int16)))
    cases.append((full_i8, asarr([-129, 127], np.int16)))
    cases.append((full_i8, asarr([1, 2], np.float32)))
    return cases


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_contains_plans_match_generic_contains(backend, device):
    for space, x in _contains_cases(backend, device):
        generic = _with_plan(space, "_contains_plan", ("generic",))
        expected = generic.contains(x)
        assert space.contains(x) == expected, (space, x)
        assert space.contains(x) == expected  # cached plan
        assert type(space.contains(x)) is bool


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_contains_plan_selection(backend, device):
    f32 = backend.float32
    assert BoxSpace(backend, -1.0, 1.0, f32, device=device, shape=(3,))._build_contains_plan()[0] == "minmax"
    assert BoxSpace(backend, 0, 255, backend.uint8, device=device, shape=(3,))._build_contains_plan() == ("minmax", 0, 255, True)
    assert BoxSpace(backend, 0, 9, backend.uint8, device=device, shape=(3,))._build_contains_plan() == ("minmax", 0, 9, False)
    arr = backend.from_numpy(np.array([0.0, 1.0], np.float32), device=device)
    assert BoxSpace(backend, arr, 2.0, f32, device=device)._build_contains_plan() == ("generic",)
    space = BoxSpace(backend, -1.0, 1.0, f32, device=device, shape=(3,))
    assert space._contains_plan is None
    space.contains(space.low)
    assert space._contains_plan is not None
    assert "_contains_plan" not in copy.copy(space).__getstate__()


def test_contains_on_box_built_inside_jit_is_not_cached():
    """A box built inside a trace holds tracer bounds: plans must not be cached from them."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from unienv_interface.backends.jax import JaxComputeBackend as B
    seen = {}

    def f(x):
        space = BoxSpace(B, -1.0, 1.0, B.float32, shape=(3,))
        seen["plan"] = space._build_contains_plan()
        seen["space"] = space
        return space.clip(x)

    np.testing.assert_allclose(np.asarray(jax.jit(f)(jnp.full(3, 2.0))), np.ones(3))
    assert seen["plan"] is None
    assert seen["space"]._contains_plan is None


def _float_sample_boxes(backend, device):
    asarr = lambda x: backend.from_numpy(np.asarray(x, dtype=np.float32), device=device)
    f32 = backend.float32
    boxes = [
        BoxSpace(backend, -1.0, 2.0, f32, device=device, shape=(4, 7)),
        BoxSpace(backend, -1.0, 2.0, f32, device=device, shape=()),
        BoxSpace(backend, -1.0, 2.0, f32, device=device, shape=(0, 3)),
        BoxSpace(backend, asarr(np.arange(7)), 10.0, f32, device=device),
        BoxSpace(backend, asarr([[-1.0], [0.0]]), asarr([[1.0], [5.0]]), f32, device=device, shape=(2, 3)),
        BoxSpace(backend, 3.0, 3.0, f32, device=device, shape=(2,)),
    ]
    if backend.__name__ != "JaxComputeBackend":  # jax without x64 has no float64
        boxes.append(BoxSpace(backend, -5.0, 5.0, backend.float64, device=device, shape=(3,)))
    return boxes


def _draw(space, backend, device, seed=3, n=3):
    rng = backend.random.random_number_generator(seed, device=device)
    out = []
    for _ in range(n):
        rng, x = space.sample(rng)
        out.append(x)
    return out


NON_JAX_BACKENDS = [p for p in CPU_BACKENDS if p.values[0].__name__ != "JaxComputeBackend"]


@pytest.mark.parametrize("backend,device", NON_JAX_BACKENDS)
def test_bounded_float_sample_is_bit_identical_to_generic_path(backend, device):
    """numpy/torch: the single uniform draw yields exactly the old per-interval draw."""
    from unienv_interface.space import DynamicBoxSpace
    spaces = _float_sample_boxes(backend, device)
    for space in spaces:
        fast = _draw(space, backend, device)
        assert space._sample_plan[0] == "uniform"
        slow = _draw(_with_plan(space, "_sample_plan", ("generic",)), backend, device)
        for a, b in zip(fast, slow):
            assert backend.is_backendarray(a)
            assert a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) == space.shape
            np.testing.assert_array_equal(_np(backend, a), _np(backend, b))
    dyn = DynamicBoxSpace(backend, -1.0, 2.0, (1, 3), (6, 3), backend.float32, device=device)
    fast = _draw(dyn, backend, device, n=5)
    assert dyn._sample_plan[0] == "uniform"
    slow = _draw(_with_plan(dyn, "_sample_plan", ("generic",)), backend, device, n=5)
    for a, b in zip(fast, slow):
        assert tuple(a.shape) == tuple(b.shape)
        np.testing.assert_array_equal(_np(backend, a), _np(backend, b))


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_bounded_float_sample_is_inside_the_box(backend, device):
    for space in _float_sample_boxes(backend, device):
        for x in _draw(space, backend, device, n=5):
            assert backend.is_backendarray(x)
            assert x.dtype == space.dtype and tuple(x.shape) == space.shape
            assert space.contains(x)


INT_SAMPLE_CASES = [
    # (low, high, dtype name, expected plan kind)
    (0, 255, "uint8", "randint"),
    (-np.inf, np.inf, "int8", "randint"),
    (-np.inf, np.inf, "int16", "randint"),
    (0, 3, "int32", "randint"),
    (-7, -7, "int64", "randint"),
    (-(2 ** 31), 2 ** 31 - 3, "int64", "randint"),
    (-np.inf, np.inf, "int32", "generic"),   # high + 1 does not fit int32
    (0, 2 ** 40, "int64", "generic"),
]


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
@pytest.mark.parametrize("low,high,dtype_name,plan_kind", INT_SAMPLE_CASES)
def test_integer_sample_plan(backend, device, low, high, dtype_name, plan_kind):
    space = BoxSpace(backend, low, high, getattr(backend, dtype_name), device=device, shape=(500,))
    lo, hi = (int(v) for v in (_np(backend, space._low).reshape(-1)[0], _np(backend, space._high).reshape(-1)[0]))
    if plan_kind == "generic" and backend.__name__ == "JaxComputeBackend" and hi - lo < 2 ** 32 - 2:
        plan_kind = "randint"  # jax without x64 stores int64 boxes as int32: the bounds fit
    xs = _draw(space, backend, device, n=4)
    assert space._sample_plan[0] == plan_kind
    values = np.concatenate([_np(backend, x) for x in xs])
    for x in xs:
        assert x.dtype == space._low.dtype and tuple(x.shape) == space.shape
        assert space.contains(x)
    assert values.min() >= lo and values.max() <= hi
    if hi - lo <= 255:  # 2000 draws hit both ends of a small closed interval
        assert values.min() == lo and values.max() == hi


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_array_bound_integer_box_uses_generic_sampling(backend, device):
    low = backend.from_numpy(np.array([0, 10, 20], np.int32), device=device)
    space = BoxSpace(backend, low, 30, backend.int32, device=device)
    for x in _draw(space, backend, device, n=5):
        assert space.contains(x)
    assert space._sample_plan == ("generic",)


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_unbounded_float_box_keeps_generic_sampling(backend, device):
    space = BoxSpace(backend, -np.inf, 1.0, backend.float32, device=device, shape=(3,))
    xs = _draw(space, backend, device)
    assert space._sample_plan == ("generic",)
    slow = _draw(_with_plan(space, "_sample_plan", ("generic",)), backend, device)
    for a, b in zip(xs, slow):
        np.testing.assert_array_equal(_np(backend, a), _np(backend, b))


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_integer_scalar_box_samples_are_arrays(backend, device):
    space = BoxSpace(backend, 0, 9, backend.int32, device=device, shape=())
    (x,) = _draw(space, backend, device, n=1)
    assert backend.is_backendarray(x) and tuple(x.shape) == ()
    assert space.contains(x)


def test_dynamic_box_integer_sample_plan():
    from unienv_interface.backends.numpy import NumpyComputeBackend as B
    from unienv_interface.space import DynamicBoxSpace
    space = DynamicBoxSpace(B, 0, 255, (1, 2), (8, 2), B.uint8)
    for x in _draw(space, B, None, n=20):
        assert x.dtype == np.uint8 and x.shape[1] == 2 and 1 <= x.shape[0] <= 8
        assert space.contains(x)
    assert space._sample_plan[0] == "randint"
    assert "_sample_plan" not in pickle.loads(pickle.dumps(space)).__getstate__()


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_sample_plan_is_not_pickled(backend, device):
    space = BoxSpace(backend, -1.0, 1.0, backend.float32, shape=(3,))
    _draw(space, backend, None)
    assert space._sample_plan is not None
    restored = pickle.loads(pickle.dumps(space))
    assert restored._sample_plan is None
    assert restored.contains(_draw(restored, backend, None)[0])


def test_sampling_inside_jit_does_not_cache_tracers():
    jax = pytest.importorskip("jax")
    from unienv_interface.backends.jax import JaxComputeBackend as B
    from unienv_interface.space import DynamicBoxSpace
    rng = B.random.random_number_generator(0)
    for space in (
        BoxSpace(B, -1.0, 1.0, B.float32, shape=(3,)),
        BoxSpace(B, 0, 255, B.uint8, shape=(3,)),
        BoxSpace(B, B.asarray([0, 1, 2]), 9, B.int32),  # generic integer path stays jittable
    ):
        x = jax.jit(lambda r: space.sample(r)[1])(rng)
        assert space.contains(x)
        if space._sample_plan is not None:
            assert not any(isinstance(v, jax.core.Tracer) for v in space._sample_plan)
        _, y = space.sample(rng)  # eager after jit: no leaked tracer
        assert space.contains(y)
    dyn = DynamicBoxSpace(B, -1.0, 1.0, (1, 2), (4, 2), B.float32)
    _, y = dyn.sample(rng)
    assert dyn.contains(y)
