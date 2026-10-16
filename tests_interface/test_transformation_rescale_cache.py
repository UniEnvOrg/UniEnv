"""RescaleTransformation caches its affine constants per (source space, data device)."""
import copy
import pickle

import numpy as np
import pytest

from unienv_interface.space import BoxSpace
from unienv_interface.transformations import RescaleTransformation
from unienv_interface.transformations.rescale import _get_broadcastable_value
from unienv_interface.backends.numpy import NumpyComputeBackend
from tests_common.backends import cpu_backend_params

CPU_BACKENDS = cpu_backend_params()
B = NumpyComputeBackend


def _np(backend, x):
    return np.asarray(backend.to_numpy(x))


def _reference_transform(t: RescaleTransformation, source_space: BoxSpace, data):
    """The original, uncached implementation."""
    backend = source_space.backend
    ndim = len(source_space.shape)
    target_low = backend.to_device(_get_broadcastable_value(backend, t.new_low, ndim), backend.device(data)) if backend.is_backendarray(t.new_low) else t.new_low
    target_high = backend.to_device(_get_broadcastable_value(backend, t.new_high, ndim), backend.device(data)) if backend.is_backendarray(t.new_high) else t.new_high
    span = source_space._high - source_space._low
    mask = span == 0
    scaling = (target_high - target_low) / backend.where(mask, 1, span)
    out = (data - source_space._low) * scaling + target_low
    out = backend.where(mask, target_low, out)
    if t.nan_to is not None:
        out = backend.where(backend.isnan(out), t.nan_to, out)
    if t.new_dtype is not None:
        if backend.dtype_is_real_integer(t.new_dtype) and backend.dtype_is_real_floating(out.dtype):
            out = backend.round(out)
        out = backend.astype(out, t.new_dtype)
    return out


def _cases(backend, device):
    """(source space, transformation, data) triples."""
    asarr = lambda x, dt=np.float32: backend.from_numpy(np.asarray(x, dtype=dt), device=device)
    f32 = backend.float32
    act = BoxSpace(backend, -2.0, 2.0, f32, device=device, shape=(7,))
    arr_box = BoxSpace(backend, asarr([-1.0, 0.0, 3.0]), asarr([1.0, 0.0, 5.0]), f32, device=device)  # one zero-span dim
    scalar_box = BoxSpace(backend, -1.0, 3.0, f32, device=device, shape=())
    flat_box = BoxSpace(backend, 2.0, 2.0, f32, device=device, shape=(2,))  # all zero-span
    img = BoxSpace(backend, 0, 255, backend.uint8, device=device, shape=(2, 2, 3))
    cases = [
        (act, RescaleTransformation(-1.0, 1.0), asarr(np.linspace(-2, 2, 7))),
        (act, RescaleTransformation(0.0, 10.0), asarr(np.zeros((4, 7)))),  # extra leading batch dim
        (act, RescaleTransformation(asarr(np.full(7, -3.0)), asarr(np.arange(7))), asarr(np.linspace(-2, 2, 7))),
        (act, RescaleTransformation(-1.0, asarr(np.ones(7))), asarr(np.linspace(-2, 2, 7))),
        (act, RescaleTransformation(-1.0, 1.0, nan_to=0.0), asarr([np.nan, 0, 1, 2, -2, 0.5, 1.5])),
        (act, RescaleTransformation(0.0, 100.0, new_dtype=backend.int32), asarr(np.linspace(-2, 2, 7))),
        (arr_box, RescaleTransformation(-1.0, 1.0), asarr([0.5, 0.0, 4.0])),
        (arr_box, RescaleTransformation(asarr([-5.0, -6.0, -7.0]), 1.0), asarr([0.5, 0.0, 4.0])),
        (scalar_box, RescaleTransformation(-1.0, 1.0), asarr(1.0)),
        (flat_box, RescaleTransformation(-1.0, 1.0), asarr([2.0, 2.0])),
        (img, RescaleTransformation(0.0, 1.0), asarr(np.arange(12).reshape(2, 2, 3) * 20, np.uint8)),
    ]
    return cases


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_rescale_matches_uncached_transform(backend, device):
    for space, t, x in _cases(backend, device):
        expected = _reference_transform(t, space, x)
        for _ in range(2):  # second call uses the cached constants
            got = t.transform(space, x)
            assert backend.is_backendarray(got)
            assert got.dtype == expected.dtype and tuple(got.shape) == tuple(expected.shape)
            np.testing.assert_array_equal(_np(backend, got), _np(backend, expected))


def test_rescale_constants_cache_keys():
    space = BoxSpace(B, -2.0, 2.0, B.float32, shape=(3,))
    other = BoxSpace(B, -4.0, 4.0, B.float32, shape=(3,))
    x = np.ones(3, np.float32)
    t = RescaleTransformation(-1.0, 1.0)
    np.testing.assert_allclose(t.transform(space, x), np.full(3, 0.5))
    cache = t._constants_cache
    assert cache[0] is space and cache[1] is None  # scalar targets: data device irrelevant
    t.transform(space, x)
    assert t._constants_cache is cache
    np.testing.assert_allclose(t.transform(other, x), np.full(3, 0.25))  # new space -> rebuilt
    assert t._constants_cache[0] is other
    arr_t = RescaleTransformation(np.full(3, -1.0, np.float32), 1.0)
    arr_t.transform(space, x)
    assert arr_t._constants_cache[1] == B.device(x)


def test_reassigning_targets_rebuilds_constants():
    space = BoxSpace(B, -2.0, 2.0, B.float32, shape=(3,))
    x = np.ones(3, np.float32)
    t = RescaleTransformation(-1.0, 1.0)
    np.testing.assert_allclose(t.transform(space, x), np.full(3, 0.5))
    t.new_low = 0.0
    np.testing.assert_allclose(t.transform(space, x), np.full(3, 0.75))
    t.new_high = np.full(3, 2.0, np.float32)
    np.testing.assert_allclose(t.transform(space, x), np.full(3, 1.5))


def test_rescale_constants_are_keyed_on_the_data_device(monkeypatch):
    """Array targets are moved to the data's device: a new data device rebuilds the constants."""
    current = {"device": "dev0"}
    moved_to = []
    monkeypatch.setattr(B, "device", lambda x: current["device"])
    monkeypatch.setattr(B, "to_device", lambda x, device: (moved_to.append(device), x)[1])
    space = BoxSpace(B, -2.0, 2.0, B.float32, shape=(3,))
    t = RescaleTransformation(np.full(3, -1.0, np.float32), 1.0)
    x = np.ones(3, np.float32)
    for device in ("dev0", "dev0", "dev1", "dev1", "dev0"):
        current["device"] = device
        np.testing.assert_allclose(t.transform(space, x), np.full(3, 0.5))
        assert t._constants_cache[1] == device
    assert moved_to == ["dev0", "dev1", "dev0"]  # one transfer per device change, none per call


@pytest.mark.gpu
def test_rescale_constants_per_device_on_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    t = RescaleTransformation(torch.full((3,), -1.0), torch.full((3,), 1.0))
    cpu_space = BoxSpace(TB, -2.0, 2.0, TB.float32, shape=(3,), device=torch.device("cpu"))
    cuda_space = BoxSpace(TB, -2.0, 2.0, TB.float32, shape=(3,), device=torch.device("cuda"))
    for space in (cpu_space, cuda_space, cpu_space):
        x = torch.ones(3, device=space.device)
        out = t.transform(space, x)
        assert out.device == x.device
        assert t._constants_cache[0] is space and t._constants_cache[1] == x.device
        assert t._constants_cache[3][2].device == x.device  # the moved target bound
        np.testing.assert_allclose(out.cpu().numpy(), np.full(3, 0.5))


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_rescale_cache_is_not_pickled(backend, device):
    space = BoxSpace(backend, -2.0, 2.0, backend.float32, shape=(3,))
    t = RescaleTransformation(-1.0, 1.0)
    x = backend.from_numpy(np.ones(3, np.float32))
    expected = _np(backend, t.transform(space, x))
    assert "_constants_cache" not in t.__getstate__()
    for restored in (pickle.loads(pickle.dumps(t)), copy.deepcopy(t)):
        assert "_constants_cache" not in restored.__dict__
        np.testing.assert_array_equal(_np(backend, restored.transform(space, x)), expected)
    old_state = {k: v for k, v in t.__dict__.items() if k not in ("_constants_cache", "nan_to")}
    old = RescaleTransformation.__new__(RescaleTransformation)
    old.__setstate__(old_state)
    assert old.nan_to is None
    np.testing.assert_array_equal(_np(backend, old.transform(space, x)), expected)


def test_rescale_under_jit_caches_nothing_and_leaks_no_tracers():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from unienv_interface.backends.jax import JaxComputeBackend as JB
    zero_span = BoxSpace(JB, JB.asarray([-1.0, 2.0]), JB.asarray([1.0, 2.0]), JB.float32)
    for t in (RescaleTransformation(-1.0, 1.0), RescaleTransformation(jnp.full((2,), -1.0), jnp.full((2,), 1.0))):
        x = jnp.asarray([0.5, 2.0])
        eager = t.transform(zero_span, x)
        np.testing.assert_allclose(np.asarray(eager), [0.5, -1.0])
        jitted = jax.jit(lambda v: t.transform(zero_span, v))(x)
        np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager))
        assert "_constants_cache" not in t.__dict__
        np.testing.assert_allclose(np.asarray(t.transform(zero_span, x)), np.asarray(eager))
        np.testing.assert_allclose(np.asarray(jax.jit(lambda v: t.transform(zero_span, v))(x * 0)), [0.0, -1.0])
