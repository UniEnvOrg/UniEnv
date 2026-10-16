"""Tests for ``ToBackendOrDeviceWrapper`` (numpy <-> torch <-> jax)."""

import warnings

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.wrapper import ToBackendOrDeviceWrapper, ToBackendWrapper, ToDeviceWrapper
from unienv_interface.wrapper.backend_compat import data_to
from tests_common.backends import backend_params
from tests_common.envs import StubEnv
from tests_common.trees import to_numpy


def _non_numpy_backend_params():
    return [p for p in backend_params() if p.values[0] is not NumpyComputeBackend]


def _device_of(backend, array):
    return backend.device(array)


@pytest.mark.parametrize("backend,device", _non_numpy_backend_params())
@pytest.mark.parametrize("batch_size", [None, 3])
def test_converts_every_output_and_input(backend, device, batch_size):
    env = StubEnv(batch_size)
    wrapper = ToBackendOrDeviceWrapper(env, backend=backend, device=device)
    assert wrapper.backend is backend
    assert wrapper.device == device
    assert wrapper.observation_space.backend is backend
    assert wrapper.action_space.backend is backend
    assert wrapper.context_space.backend is backend

    context, obs, info = wrapper.reset(seed=3)
    assert wrapper.context_space.contains(context)
    assert wrapper.observation_space.contains(obs)
    assert backend.is_backendarray(obs["position"])
    np.testing.assert_allclose(to_numpy(obs["position"]), env.position)
    if batch_size is not None:
        assert backend.is_backendarray(info["step"])

    action = wrapper.sample_action()
    assert wrapper.action_space.contains(action)
    obs, reward, terminated, truncated, info = wrapper.step(action)
    assert isinstance(env.last_action, np.ndarray)
    np.testing.assert_allclose(env.last_action, to_numpy(action), rtol=1e-6)
    assert wrapper.observation_space.contains(obs)
    if batch_size is None:
        assert isinstance(reward, float) and isinstance(terminated, bool)
    else:
        for value in (reward, terminated, truncated, info["step"]):
            assert backend.is_backendarray(value)
            assert value.shape == (batch_size,)
            if device is not None:
                assert str(_device_of(backend, value)) == str(_device_of(backend, obs["position"]))
        np.testing.assert_allclose(to_numpy(reward), -np.abs(env.position).sum(-1))

        mask = backend.asarray([True, False, True], device=device)
        _, masked_obs, _ = wrapper.reset(mask=mask, seed=1)
        np.testing.assert_array_equal(env.reset_calls[-1]["mask"], [True, False, True])
        assert masked_obs["position"].shape == (2, 2)
        merged = wrapper.update_observation_post_reset(obs, masked_obs, mask)
        assert wrapper.observation_space.contains(merged)

    frame = wrapper.render()
    assert backend.is_backendarray(frame)


@pytest.mark.parametrize("backend,device", _non_numpy_backend_params())
def test_round_trip_back_to_numpy_is_identity(backend, device):
    reference = StubEnv(batch_size=2)
    env = ToBackendOrDeviceWrapper(
        ToBackendOrDeviceWrapper(StubEnv(batch_size=2), backend=backend, device=device),
        backend=NumpyComputeBackend,
    )
    assert env.backend is NumpyComputeBackend
    ref = reference.reset(seed=5)
    out = env.reset(seed=5)
    np.testing.assert_array_equal(out[1]["position"], ref[1]["position"])
    action = np.full((2, 2), 0.5, np.float32)
    ref = reference.step(action)
    out = env.step(action)
    assert isinstance(out[1], np.ndarray)
    np.testing.assert_allclose(out[1], ref[1])
    np.testing.assert_array_equal(out[2], ref[2])
    np.testing.assert_array_equal(out[4]["step"], ref[4]["step"])


@pytest.mark.parametrize("backend,device", _non_numpy_backend_params())
def test_device_defaults_to_inner_device_when_backend_unchanged(backend, device):
    inner = ToBackendOrDeviceWrapper(StubEnv(), backend=backend, device=device)
    outer = ToBackendOrDeviceWrapper(inner)
    assert outer.device == inner.device
    same_backend = ToBackendOrDeviceWrapper(inner, backend=backend)
    assert same_backend.device == inner.device
    _, obs, _ = outer.reset(seed=0)
    assert outer.observation_space.contains(obs)


def test_numpy_to_numpy_is_passthrough(stub_env):
    wrapper = ToBackendOrDeviceWrapper(stub_env)
    assert wrapper.backend is NumpyComputeBackend
    assert wrapper.device is None
    assert wrapper.observation_space is stub_env.observation_space


def test_deprecated_constructors_still_work(stub_env):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert isinstance(ToBackendWrapper(stub_env, NumpyComputeBackend), ToBackendOrDeviceWrapper)
        assert isinstance(ToDeviceWrapper(stub_env, None), ToBackendOrDeviceWrapper)


def test_data_to_handles_nested_containers():
    data = {"a": np.zeros(2), "b": [np.ones(1), "text", 3], "c": (np.ones(1),), "d": None}
    out = data_to(data, NumpyComputeBackend, NumpyComputeBackend)
    assert out["b"][1] == "text" and out["b"][2] == 3 and out["d"] is None
    assert isinstance(out["c"], tuple)


def _tree_items(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _tree_items(value, f"{prefix}.{key}")
    elif isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree):
            yield from _tree_items(value, f"{prefix}[{i}]")
    else:
        yield prefix, tree


@pytest.mark.parametrize("backend,device", _non_numpy_backend_params())
@pytest.mark.parametrize("batch_size", [None, 3])
def test_skipping_noop_device_moves_keeps_step_outputs(backend, device, batch_size):
    fast = ToBackendOrDeviceWrapper(StubEnv(batch_size), backend=backend, device=device)
    slow = ToBackendOrDeviceWrapper(StubEnv(batch_size), backend=backend, device=device)
    slow._step_device = slow.device  # always move explicitly (previous behaviour)
    probe = backend.from_other_backend(NumpyComputeBackend, np.zeros(1))
    lands_on_target = backend.device(probe) == backend.device(backend.to_device(probe, device))
    assert (fast._step_device is None) == lands_on_target
    fast.reset(seed=2)
    slow.reset(seed=2)
    action = backend.from_numpy(np.full(fast.action_space.shape, 0.3, np.float32), device=device)
    for _ in range(3):
        fast_out = fast.step(action)
        slow_out = slow.step(action)
        fast_items, slow_items = list(_tree_items(fast_out)), list(_tree_items(slow_out))
        assert [k for k, _ in fast_items] == [k for k, _ in slow_items]
        for (path, a), (_, b) in zip(fast_items, slow_items):
            assert type(a) is type(b), path
            if backend.is_backendarray(a):
                assert backend.device(a) == backend.device(b), path
                assert a.dtype == b.dtype, path
                np.testing.assert_array_equal(to_numpy(a), to_numpy(b), err_msg=path)
            else:
                assert a == b, path
        assert fast.observation_space.contains(fast_out[0])


def test_step_device_kept_for_non_numpy_source():
    pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    from unienv_interface.backends.jax import JaxComputeBackend
    from unienv_interface.backends.pytorch import PyTorchComputeBackend
    inner = ToBackendOrDeviceWrapper(StubEnv(), backend=PyTorchComputeBackend, device="cpu")
    jax_device = jax.devices("cpu")[0]
    outer = ToBackendOrDeviceWrapper(inner, backend=JaxComputeBackend, device=jax_device)
    # only numpy sources are probed; torch -> jax keeps the explicit move
    assert outer._step_device == jax_device
    outer.reset(seed=0)
    obs, *_ = outer.step(JaxComputeBackend.from_numpy(np.zeros(2, np.float32)))
    assert JaxComputeBackend.device(obs["position"]) == jax_device
