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
