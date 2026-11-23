"""Tests for ``TransformWrapper`` / ``ContextObservationTransformWrapper``."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace
from unienv_interface.transformations.batch_and_unbatch import BatchifyTransformation
from unienv_interface.transformations.filter_dict import DictIncludeKeyTransformation
from unienv_interface.transformations.rescale import RescaleTransformation
from unienv_interface.wrapper import ContextObservationTransformWrapper
from unienv_interface.wrapper.transformation import TransformWrapper
from tests_common.envs import StubEnv


@pytest.mark.parametrize("cls", [TransformWrapper, ContextObservationTransformWrapper])
def test_context_space_forwarded_without_context_transformation(any_stub_env, cls):
    wrapper = cls(any_stub_env, observation_transformation=DictIncludeKeyTransformation(["position"]))
    assert wrapper.context_space is any_stub_env.context_space
    context, obs, _ = wrapper.reset(seed=2)
    assert wrapper.context_space.contains(context)
    assert wrapper.observation_space.contains(obs)
    assert obs.keys() == {"position"}


@pytest.mark.parametrize("batch_size", [None, 3])
def test_transform_wrapper_all_transformations(batch_size):
    env = StubEnv(batch_size)
    shape = env.action_space.shape
    target_action_space = BoxSpace(B, 0.0, 4.0, np.float32, shape=shape)
    wrapper = TransformWrapper(
        env,
        context_transformation=BatchifyTransformation(),
        observation_transformation=DictIncludeKeyTransformation(["velocity"]),
        action_transformation=RescaleTransformation(-1.0, 1.0),
        target_action_space=target_action_space,
    )
    assert wrapper.action_space is target_action_space
    context, obs, _ = wrapper.reset(seed=3)
    assert wrapper.context_space.contains(context)
    np.testing.assert_array_equal(context[0], np.full(shape, 3.0))
    obs, *_ = wrapper.step(np.full(shape, 3.0, np.float32))
    np.testing.assert_allclose(env.last_action, 0.5)
    np.testing.assert_allclose(obs["velocity"], 0.5)
    assert wrapper.observation_space.contains(obs)
    with pytest.raises(AssertionError):
        wrapper.set_action_transformation(RescaleTransformation(-1.0, 1.0), None)


def test_context_observation_transform_wrapper_reverse_maps():
    env = StubEnv()
    wrapper = ContextObservationTransformWrapper(env, context_transformation=BatchifyTransformation())
    context, obs, _ = wrapper.reset(seed=1)
    assert context.shape == (1, 2)
    np.testing.assert_array_equal(wrapper.reverse_map_context(context), env.goal)
    assert wrapper.reverse_map_observation(obs) is obs
    non_invertible = ContextObservationTransformWrapper(
        env, observation_transformation=DictIncludeKeyTransformation(["position"]))
    with pytest.raises(NotImplementedError):
        non_invertible.reverse_map_observation({"position": np.zeros(2, np.float32)})
    with pytest.raises(AssertionError):
        ContextObservationTransformWrapper(env)


# ---------------------------------------------------------------------------
# Regression: masked resets are transformed with the masked sub-space
# ---------------------------------------------------------------------------

_PER_ENV_HIGH = np.array([[10.0, 10.0], [20.0, 20.0], [40.0, 40.0]], np.float32)


def _per_env_bounds_env():
    """Batched StubEnv whose ``position`` / context bounds differ per env (``high`` = 10, 20, 40)."""
    from unienv_interface.space import DictSpace
    env = StubEnv(3)
    box = lambda: BoxSpace(B, np.zeros((3, 2), np.float32), _PER_ENV_HIGH, np.float32, shape=(3, 2))
    env.observation_space = DictSpace(B, {"position": box(), "velocity": env.observation_space["velocity"]})
    env.context_space = box()
    return env


@pytest.mark.parametrize("cls", [TransformWrapper, ContextObservationTransformWrapper])
@pytest.mark.parametrize("mask", [[False, True, False], [True, False, True], [False, False, False], [True, True, True]])
def test_masked_reset_uses_masked_space_with_per_env_bounds(cls, mask):
    from unienv_interface.transformations.dict_transform import DictTransformation
    env = _per_env_bounds_env()
    wrapper = cls(
        env,
        context_transformation=RescaleTransformation(0.0, 1.0),
        observation_transformation=DictTransformation({"position": RescaleTransformation(0.0, 1.0)}),
    )
    wrapper.reset(seed=1)
    mask = np.array(mask)
    context, obs, _ = wrapper.reset(mask=mask, seed=4)
    k = int(mask.sum())
    assert context.shape == (k, 2) and obs["position"].shape == (k, 2)
    np.testing.assert_allclose(context, env.goal[mask] / _PER_ENV_HIGH[mask], rtol=1e-6)
    np.testing.assert_allclose(obs["position"], env.position[mask] / _PER_ENV_HIGH[mask], rtol=1e-6)


from tests_common.backends import backend_params


@pytest.mark.parametrize("backend,device", backend_params())
def test_masked_reset_masked_space_on_every_backend(backend, device):
    from unienv_interface.wrapper import ToBackendOrDeviceWrapper
    env = _per_env_bounds_env()
    converted = ToBackendOrDeviceWrapper(env, backend=backend, device=device)
    wrapper = ContextObservationTransformWrapper(converted, context_transformation=RescaleTransformation(0.0, 1.0))
    wrapper.reset(seed=1)
    mask_np = np.array([True, False, True])
    mask = backend.from_numpy(mask_np, device=device) if backend.simplified_name != "numpy" else mask_np
    context, _, _ = wrapper.reset(mask=mask, seed=4)
    assert tuple(context.shape) == (2, 2)
    np.testing.assert_allclose(backend.to_numpy(context), env.goal[mask_np] / _PER_ENV_HIGH[mask_np], rtol=1e-6)


@pytest.mark.parametrize("mask", [[True, False, True], [False, False, False], [True, True, True]])
def test_masked_space_of_nested_spaces(mask):
    from unienv_interface.env_base._utils import masked_space
    from unienv_interface.space import BinarySpace, DictSpace, TupleSpace
    from unienv_interface.space.space_utils import batch_utils as sbu
    space = DictSpace(B, {
        "a": BoxSpace(B, np.zeros((3, 2), np.float32), _PER_ENV_HIGH, np.float32, shape=(3, 2)),
        "t": TupleSpace(B, [BinarySpace(B, shape=(3, 4)), BoxSpace(B, -1.0, 1.0, np.float32, shape=(3,))]),
    })
    mask = np.array(mask)
    sub = masked_space(space, mask)
    k = int(mask.sum())
    assert sub["a"].shape == (k, 2) and sub["t"][0].shape == (k, 4) and sub["t"][1].shape == (k,)
    data = space.sample(B.random.random_number_generator(0))[1]
    assert sub.contains(sbu.get_at(space, data, mask))
    if k:
        np.testing.assert_array_equal(np.broadcast_to(sub["a"].high, (k, 2)), _PER_ENV_HIGH[mask])
    str(sub)  # printable, also with 0 rows
