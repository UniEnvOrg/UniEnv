"""Tests for FlattenDictTransformation and UnflattenDictTransformation."""
import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.transformations import (
    FlattenDictTransformation,
    UnflattenDictTransformation,
)
from unienv_interface.space import DictSpace
from test_utils import make_random_box_space, verify_transformation_serialization


ALL_BACKENDS = [NumpyComputeBackend, JaxComputeBackend, PyTorchComputeBackend]
SEEDS = [0, 42, 123]


def make_nested_dict_space(backend, rng, np_rng):
    arm_pos_space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    arm_vel_space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    gripper_space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    reward_space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    return DictSpace(
        backend,
        {
            "robot": DictSpace(
                backend,
                {
                    "arm": DictSpace(
                        backend,
                        {
                            "pos": arm_pos_space,
                            "vel": arm_vel_space,
                        },
                        device=None,
                    ),
                    "gripper": gripper_space,
                },
                device=None,
            ),
            "reward": reward_space,
        },
        device=None,
    ), rng


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_flatten_dict_transformation(backend, seed):
    """Test flattening nested dict spaces and data."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)

    source_space, rng = make_nested_dict_space(backend, rng, np_rng)
    transform = FlattenDictTransformation(nested_separator=".")

    target_space = transform.get_target_space_from_source(source_space)
    assert set(target_space.spaces.keys()) == {
        "robot.arm.pos",
        "robot.arm.vel",
        "robot.gripper",
        "reward",
    }

    rng, data = source_space.sample(rng)
    transformed = transform.transform(source_space, data)
    assert set(transformed.keys()) == set(target_space.spaces.keys())
    assert backend.all(transformed["robot.arm.pos"] == data["robot"]["arm"]["pos"])
    assert backend.all(transformed["robot.arm.vel"] == data["robot"]["arm"]["vel"])
    assert backend.all(transformed["robot.gripper"] == data["robot"]["gripper"])
    assert backend.all(transformed["reward"] == data["reward"])


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_unflatten_dict_transformation(backend, seed):
    """Test unflattening dict spaces and data."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)

    nested_space, rng = make_nested_dict_space(backend, rng, np_rng)
    flatten_transform = FlattenDictTransformation(nested_separator=".")
    flat_space = flatten_transform.get_target_space_from_source(nested_space)

    unflatten_transform = UnflattenDictTransformation(nested_separator=".")
    target_space = unflatten_transform.get_target_space_from_source(flat_space)
    assert target_space == nested_space

    rng, nested_data = nested_space.sample(rng)
    flat_data = flatten_transform.transform(nested_space, nested_data)
    unflattened_data = unflatten_transform.transform(flat_space, flat_data)
    assert backend.all(unflattened_data["robot"]["arm"]["pos"] == nested_data["robot"]["arm"]["pos"])
    assert backend.all(unflattened_data["robot"]["arm"]["vel"] == nested_data["robot"]["arm"]["vel"])
    assert backend.all(unflattened_data["robot"]["gripper"] == nested_data["robot"]["gripper"])
    assert backend.all(unflattened_data["reward"] == nested_data["reward"])

    inverse_transform = unflatten_transform.direction_inverse(flat_space)
    recovered_flat_data = inverse_transform.transform(target_space, unflattened_data)
    for key in flat_data.keys():
        assert backend.all(recovered_flat_data[key] == flat_data[key])


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_flatten_unflatten_serialization(backend, seed):
    """Test FlattenDictTransformation and UnflattenDictTransformation serialization."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)

    nested_space, rng = make_nested_dict_space(backend, rng, np_rng)
    flatten_transform = FlattenDictTransformation(nested_separator=".")
    flat_space = flatten_transform.get_target_space_from_source(nested_space)

    verify_transformation_serialization(flatten_transform, nested_space, rng, num_samples=3)
    verify_transformation_serialization(
        UnflattenDictTransformation(nested_separator="."),
        flat_space,
        rng,
        num_samples=3,
    )
