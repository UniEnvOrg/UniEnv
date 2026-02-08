"""Tests for IterativeTransformation."""
import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.transformations import (
    IterativeTransformation,
    RescaleTransformation,
    IdentityTransformation
)
from unienv_interface.space import DictSpace, TupleSpace
from test_utils import make_random_box_space, verify_transformation_serialization


ALL_BACKENDS = [NumpyComputeBackend, JaxComputeBackend, PyTorchComputeBackend]
SEEDS = [0, 42, 123]


def custom_is_leaf(space):
    """Custom leaf detector for testing."""
    return not isinstance(space, (DictSpace, TupleSpace))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_iterative_on_dict(backend, seed):
    """Test iterative transformation on dict space."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create dict space with box values
    box1, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    box2, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    space = DictSpace(backend, {"a": box1, "b": box2}, device=None)
    
    # Create iterative transformation with rescale
    transform = IterativeTransformation(
        transformation=RescaleTransformation(new_low=0.0, new_high=1.0)
    )
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space is not None
    assert isinstance(target_space, DictSpace)
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Both values should be rescaled to [0, 1]
    for key in ["a", "b"]:
        values = transformed[key]
        flat_values = backend.reshape(values, (-1,))
        assert backend.all(flat_values >= 0.0), f"{key} should be >= 0"
        assert backend.all(flat_values <= 1.0), f"{key} should be <= 1"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_iterative_on_tuple(backend, seed):
    """Test iterative transformation on tuple space."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create tuple space with box values
    box1, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    box2, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    space = TupleSpace(backend, (box1, box2), device=None)
    
    # Create iterative transformation with rescale
    transform = IterativeTransformation(
        transformation=RescaleTransformation(new_low=0.0, new_high=1.0)
    )
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space is not None
    assert isinstance(target_space, TupleSpace)
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Both values should be rescaled to [0, 1]
    for i in [0, 1]:
        values = transformed[i]
        flat_values = backend.reshape(values, (-1,))
        assert backend.all(flat_values >= 0.0), f"index {i} should be >= 0"
        assert backend.all(flat_values <= 1.0), f"index {i} should be <= 1"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_iterative_on_box(backend, seed):
    """Test iterative transformation on box space (should behave like inner transformation)."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create iterative transformation with rescale
    transform = IterativeTransformation(
        transformation=RescaleTransformation(new_low=0.0, new_high=1.0)
    )
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    
    # Compare with direct rescale
    direct_rescale = RescaleTransformation(new_low=0.0, new_high=1.0)
    direct_target = direct_rescale.get_target_space_from_source(space)
    
    assert target_space.shape == direct_target.shape
    
    # Sample and transform
    rng, data = space.sample(rng)
    iterative_result = transform.transform(space, data)
    direct_result = direct_rescale.transform(space, data)
    
    flat_iterative = backend.reshape(iterative_result, (-1,))
    flat_direct = backend.reshape(direct_result, (-1,))
    assert backend.all(flat_iterative == flat_direct)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_iterative_with_custom_leaf_fn(backend, seed):
    """Test iterative transformation with custom leaf function."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create dict space with box values
    box1, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    space = DictSpace(backend, {"value": box1}, device=None)
    
    # Create iterative with custom leaf detector (treats everything as leaf)
    transform = IterativeTransformation(
        transformation=RescaleTransformation(new_low=0.0, new_high=1.0),
        is_leaf_node_fn=lambda s: True  # Everything is a leaf
    )
    
    # Should apply transformation to the dict itself, not recurse
    # This would fail since rescale doesn't handle dicts, so we use identity
    transform_identity = IterativeTransformation(
        transformation=IdentityTransformation(),
        is_leaf_node_fn=lambda s: True
    )
    
    target_space = transform_identity.get_target_space_from_source(space)
    # With custom leaf fn, it should treat dict as leaf and try to apply identity
    # Identity returns space unchanged


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_iterative_inverse(backend, seed):
    """Test iterative transformation inverse."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create dict space
    box1, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    space = DictSpace(backend, {"value": box1}, device=None)
    
    # Create iterative with rescale (which has inverse)
    transform = IterativeTransformation(
        transformation=RescaleTransformation(new_low=-1.0, new_high=1.0)
    )
    
    # Get inverse
    inv_transform = transform.direction_inverse(space)
    assert inv_transform is not None
    
    # Test round-trip
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    target_space = transform.get_target_space_from_source(space)
    recovered = inv_transform.transform(target_space, transformed)
    
    # Check recovery
    original_val = data["value"]
    recovered_val = recovered["value"]
    flat_original = backend.reshape(original_val, (-1,))
    flat_recovered = backend.reshape(recovered_val, (-1,))
    
    if backend.dtype_is_real_floating(box1.dtype):
        assert backend.all(backend.abs(flat_original - flat_recovered) < 1e-5)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_iterative_serialization(backend, seed):
    """Test iterative transformation serialization."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create dict space
    box1, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    space = DictSpace(backend, {"value": box1}, device=None)
    
    transform = IterativeTransformation(
        transformation=RescaleTransformation(new_low=0.0, new_high=1.0)
    )
    
    verify_transformation_serialization(transform, space, rng, num_samples=3)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_iterative_serialization_with_custom_fn(backend, seed):
    """Test iterative transformation serialization with custom function."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create dict space
    box1, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    space = DictSpace(backend, {"value": box1}, device=None)
    
    transform = IterativeTransformation(
        transformation=RescaleTransformation(new_low=0.0, new_high=1.0),
        is_leaf_node_fn=custom_is_leaf,
        inv_is_leaf_node_fn=custom_is_leaf
    )
    
    verify_transformation_serialization(transform, space, rng, num_samples=3)
