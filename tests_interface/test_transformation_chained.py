"""Tests for ChainedTransformation."""
import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.transformations import (
    ChainedTransformation,
    RescaleTransformation,
    CropTransformation,
    IdentityTransformation
)
from test_utils import make_random_box_space, verify_transformation_serialization


ALL_BACKENDS = [NumpyComputeBackend, JaxComputeBackend, PyTorchComputeBackend]
SEEDS = [0, 42, 123]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_chained_two_transformations(backend, seed):
    """Test chaining two transformations."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create chained transformation: rescale then crop
    transform = ChainedTransformation([
        RescaleTransformation(new_low=0.0, new_high=2.0),
        CropTransformation(crop_low=0.0, crop_high=1.0)
    ])
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space is not None
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Verify result is cropped to [0, 1]
    flat_transformed = space.backend.reshape(transformed, (-1,))
    assert space.backend.all(flat_transformed >= 0.0)
    assert space.backend.all(flat_transformed <= 1.0)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_chained_with_identity(backend, seed):
    """Test that chaining with identity doesn't change behavior."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create rescale transformation
    rescale = RescaleTransformation(new_low=0.0, new_high=1.0)
    
    # Create chained with identity
    chained = ChainedTransformation([rescale, IdentityTransformation()])
    
    # Both should produce same result
    rng, data = space.sample(rng)
    
    single_result = rescale.transform(space, data)
    chained_result = chained.transform(space, data)
    
    flat_single = space.backend.reshape(single_result, (-1,))
    flat_chained = space.backend.reshape(chained_result, (-1,))
    
    assert space.backend.all(flat_single == flat_chained)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_chained_inverse(backend, seed):
    """Test that chained inverse works correctly."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create chained transformation with invertible operations
    transform = ChainedTransformation([
        RescaleTransformation(new_low=-1.0, new_high=1.0),
        CropTransformation(crop_low=-0.5, crop_high=0.5)
    ])
    
    # Get inverse
    inv_transform = transform.direction_inverse(space)
    assert inv_transform is not None


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_chained_has_inverse(backend, seed):
    """Test that chained has_inverse is computed correctly."""
    # All invertible
    all_invertible = ChainedTransformation([
        RescaleTransformation(new_low=-1.0, new_high=1.0),
        RescaleTransformation(new_low=0.0, new_high=1.0)
    ])
    assert all_invertible.has_inverse
    
    # With non-invertible (crop's inverse is identity, which is invertible)
    with_crop = ChainedTransformation([
        RescaleTransformation(new_low=-1.0, new_high=1.0),
        CropTransformation(crop_low=0.0, crop_high=1.0)
    ])
    # Crop has inverse (returns identity), so chain should have inverse
    assert with_crop.has_inverse


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_chained_serialization(backend, seed):
    """Test chained transformation serialization."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    transform = ChainedTransformation([
        RescaleTransformation(new_low=-1.0, new_high=1.0),
        CropTransformation(crop_low=0.0, crop_high=1.0)
    ])
    
    verify_transformation_serialization(transform, space, rng, num_samples=3)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_empty_chained(backend, seed):
    """Test empty chained transformation (should behave like identity)."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create empty chained transformation
    transform = ChainedTransformation([])
    
    # Should have no effect
    target_space = transform.get_target_space_from_source(space)
    assert target_space == space
    
    # Transform should return data unchanged
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    flat_original = space.backend.reshape(data, (-1,))
    flat_transformed = space.backend.reshape(transformed, (-1,))
    assert space.backend.all(flat_original == flat_transformed)
