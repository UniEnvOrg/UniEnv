"""Tests for CropTransformation."""
import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.transformations import CropTransformation
from test_utils import make_random_box_space, verify_transformation_serialization


ALL_BACKENDS = [NumpyComputeBackend, JaxComputeBackend, PyTorchComputeBackend]
SEEDS = [0, 42, 123]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_crop_with_scalar_bounds(backend, seed):
    """Test cropping with scalar bounds."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space with values in known range
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create crop transformation with tighter scalar bounds
    transform = CropTransformation(crop_low=-0.5, crop_high=0.5)
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space is not None
    
    # Sample and transform - values should be clipped
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Verify all values are within crop bounds
    flat_transformed = space.backend.reshape(transformed, (-1,))
    assert space.backend.all(flat_transformed >= -0.5), "Cropped values should be >= -0.5"
    assert space.backend.all(flat_transformed <= 0.5), "Cropped values should be <= 0.5"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_crop_with_array_bounds(backend, seed):
    """Test cropping with per-element array bounds."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create array bounds matching the space shape
    shape = space.shape
    crop_low = space.backend.full(shape, -0.3, dtype=space.dtype, device=space.device)
    crop_high = space.backend.full(shape, 0.3, dtype=space.dtype, device=space.device)
    
    # Create crop transformation
    transform = CropTransformation(crop_low=crop_low, crop_high=crop_high)
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space is not None
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Verify bounds
    flat_transformed = space.backend.reshape(transformed, (-1,))
    assert space.backend.all(flat_transformed >= -0.3), "Cropped values should be >= -0.3"
    assert space.backend.all(flat_transformed <= 0.3), "Cropped values should be <= 0.3"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_crop_inverse(backend, seed):
    """Test that crop inverse returns identity transformation."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create crop transformation
    transform = CropTransformation(crop_low=-1.0, crop_high=1.0)
    
    # Get inverse
    inv_transform = transform.direction_inverse(space)
    assert inv_transform is not None
    
    # Crop inverse should return IdentityTransformation
    from unienv_interface.transformations import IdentityTransformation
    assert isinstance(inv_transform, IdentityTransformation)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_crop_doesnt_affect_in_range_values(backend, seed):
    """Test that values already in range are not affected."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space with tight bounds
    space, rng = make_random_box_space(
        backend, None, rng, np_rng, 
        ndims_range=(2, 3),
        allow_unbounded=False
    )
    
    # Create crop with very wide bounds (wider than space bounds)
    transform = CropTransformation(crop_low=-1000.0, crop_high=1000.0)
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Values should be unchanged
    flat_original = space.backend.reshape(data, (-1,))
    flat_transformed = space.backend.reshape(transformed, (-1,))
    assert space.backend.all(flat_original == flat_transformed), "In-range values should not change"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_crop_serialization(backend, seed):
    """Test crop transformation serialization."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    transform = CropTransformation(crop_low=-0.5, crop_high=0.5)
    
    verify_transformation_serialization(transform, space, rng, num_samples=5)
