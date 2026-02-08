"""Tests for RescaleTransformation."""
import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.transformations import RescaleTransformation
from .test_utils import make_random_box_space, verify_transformation_serialization


ALL_BACKENDS = [NumpyComputeBackend, JaxComputeBackend, PyTorchComputeBackend]
SEEDS = [0, 42, 123]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rescale_to_unit_range(backend, seed):
    """Test rescaling to [0, 1] range."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a bounded box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create rescale transformation to [0, 1]
    transform = RescaleTransformation(new_low=0.0, new_high=1.0)
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space is not None
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Verify transformed data is in [0, 1] range
    flat_transformed = space.backend.reshape(transformed, (-1,))
    assert space.backend.all(flat_transformed >= 0.0), "Transformed data should be >= 0"
    assert space.backend.all(flat_transformed <= 1.0), "Transformed data should be <= 1"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rescale_to_custom_range(backend, seed):
    """Test rescaling to custom range."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a bounded box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create rescale transformation to [-5, 5]
    transform = RescaleTransformation(new_low=-5.0, new_high=5.0)
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Verify transformed data is in [-5, 5] range
    flat_transformed = space.backend.reshape(transformed, (-1,))
    assert space.backend.all(flat_transformed >= -5.0), "Transformed data should be >= -5"
    assert space.backend.all(flat_transformed <= 5.0), "Transformed data should be <= 5"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rescale_inverse(backend, seed):
    """Test that rescale inverse works correctly."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a bounded box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create rescale transformation
    transform = RescaleTransformation(new_low=-1.0, new_high=1.0)
    
    # Get inverse
    inv_transform = transform.direction_inverse(space)
    assert inv_transform is not None, "Rescale should have inverse"
    
    # Sample and round-trip
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    target_space = transform.get_target_space_from_source(space)
    recovered = inv_transform.transform(target_space, transformed)
    
    # Verify recovery (approximately, due to floating point)
    flat_original = space.backend.reshape(data, (-1,))
    flat_recovered = space.backend.reshape(recovered, (-1,))
    assert space.backend.all(
        space.backend.abs(flat_original - flat_recovered) < 1e-5
    ), "Inverse should recover original data"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rescale_with_array_bounds(backend, seed):
    """Test rescaling with per-element bounds."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.number_generator(seed)
    
    # Create a bounded box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create array bounds
    shape = space.shape
    new_low = space.backend.full(shape, -2.0, dtype=space.dtype, device=space.device)
    new_high = space.backend.full(shape, 2.0, dtype=space.dtype, device=space.device)
    
    # Create rescale transformation with array bounds
    transform = RescaleTransformation(new_low=new_low, new_high=new_high)
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Verify all values are in range
    flat_transformed = space.backend.reshape(transformed, (-1,))
    assert space.backend.all(flat_transformed >= -2.0 - 1e-5)
    assert space.backend.all(flat_transformed <= 2.0 + 1e-5)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rescale_serialization(backend, seed):
    """Test that rescale transformation serializes correctly."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a bounded box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create rescale transformation with scalar bounds
    transform = RescaleTransformation(new_low=-1.0, new_high=1.0)
    
    # Verify serialization
    verify_transformation_serialization(transform, space, rng, num_samples=5)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rescale_with_nan_replacement(backend, seed):
    """Test rescaling with NaN replacement."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a bounded box space with float dtype
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Skip if not floating point
    if not space.backend.dtype_is_real_floating(space.dtype):
        pytest.skip("NaN replacement only works with floating point dtypes")
    
    # Create rescale transformation with NaN replacement
    transform = RescaleTransformation(new_low=0.0, new_high=1.0, nan_to=0.5)
    
    # Sample and inject NaN
    rng, data = space.sample(rng)
    data_with_nan = space.backend.at(data)[0, ...].set(space.backend.nan)
    
    # Transform
    transformed = transform.transform(space, data_with_nan)
    
    # Verify NaN was replaced
    flat_transformed = space.backend.reshape(transformed, (-1,))
    assert not space.backend.any(space.backend.isnan(flat_transformed)), "NaN should be replaced"
