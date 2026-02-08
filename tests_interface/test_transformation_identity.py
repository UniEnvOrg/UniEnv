"""Tests for IdentityTransformation."""
import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.transformations import IdentityTransformation
from .test_utils import make_random_box_space, sample_and_verify_transformation, verify_transformation_serialization


ALL_BACKENDS = [NumpyComputeBackend, JaxComputeBackend, PyTorchComputeBackend]
SEEDS = [0, 42, 123]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_identity_transformation(backend, seed):
    """Test that identity transformation works correctly."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a random box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create identity transformation
    transform = IdentityTransformation()
    
    # Verify transformation behavior
    sample_and_verify_transformation(transform, space, rng, num_samples=10)
    
    # Test that target space equals source space
    target_space = transform.get_target_space_from_source(space)
    assert target_space == space, "Identity transformation should preserve space"
    
    # Test that transform returns data unchanged
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Verify data is unchanged
    flat_original = space.backend.reshape(data, (-1,))
    flat_transformed = space.backend.reshape(transformed, (-1,))
    assert space.backend.all(flat_original == flat_transformed), "Identity should not change data"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_identity_serialization(backend, seed):
    """Test that identity transformation serializes and deserializes correctly."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a random box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create identity transformation
    transform = IdentityTransformation()
    
    # Verify serialization
    verify_transformation_serialization(transform, space, rng, num_samples=5)
