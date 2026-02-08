"""Tests for BatchifyTransformation and UnBatchifyTransformation."""
import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.transformations import BatchifyTransformation, UnBatchifyTransformation
from unienv_interface.space.space_utils import batch_utils as sbu
from .test_utils import make_random_box_space, verify_transformation_serialization


ALL_BACKENDS = [NumpyComputeBackend, JaxComputeBackend, PyTorchComputeBackend]
SEEDS = [0, 42, 123]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_batchify_transformation(backend, seed, axis):
    """Test batchifying data at different axes."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space with at least 3 dimensions
    space, rng = make_random_box_space(
        backend, None, rng, np_rng, 
        ndims_range=(3, 4), 
        allow_unbounded=False
    )
    
    # Create batchify transformation
    transform = BatchifyTransformation(axis=axis)
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space is not None
    
    # Target space should have batch dimension
    expected_batch_size = sbu.batch_size(target_space)
    assert expected_batch_size == 1, f"Batchified space should have batch_size=1, got {expected_batch_size}"
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Verify shape has extra dimension
    assert transformed.shape[axis] == 1, f"Axis {axis} should have size 1"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("axis", [0, 1])
def test_unbatchify_transformation(backend, seed, axis):
    """Test unbatchifying data at different axes."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space, rng = make_random_box_space(
        backend, None, rng, np_rng, 
        ndims_range=(2, 4), 
        allow_unbounded=False
    )
    
    # First batchify
    batchify = BatchifyTransformation(axis=axis)
    batched_space = batchify.get_target_space_from_source(space)
    
    # Then unbatchify
    unbatchify = UnBatchifyTransformation(axis=axis)
    unbatched_space = unbatchify.get_target_space_from_source(batched_space)
    
    # Spaces should be equal (ignoring batch dimension)
    assert unbatched_space.shape == space.shape, "Unbatchified space should match original"
    
    # Test round-trip
    rng, data = space.sample(rng)
    batched = batchify.transform(space, data)
    unbatched = unbatchify.transform(batched_space, batched)
    
    # Verify data is recovered
    flat_original = space.backend.reshape(data, (-1,))
    flat_unbatched = space.backend.reshape(unbatched, (-1,))
    assert space.backend.all(flat_original == flat_unbatched), "Round-trip should recover data"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_batchify_unbatchify_inverse(backend, seed):
    """Test that batchify and unbatchify are inverses."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    
    # Create transformations
    batchify = BatchifyTransformation(axis=0)
    unbatchify = UnBatchifyTransformation(axis=0)
    
    # Get inverse
    batchify_inv = batchify.direction_inverse(space)
    assert batchify_inv is not None
    assert isinstance(batchify_inv, UnBatchifyTransformation)
    
    # Test round-trip
    rng, data = space.sample(rng)
    batched = batchify.transform(space, data)
    target_space = batchify.get_target_space_from_source(space)
    unbatched = batchify_inv.transform(target_space, batched)
    
    flat_original = space.backend.reshape(data, (-1,))
    flat_unbatched = space.backend.reshape(unbatched, (-1,))
    assert space.backend.all(flat_original == flat_unbatched)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_batchify_serialization(backend, seed):
    """Test batchify transformation serialization."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    transform = BatchifyTransformation(axis=1)
    
    verify_transformation_serialization(transform, space, rng, num_samples=5)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_unbatchify_serialization(backend, seed):
    """Test unbatchify transformation serialization."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    transform = UnBatchifyTransformation(axis=0)
    
    verify_transformation_serialization(transform, space, rng, num_samples=5)
