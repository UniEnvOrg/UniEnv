"""Tests for Space utilities."""
import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu


ALL_BACKENDS = [NumpyComputeBackend, JaxComputeBackend, PyTorchComputeBackend]
SEEDS = [0, 42, 123]


def make_box_space(backend, shape, dtype=None):
    """Helper to create a box space."""
    if dtype is None:
        dtype = backend.default_floating_dtype
    return BoxSpace(
        backend,
        low=0.0,
        high=1.0,
        dtype=dtype,
        device=None,
        shape=shape
    )


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_batch_space(backend, seed):
    """Test batching a space."""
    np_rng = np.random.default_rng(seed)
    
    # Create a box space
    space = make_box_space(backend, (3, 4))
    
    # Batch with size 10
    batched = sbu.batch_space(space, 10)
    
    # Batched space should have shape (10, 3, 4)
    assert batched.shape == (10, 3, 4)
    assert sbu.batch_size(batched) == 10


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_unbatch_spaces(backend, seed):
    """Test unbatching a space."""
    np_rng = np.random.default_rng(seed)
    
    # Create a box space
    space = make_box_space(backend, (3, 4))
    
    # Batch with size 5
    batched = sbu.batch_space(space, 5)
    
    # Unbatch
    unbatched = list(sbu.unbatch_spaces(batched))
    
    # Should get 5 identical spaces
    assert len(unbatched) == 5
    for s in unbatched:
        assert s.shape == (3, 4)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_batch_data(backend, seed):
    """Test batching data."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space = make_box_space(backend, (3, 4))
    
    # Sample multiple instances
    num_samples = 5
    samples = []
    for _ in range(num_samples):
        rng, data = space.sample(rng)
        samples.append(data)
    
    # Batch the data
    batched = sbu.stack(backend, samples)
    assert batched.shape == (num_samples, 3, 4)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_flatten_unflatten_roundtrip(backend, seed, batch_size):
    """Test that flatten and unflatten are inverses."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space = make_box_space(backend, (3, 4, 5))
    
    # Sample data
    if batch_size == 1:
        rng, data = space.sample(rng)
    else:
        batched_space = sbu.batch_space(space, batch_size)
        rng, data = batched_space.sample(rng)
        space = batched_space
    
    # Flatten
    flat = sfu.flatten_data(space, data)
    
    # Check flattened shape
    expected_flat_dim = 3 * 4 * 5
    if batch_size > 1:
        expected_flat_dim *= batch_size
    assert flat.shape[-1] == expected_flat_dim
    
    # Unflatten
    if batch_size == 1:
        unflat = sfu.unflatten_data(space, flat)
    else:
        unflat = sfu.unflatten_data(space, flat, start_dim=1)
    
    # Verify round-trip
    assert space.backend.all(space.backend.reshape(data, (-1,)) == space.backend.reshape(unflat, (-1,)))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_concatenate_spaces(backend, seed):
    """Test concatenating data from spaces."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space = make_box_space(backend, (3, 4))
    
    # Sample two instances
    rng, data1 = space.sample(rng)
    rng, data2 = space.sample(rng)
    
    # Concatenate
    concat = sbu.concatenate(space, [data1, data2], axis=0)
    assert concat.shape == (2, 3, 4)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_get_at_batch(backend, seed):
    """Test getting items from batched data."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a box space
    space = make_box_space(backend, (3, 4))
    batched_space = sbu.batch_space(space, 10)
    
    # Sample batched data
    rng, data = batched_space.sample(rng)
    
    # Get first element
    first = sbu.get_at(batched_space, data, 0)
    assert first.shape == (3, 4)
    
    # Get slice
    slice_data = sbu.get_at(batched_space, data, slice(0, 3))
    assert slice_data.shape == (3, 3, 4)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_dict_space_batch_operations(backend, seed):
    """Test batch operations on dict spaces."""
    np_rng = np.random.default_rng(seed)
    
    # Create dict space
    box1 = make_box_space(backend, (3,))
    box2 = make_box_space(backend, (4, 5))
    space = DictSpace(backend, {"a": box1, "b": box2}, device=None)
    
    # Batch the space
    batched = sbu.batch_space(space, 5)
    
    # Check batch size
    assert sbu.batch_size(batched) == 5
    
    # Check each subspace is batched
    assert batched["a"].shape == (5, 3)
    assert batched["b"].shape == (5, 4, 5)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_space_equality(backend, seed):
    """Test space equality."""
    np_rng = np.random.default_rng(seed)
    
    # Create identical spaces
    space1 = make_box_space(backend, (3, 4))
    space2 = make_box_space(backend, (3, 4))
    
    assert space1 == space2
    
    # Create different spaces
    space3 = make_box_space(backend, (3, 5))
    assert space1 != space3


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_space_contains(backend, seed):
    """Test space.contains method."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create bounded box space
    space = BoxSpace(
        backend,
        low=0.0,
        high=1.0,
        dtype=backend.default_floating_dtype,
        device=None,
        shape=(3, 4)
    )
    
    # Sample should be in space
    rng, data = space.sample(rng)
    assert space.contains(data)
    
    # Data outside bounds should not be in space
    out_of_bounds = backend.full((3, 4), 2.0, dtype=space.dtype, device=space.device)
    assert not space.contains(out_of_bounds)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_space_sample_deterministic(backend, seed):
    """Test that sampling with same rng produces deterministic results."""
    np_rng = np.random.default_rng(seed)
    rng1 = backend.random.random_number_generator(seed)
    rng2 = backend.random.random_number_generator(seed)
    
    # Create space
    space = make_box_space(backend, (3, 4))
    
    # Sample with same rng
    rng1, data1 = space.sample(rng1)
    rng2, data2 = space.sample(rng2)
    
    # Should be identical
    assert space.backend.all(space.backend.reshape(data1, (-1,)) == space.backend.reshape(data2, (-1,)))
