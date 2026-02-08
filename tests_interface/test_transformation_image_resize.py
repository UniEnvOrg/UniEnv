"""Tests for ImageResizeTransformation."""
import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace
from unienv_interface.transformations import ImageResizeTransformation
from test_utils import verify_transformation_serialization


ALL_BACKENDS = [NumpyComputeBackend, JaxComputeBackend, PyTorchComputeBackend]
SEEDS = [0, 42, 123]


def make_image_space(backend, device, height=64, width=64, channels=3):
    """Create a BoxSpace representing an image."""
    shape = (height, width, channels)
    return BoxSpace(
        backend,
        low=0.0,
        high=255.0,
        dtype=backend.default_floating_dtype,
        device=device,
        shape=shape
    )


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_image_resize_downscale(backend, seed):
    """Test resizing images to smaller dimensions."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create image space
    space = make_image_space(backend, None, height=64, width=64, channels=3)
    
    # Create resize transformation to downscale
    transform = ImageResizeTransformation(new_height=32, new_width=32)
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space.shape == (32, 32, 3), f"Expected (32, 32, 3), got {target_space.shape}"
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    assert transformed.shape[-3:] == (32, 32, 3), f"Transformed shape should be (..., 32, 32, 3)"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_image_resize_upscale(backend, seed):
    """Test resizing images to larger dimensions."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create image space
    space = make_image_space(backend, None, height=32, width=32, channels=3)
    
    # Create resize transformation to upscale
    transform = ImageResizeTransformation(new_height=64, new_width=64)
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space.shape == (64, 64, 3), f"Expected (64, 64, 3), got {target_space.shape}"
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    assert transformed.shape[-3:] == (64, 64, 3), f"Transformed shape should be (..., 64, 64, 3)"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_image_resize_preserves_channels(backend, seed):
    """Test that image resize preserves the channel dimension."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create image space with different channel counts
    for channels in [1, 3, 4]:
        space = make_image_space(backend, None, height=32, width=32, channels=channels)
        
        transform = ImageResizeTransformation(new_height=64, new_width=64)
        target_space = transform.get_target_space_from_source(space)
        
        assert target_space.shape[-1] == channels, f"Channel count should be preserved: {channels}"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_image_resize_inverse(backend, seed):
    """Test that image resize inverse works."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create image space
    space = make_image_space(backend, None, height=64, width=64, channels=3)
    
    # Create resize transformation
    transform = ImageResizeTransformation(new_height=32, new_width=32)
    
    # Get inverse
    inv_transform = transform.direction_inverse(space)
    assert inv_transform is not None
    
    # Check that inverse has correct dimensions
    assert inv_transform.new_height == 64
    assert inv_transform.new_width == 64


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_image_resize_serialization(backend, seed):
    """Test image resize transformation serialization."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    space = make_image_space(backend, None, height=64, width=64, channels=3)
    transform = ImageResizeTransformation(new_height=32, new_width=32)
    
    verify_transformation_serialization(transform, space, rng, num_samples=3)
