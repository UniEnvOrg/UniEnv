"""Test utilities for generating random spaces and testing transformations."""
import typing
import numpy as np
from unienv_interface.backends import ComputeBackend
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import Space, BoxSpace, DictSpace
from unienv_interface.transformations.transformation import DataTransformation
from unienv_interface.space.space_utils import flatten_data


def make_random_box_space(
    backend_class,
    device: typing.Optional[typing.Any],
    rng: typing.Any,
    np_rng: np.random.Generator,
    ndims_range: typing.Tuple[int, int] = (1, 5),
    shape_range: typing.Tuple[int, int] = (1, 10),
    allow_unbounded: bool = True,
) -> Space:
    """Create a random BoxSpace for testing.
    
    Args:
        backend_class: The backend class to use (not an instance)
        device: The device to use
        rng: The backend random number generator
        np_rng: NumPy random generator for shape/parameters
        ndims_range: Range of dimensions for the shape
        shape_range: Range for each dimension size
        allow_unbounded: Whether to allow unbounded (inf) bounds
    
    Returns:
        A randomly generated BoxSpace
    """
    shape_ndims = int(np_rng.integers(ndims_range[0], ndims_range[1] + 1))
    shape = tuple([int(np_rng.integers(shape_range[0], shape_range[1] + 1)) for _ in range(shape_ndims)])
    dtype = np_rng.choice([backend_class.default_floating_dtype, backend_class.default_integer_dtype])
    
    # Generate random bounds
    rng, values_1 = backend_class.random.random_exponential(
        shape, lambd=0.5, rng=rng, dtype=dtype, device=device
    ) if backend_class.dtype_is_real_floating(dtype) else backend_class.random.random_discrete_uniform(
        shape, 0, 127, rng=rng, dtype=dtype, device=device
    )
    rng, values_2 = backend_class.random.random_exponential(
        shape, lambd=0.5, rng=rng, dtype=dtype, device=device
    ) if backend_class.dtype_is_real_floating(dtype) else backend_class.random.random_discrete_uniform(
        shape, 0, 127, rng=rng, dtype=dtype, device=device
    )

    min_values = backend_class.minimum(values_1, values_2)
    max_values = backend_class.maximum(values_1, values_2)
    
    # Make some dimensions unbounded if allowed
    if allow_unbounded and backend_class.dtype_is_real_floating(dtype):
        rng, idx_min_unbound = backend_class.random.random_uniform(shape, rng=rng, low=0.0, high=1.0, device=device)
        rng, idx_max_unbound = backend_class.random.random_uniform(shape, rng=rng, low=0.0, high=1.0, device=device)
        idx_min_unbound = idx_min_unbound <= 0.2
        idx_max_unbound = idx_max_unbound <= 0.2
        min_values = backend_class.at(min_values)[idx_min_unbound].set(-backend_class.inf)
        max_values = backend_class.at(max_values)[idx_max_unbound].set(backend_class.inf)
    
    if backend_class.dtype_is_real_integer(dtype):
        min_values = backend_class.round(min_values)
        max_values = backend_class.round(max_values)
    
    return BoxSpace(backend_class, min_values, max_values, dtype=dtype, device=device), rng


def make_random_dict_space(
    backend_class,
    device: typing.Optional[typing.Any],
    rng: typing.Any,
    np_rng: np.random.Generator,
    num_keys_range: typing.Tuple[int, int] = (1, 5),
    nested: bool = False,
) -> typing.Tuple[Space, typing.Any]:
    """Create a random DictSpace for testing.
    
    Args:
        backend_class: The backend class to use (not an instance)
        device: The device to use
        rng: The backend random number generator
        np_rng: NumPy random generator
        num_keys_range: Range for number of keys in the dict
        nested: Whether to create nested dict spaces
    
    Returns:
        A tuple of (randomly generated DictSpace, updated rng)
    """
    num_keys = int(np_rng.integers(num_keys_range[0], num_keys_range[1] + 1))
    spaces = {}
    
    for i in range(num_keys):
        key = f"key_{i}"
        if nested and np_rng.random() < 0.3 and i > 0:
            # Create a nested dict
            nested_space, rng = make_random_dict_space(
                backend_class, device, rng, np_rng, 
                num_keys_range=(1, 3), 
                nested=False
            )
            spaces[key] = nested_space
        else:
            # Create a box space
            box_space, rng = make_random_box_space(backend_class, device, rng, np_rng)
            spaces[key] = box_space
    
    return DictSpace(backend_class, spaces, device=device), rng


def sample_and_verify_transformation(
    transformation: DataTransformation,
    source_space: Space,
    rng: typing.Any,
    num_samples: int = 20,
) -> None:
    """Test a transformation by sampling data and verifying transform/inverse.
    
    Args:
        transformation: The transformation to test
        source_space: The source space to transform from
        rng: Random number generator
        num_samples: Number of samples to test
    """
    # Get target space
    target_space = transformation.get_target_space_from_source(source_space)
    assert target_space is not None, "Transformation should return a target space"
    
    # Test forward and inverse transforms
    for _ in range(num_samples):
        rng, data = source_space.sample(rng)
        
        # Forward transform
        transformed = transformation.transform(source_space, data)
        assert target_space.contains(transformed), "Transformed data should be in target space"
        
        # Test inverse if available
        if transformation.has_inverse:
            inv_transform = transformation.direction_inverse(source_space)
            if inv_transform is not None:
                recovered = inv_transform.transform(target_space, transformed)
                # Check that recovered data is approximately equal to original
                flat_original = source_space.backend.reshape(data, (-1,))
                flat_recovered = source_space.backend.reshape(recovered, (-1,))
                # Use allclose for floating point comparison
                if source_space.backend.dtype_is_real_floating(source_space.dtype):
                    assert source_space.backend.all(
                        source_space.backend.abs(flat_original - flat_recovered) < 1e-5
                    ), "Inverse transform should recover original data"


def verify_transformation_serialization(
    transformation: DataTransformation,
    source_space: Space,
    rng: typing.Any,
    num_samples: int = 10,
) -> None:
    """Verify that a transformation serializes and deserializes correctly.
    
    Args:
        transformation: The transformation to test
        source_space: The source space to use for testing
        rng: Random number generator
        num_samples: Number of samples to test after deserialization
    """
    from unienv_interface.transformations import json_to_transformation, transformation_to_json
    
    # Raw class serialization should not include type metadata.
    serialized_without_type = transformation.serialize(source_space=source_space)
    assert "type" not in serialized_without_type, "Class serialize() should not include 'type' field"

    # Serialize
    json_data = transformation_to_json(transformation, source_space=source_space)
    assert "type" in json_data, "Serialized data should contain 'type' field"
    
    # Deserialize
    restored = json_to_transformation(json_data, source_space=source_space)
    assert type(restored) == type(transformation), f"Deserialized type should match: {type(restored)} vs {type(transformation)}"
    
    # Test that the restored transformation works the same
    target_space_original = transformation.get_target_space_from_source(source_space)
    target_space_restored = restored.get_target_space_from_source(source_space)
    
    # Verify target spaces are equivalent (checking shape and bounds)
    assert target_space_original.shape == target_space_restored.shape, "Target spaces should have same shape"
    
    # Test transforms produce same results
    for _ in range(num_samples):
        rng, data = source_space.sample(rng)
        original_transformed = transformation.transform(source_space, data)
        restored_transformed = restored.transform(source_space, data)
        
        # Use flatten_data to handle both array and dict/tuple data
        target_space_original = transformation.get_target_space_from_source(source_space)
        target_space_restored = restored.get_target_space_from_source(source_space)
        
        flat_original = flatten_data(target_space_original, original_transformed, start_dim=0)
        flat_restored = flatten_data(target_space_restored, restored_transformed, start_dim=0)
        
        # Determine dtype for comparison - use source space's dtype
        compare_dtype = source_space.dtype
        
        if source_space.backend.dtype_is_real_floating(compare_dtype):
            assert source_space.backend.all(
                source_space.backend.abs(flat_original - flat_restored) < 1e-5
            ), "Restored transformation should produce same results"
        else:
            assert source_space.backend.all(flat_original == flat_restored), "Restored transformation should produce same results"
