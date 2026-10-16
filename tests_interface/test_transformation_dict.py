"""Tests for DictTransformation, DictIncludeKeyTransformation, DictExcludeKeyTransformation."""
import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.jax import JaxComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.transformations import (
    DictTransformation, 
    DictIncludeKeyTransformation, 
    DictExcludeKeyTransformation,
    RescaleTransformation
)
from unienv_interface.space import DictSpace
from test_utils import make_random_box_space, make_random_dict_space, verify_transformation_serialization


ALL_BACKENDS = [NumpyComputeBackend, JaxComputeBackend, PyTorchComputeBackend]
SEEDS = [0, 42, 123]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_dict_include_keys(backend, seed):
    """Test including specific keys from a dict."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a dict space
    space, rng = make_random_dict_space(backend, None, rng, np_rng, num_keys_range=(3, 5))
    
    # Get available keys
    available_keys = list(space.spaces.keys())
    
    # Select a subset of keys to include
    keys_to_include = available_keys[:2]
    
    # Create include transformation
    transform = DictIncludeKeyTransformation(enabled_keys=keys_to_include)
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space is not None
    
    # Target space should only have included keys
    assert set(target_space.spaces.keys()) == set(keys_to_include)
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Transformed data should only have included keys
    assert set(transformed.keys()) == set(keys_to_include)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_dict_exclude_keys(backend, seed):
    """Test excluding specific keys from a dict."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a dict space
    space, rng = make_random_dict_space(backend, None, rng, np_rng, num_keys_range=(3, 5))
    
    # Get available keys
    available_keys = list(space.spaces.keys())
    
    # Select keys to exclude
    keys_to_exclude = available_keys[:1]
    
    # Create exclude transformation
    transform = DictExcludeKeyTransformation(excluded_keys=keys_to_exclude)
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space is not None
    
    # Target space should not have excluded keys
    for key in keys_to_exclude:
        assert key not in target_space.spaces


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_dict_transformation(backend, seed):
    """Test applying transformations to dict values."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create a dict space with box values
    num_boxes = 3
    spaces = {}
    for i in range(num_boxes):
        box_space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
        spaces[f"box_{i}"] = box_space
    
    space = DictSpace(backend, spaces, device=None)
    
    # Create dict transformation applying rescale to each value
    mapping = {}
    for i in range(num_boxes):
        mapping[f"box_{i}"] = RescaleTransformation(new_low=0.0, new_high=1.0)
    
    transform = DictTransformation(mapping=mapping)
    
    # Get target space
    target_space = transform.get_target_space_from_source(space)
    assert target_space is not None
    
    # Sample and transform
    rng, data = space.sample(rng)
    transformed = transform.transform(space, data)
    
    # Each transformed value should be in [0, 1] range
    for key in transformed.keys():
        values = transformed[key]
        flat_values = backend.reshape(values, (-1,))
        assert backend.all(flat_values >= 0.0), f"{key} should be >= 0"
        assert backend.all(flat_values <= 1.0), f"{key} should be <= 1"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_dict_include_nested(backend, seed):
    """Test including nested keys using separator."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create nested dict space
    inner_space, rng = make_random_dict_space(backend, None, rng, np_rng, num_keys_range=(2, 3))
    outer_space = DictSpace(backend, {"outer_key": inner_space}, device=None)
    
    # Get nested key path
    inner_keys = list(inner_space.spaces.keys())
    nested_key = f"outer_key/{inner_keys[0]}"
    
    # Create include transformation
    transform = DictIncludeKeyTransformation(enabled_keys=[nested_key], nested_separator="/")
    
    # Get target space
    target_space = transform.get_target_space_from_source(outer_space)
    assert target_space is not None
    
    # Target space should have the nested structure
    assert "outer_key" in target_space.spaces


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_dict_include_serialization(backend, seed):
    """Test DictIncludeKeyTransformation serialization."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    space, rng = make_random_dict_space(backend, None, rng, np_rng, num_keys_range=(3, 5))
    available_keys = list(space.spaces.keys())
    
    transform = DictIncludeKeyTransformation(enabled_keys=available_keys[:2])
    verify_transformation_serialization(transform, space, rng, num_samples=3)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_dict_exclude_serialization(backend, seed):
    """Test DictExcludeKeyTransformation serialization."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    space, rng = make_random_dict_space(backend, None, rng, np_rng, num_keys_range=(3, 5))
    available_keys = list(space.spaces.keys())
    
    transform = DictExcludeKeyTransformation(excluded_keys=available_keys[:1])
    verify_transformation_serialization(transform, space, rng, num_samples=3)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_dict_transformation_serialization(backend, seed):
    """Test DictTransformation serialization."""
    np_rng = np.random.default_rng(seed)
    rng = backend.random.random_number_generator(seed)
    
    # Create dict space
    box_space, rng = make_random_box_space(backend, None, rng, np_rng, allow_unbounded=False)
    space = DictSpace(backend, {"value": box_space}, device=None)
    
    # Create dict transformation
    transform = DictTransformation(mapping={"value": RescaleTransformation(new_low=0.0, new_high=1.0)})
    
    verify_transformation_serialization(transform, space, rng, num_samples=3)


def _nested_space():
    from unienv_interface.space import BoxSpace
    box = lambda lo, hi: BoxSpace(NumpyComputeBackend, lo, hi, np.float32, shape=(2,))
    return DictSpace(NumpyComputeBackend, {"a": {"x": box(-2.0, 2.0), "y": box(0.0, 4.0)}, "b": box(5.0, 6.0)})


def test_dict_transformation_nested_keys_and_inverse():
    from unienv_interface.transformations import RescaleTransformation, transformation_to_json, json_to_transformation
    space = _nested_space()
    t = DictTransformation({"a/x": RescaleTransformation(0.0, 1.0), "b": RescaleTransformation(-1.0, 1.0)})
    target = t.get_target_space_from_source(space)
    assert float(target["a"]["x"].low.min()) == 0.0 and target["a"]["y"] == space["a"]["y"]
    _, x = space.sample(NumpyComputeBackend.random.random_number_generator(0))
    y = t.transform(space, x)
    assert target.contains(y)
    inverse = t.direction_inverse(space)
    back = inverse.transform(target, y)
    np.testing.assert_allclose(back["a"]["x"], x["a"]["x"], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(back["a"]["y"], x["a"]["y"])
    restored = json_to_transformation(transformation_to_json(t, space), space)
    np.testing.assert_allclose(restored.transform(space, x)["b"], y["b"])


def test_dict_transformation_missing_keys():
    from unienv_interface.transformations import IdentityTransformation, RescaleTransformation
    space = _nested_space()
    with pytest.raises(KeyError):
        DictTransformation({"missing": IdentityTransformation()}).get_target_space_from_source(space)
    with pytest.raises(KeyError):
        DictTransformation({"a/missing": IdentityTransformation()}).get_target_space_from_source(space)
    with pytest.raises(ValueError):
        DictTransformation({"a": IdentityTransformation()}).get_target_space_from_source(space["b"])
    t = DictTransformation({"missing": RescaleTransformation(0.0, 1.0), "b": RescaleTransformation(0.0, 1.0)},
                           ignore_missing_keys=True)
    target = t.get_target_space_from_source(space)
    _, x = space.sample(NumpyComputeBackend.random.random_number_generator(1))
    assert target.contains(t.transform(space, x))
    inverse = t.direction_inverse(space)
    assert set(inverse.mapping) == {"b"}
    no_inverse = DictTransformation({"b": DictIncludeKeyTransformation(["q"])})
    assert not no_inverse.has_inverse and no_inverse.direction_inverse(space) is None
