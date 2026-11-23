"""Regression tests for space / backend / utils bugs found before the 1.0.0 release."""

import json

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.backends.serialization import deserialize_dtype, serialize_dtype
from unienv_interface.space import (
    BoxSpace,
    DictSpace,
    DynamicBoxSpace,
    GraphInstance,
    GraphSpace,
    TextSpace,
    TupleSpace,
)
from unienv_interface.space import batch_utils as sbu
from unienv_interface.space import flatten_utils as sfu
from unienv_interface.space.space_utils import serialization_utils as ssu
from tests_common.backends import cpu_backend_params

torch = pytest.importorskip("torch")
from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB  # noqa: E402

try:
    import jax
    from unienv_interface.backends.jax import JaxComputeBackend as JB
except ImportError:  # pragma: no cover
    jax = None
    JB = None

needs_jax = pytest.mark.skipif(jax is None, reason="jax not installed")
needs_jax_x32 = pytest.mark.skipif(
    jax is None or jax.config.jax_enable_x64, reason="needs jax with x64 disabled (its default)"
)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _box(*shape, dtype=np.float32, backend=NB, device=None):
    return BoxSpace(backend, 0.0, 1.0, dtype, shape=shape, device=device)


# --- 1. integer bounds compare exactly ------------------------------------------------

@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_integer_box_equality_is_exact(backend, device):
    dtype = backend.zeros((1,), dtype=backend.default_integer_dtype).dtype
    a = BoxSpace(backend, 0, 100000, dtype, shape=(2,), device=device)
    b = BoxSpace(backend, 0, 100001, dtype, shape=(2,), device=device)
    assert a != b
    assert a == BoxSpace(backend, 0, 100000, dtype, shape=(2,), device=device)
    assert DictSpace(backend, {"k": a}) != DictSpace(backend, {"k": b})


def test_float_box_equality_tolerates_rounding():
    a = BoxSpace(NB, 0.0, 1.0, np.float64, shape=(2,))
    b = BoxSpace(NB, 0.0, 1.0 + 1e-12, np.float64, shape=(2,))
    assert a == b


def test_integer_dynamic_box_equality_is_exact():
    a = DynamicBoxSpace(NB, 0, 100000, (0,), (3,), np.int64)
    b = DynamicBoxSpace(NB, 0, 100001, (0,), (3,), np.int64)
    assert a != b


def test_vec_env_batching_keeps_differing_integer_bounds():
    from unienv_interface.env_base.vec_env import _batch_env_spaces
    a = BoxSpace(NB, 0, 100000, np.int64, shape=())
    b = BoxSpace(NB, 0, 100001, np.int64, shape=())
    assert _batch_env_spaces([a, b]).contains(np.array([0, 100001]))


# --- 2. / 10. dtype (de)serialization --------------------------------------------------

@needs_jax_x32
def test_numpy_float64_space_deserializes_on_jax():
    space = DictSpace(NB, {
        "f": BoxSpace(NB, -1.0, 1.0, np.float64, shape=(3,)),
        "i": BoxSpace(NB, -5, 5, np.int64, shape=(2,)),
    })
    payload = json.loads(json.dumps(ssu.space_to_json(space)))
    restored = ssu.json_to_space(payload, JB, None)
    assert restored == space.to(JB)


@needs_jax_x32
def test_deserialize_dtype_jax_maps_64_bit_names_to_stored_dtype():
    assert deserialize_dtype(JB, "float64") == np.float32
    assert deserialize_dtype(JB, "int64") == np.int32


@pytest.mark.parametrize("dtype", [np.float16, torch.float16, torch.bfloat16])
def test_half_precision_box_serializes(dtype):
    backend = NB if isinstance(dtype, type) else TB
    space = BoxSpace(backend, -1.5, 2.0, dtype, shape=(3,))
    payload = json.loads(json.dumps(ssu.space_to_json(space)))
    restored = ssu.json_to_space(payload, backend, None)
    assert restored == space
    assert restored.dtype == space.dtype


@pytest.mark.parametrize("backend,dtype", [(NB, np.float16), (TB, torch.float16), (TB, torch.bfloat16)])
def test_half_precision_dtype_round_trip(backend, dtype):
    name = serialize_dtype(backend, dtype)
    assert deserialize_dtype(backend, name) == dtype


def test_bfloat16_array_value_round_trip():
    from unienv_interface.utils.array_serialization import (
        deserialize_scalar_or_array_value,
        serialize_scalar_or_array_value,
    )
    value = torch.tensor([0.25, 1.5], dtype=torch.bfloat16)
    _, payload = serialize_scalar_or_array_value(value)
    _, restored = deserialize_scalar_or_array_value(json.loads(json.dumps(payload)))
    assert restored.dtype == torch.bfloat16
    assert torch.equal(restored, value)


def test_unknown_dtype_names_still_rejected():
    with pytest.raises(ValueError):
        deserialize_dtype(NB, "U5")
    with pytest.raises(ValueError):
        deserialize_dtype(TB, "float128x")


# --- 3. int64 bounds moved to jax are clipped, not wrapped ------------------------------

@needs_jax_x32
def test_int64_bounds_to_jax_are_clipped():
    box = BoxSpace(NB, 0, 2**32 + 5, np.int64, shape=(2,))
    on_jax = box.to(JB)
    assert int(on_jax.high[0]) == 2**31 - 1
    assert on_jax.contains(jax.numpy.asarray([10, 10]))


@needs_jax_x32
def test_full_range_int64_box_to_jax():
    full = BoxSpace(NB, -np.inf, np.inf, np.int64, shape=(2,))
    for converted in (full.to(JB), ssu.json_to_space(ssu.space_to_json(full), JB, None)):
        assert int(converted.low[0]) == -(2**31)
        assert int(converted.high[0]) == 2**31 - 1


@needs_jax_x32
def test_int64_dynamic_box_to_jax_is_clipped():
    space = DynamicBoxSpace(NB, 0, 2**32 + 5, (0,), (3,), np.int64)
    assert int(space.to(JB).high[0]) == 2**31 - 1


# --- 4. negative start_dim for composite spaces ----------------------------------------

def test_dict_flatten_negative_start_dim():
    space = DictSpace(NB, {"a": _box(2, 3, 4), "b": _box(2, 3, 4)})
    x = {"a": np.zeros((2, 3, 4), np.float32), "b": np.ones((2, 3, 4), np.float32)}
    flat = sfu.flatten_data(space, x, -2)
    assert flat.shape == (2, 24)
    assert sfu.flatten_space(space, -2).shape == (2, 24)
    assert sfu.flatten_space(space, -2).contains(flat)
    back = sfu.unflatten_data(space, flat, -2)
    assert all(np.array_equal(back[k], x[k]) for k in x)


def test_tuple_flatten_negative_start_dim():
    space = TupleSpace(NB, [_box(2, 3), _box(2, 5)])
    x = (np.zeros((2, 3), np.float32), np.ones((2, 5), np.float32))
    flat = sfu.flatten_data(space, x, -1)
    assert flat.shape == (2, 8)
    flat2 = sfu.flatten_data(TupleSpace(NB, [_box(2, 3), _box(2, 3)]), (x[0], x[0]), -2)
    assert flat2.shape == (12,)
    back = sfu.unflatten_data(space, flat, -1)
    assert np.array_equal(back[1], x[1])


# --- 5. unpad keeps shape_low ----------------------------------------------------------

def test_dynamic_box_unflatten_keeps_trailing_fill_within_shape_low():
    space = DynamicBoxSpace(NB, 0, 10, shape_low=(2,), shape_high=(5,), dtype=np.int64)
    x = np.array([3, 0])
    back = sfu.unflatten_data(space, sfu.flatten_data(space, x))
    assert np.array_equal(back, x)
    assert space.contains(back)


# --- 6. empty batches ------------------------------------------------------------------

@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_flatten_empty_batch(backend, device):
    f32 = backend.zeros((1,), dtype=backend.default_floating_dtype).dtype
    single = DictSpace(backend, {
        "a": BoxSpace(backend, 0.0, 1.0, f32, shape=(3,), device=device),
        "b": BoxSpace(backend, 0.0, 1.0, f32, shape=(2, 2), device=device),
    })
    batched = sbu.batch_space(single, 1)
    data = {"a": backend.zeros((0, 3), dtype=f32), "b": backend.zeros((0, 2, 2), dtype=f32)}
    flat = sfu.flatten_data(batched, data, start_dim=1)
    assert tuple(flat.shape) == (0, 7)
    back = sfu.unflatten_data(batched, flat, start_dim=1)
    assert tuple(back["b"].shape) == (0, 2, 2)
    assert sfu.flatten_space(BoxSpace(backend, 0.0, 1.0, f32, shape=(0, 3), device=device), 1).shape == (0, 3)


# --- 7. text flattening follows the device ---------------------------------------------

@needs_cuda
def test_text_flatten_on_cuda():
    text = TextSpace(TB, 4, charset="abc", device="cuda")
    space = DictSpace(TB, {"a": BoxSpace(TB, 0.0, 1.0, torch.float32, shape=(2,), device="cuda"), "t": text})
    flat = sfu.flatten_data(space, {"a": torch.zeros(2, device="cuda"), "t": "ab"})
    assert flat.device.type == "cuda"
    assert sfu.flatten_data(text, "ab").device.type == "cuda"
    assert sfu.unflatten_data(space, flat)["t"] == "ab"


# --- 8. set_at on a batched dynamic box ------------------------------------------------

def test_set_at_dynamic_box_integer_index_pads():
    batched = sbu.batch_space(DynamicBoxSpace(NB, 0, 10, shape_low=(0,), shape_high=(5,), dtype=np.int64), 3)
    items = np.zeros((3, 5), dtype=np.int64)
    result = sbu.set_at(batched, items, 0, np.array([1, 2, 3]))
    assert np.array_equal(result[0], [1, 2, 3, 0, 0])
    result = sbu.set_at(batched, np.zeros((3, 5), dtype=np.int64), slice(1, 3), np.array([[4], [5]]))
    assert np.array_equal(result[1:], [[4, 0, 0, 0, 0], [5, 0, 0, 0, 0]])


# --- 9. padding outside the value bounds ----------------------------------------------

def test_dynamic_box_concatenate_with_fill_outside_bounds():
    space = DynamicBoxSpace(NB, 1, 5, shape_low=(1,), shape_high=(4,), dtype=np.int64)
    batch = sbu.concatenate(space, [np.array([1]), np.array([2, 3, 4])])
    batched = sbu.batch_space(space, 2)
    assert batched.contains(batch)
    assert space.contains(sbu.get_at(batched, batch, 0))
    # padding is only accepted beyond shape_low, and only as the fill value
    assert not space.contains(np.array([0, 2]))
    assert not space.contains(np.array([2, 7]))
    # clip stays idempotent on padded members
    assert np.array_equal(space.clip(np.array([2, 0, 0])), [2, 0, 0])
    assert np.array_equal(space.clip(np.array([2, 9, 0])), [2, 5, 0])


def test_dynamic_box_subspace_requires_fill_value_membership():
    sub = DynamicBoxSpace(NB, 1, 5, (1,), (4,), np.int64, fill_value=0)
    assert not sub.is_subspaceeq(DynamicBoxSpace(NB, 1, 5, (1,), (4,), np.int64, fill_value=-1))
    assert sub.is_subspaceeq(DynamicBoxSpace(NB, 0, 5, (1,), (4,), np.int64, fill_value=-1))
    assert sub.is_subspaceeq(DynamicBoxSpace(NB, 1, 5, (1,), (4,), np.int64, fill_value=0))


# --- 11. contains for data on another device -------------------------------------------

@needs_cuda
def test_box_contains_data_on_other_device():
    x = torch.tensor([0.5, 0.5], device="cuda")
    per_element = BoxSpace(TB, torch.tensor([0.0, -1.0]), torch.tensor([1.0, 1.0]), torch.float32, device="cpu")
    assert per_element.contains(x)
    assert not per_element.contains(torch.tensor([2.0, 0.0], device="cuda"))
    dyn = DynamicBoxSpace(TB, 0.0, 1.0, (1,), (3,), torch.float32, device="cpu")
    assert dyn.contains(x)


# --- 12. graphs with an empty batch ------------------------------------------------------

@pytest.mark.parametrize("is_edge", [False, True])
def test_graph_empty_batch(is_edge):
    single = GraphSpace(
        NB, _box(2), _box(1) if is_edge else None, is_edge=is_edge, max_nodes=4, max_edges=3 if is_edge else None
    )
    space = sbu.batch_space(single, 0)
    _, sample = space.sample(np.random.default_rng(0))
    assert space.contains(sample)
    x = GraphInstance(n_nodes=np.zeros((0,), np.int64), nodes_features=np.zeros((0, 0, 2), np.float32))
    assert space.contains(x) == (not is_edge)


# --- 13. empty charset ---------------------------------------------------------------

def test_text_empty_charset_samples_empty_strings():
    text = TextSpace(NB, 5, charset="")
    rng = np.random.default_rng(0)
    for _ in range(5):
        rng, sample = text.sample(rng)
        assert sample == "" and text.contains(sample)
    with pytest.raises(ValueError):
        TextSpace(NB, 5, min_length=1, charset="").sample(rng)


# --- 14. bounds with fewer dims -------------------------------------------------------

def test_box_bounds_with_fewer_dims_broadcast():
    box = BoxSpace(NB, np.zeros(3), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.float32)
    assert box.shape == (2, 3)
    assert np.array_equal(box.low, np.zeros((2, 3)))
    box = BoxSpace(NB, np.zeros(3), np.ones(3), np.float32, shape=(2, 3))
    assert box.contains(np.full((2, 3), 0.5, np.float32))


# --- 15. dict_util keeps empty nested dicts -------------------------------------------

def test_flatten_keys_keeps_empty_nested_dicts():
    from unienv_interface.utils import dict_util as du
    data = {"a": {}, "b": {"c": 1}}
    assert du.unflatten_keys_in_mapping(du.flatten_keys_in_mapping(data)) == data
    space = DictSpace(NB, {"a": DictSpace(NB, {}), "b": _box(1)})
    assert du.unflatten_keys_in_mapping(du.flatten_keys_in_mapping(space)) == space
