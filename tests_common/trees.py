"""Backend-agnostic structural equality for nested space data."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def to_numpy(x: Any) -> Any:
    """Convert a backend array (numpy/torch/jax) to numpy; leave other values alone."""
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    if hasattr(x, "__array__") and hasattr(x, "dtype") and hasattr(x, "shape"):
        return np.asarray(x)
    return x


def assert_tree_equal(actual: Any, expected: Any, *, rtol: float = 0.0, atol: float = 0.0,
                      check_dtype: bool = True, path: str = "") -> None:
    """Assert two nested structures (dict/tuple/list/arrays/scalars/str/GraphInstance) are equal.

    Arrays from any backend are compared after conversion to numpy. With
    ``rtol``/``atol`` left at zero the comparison is exact (NaNs compare equal).
    """
    from unienv_interface.space.spaces.graph import GraphInstance

    where = path or "<root>"
    if isinstance(expected, GraphInstance):
        assert isinstance(actual, GraphInstance), f"{where}: expected GraphInstance, got {type(actual)}"
        for field in ("n_nodes", "n_edges", "nodes_features", "edges_features", "edges"):
            assert_tree_equal(getattr(actual, field, None), getattr(expected, field, None),
                              rtol=rtol, atol=atol, check_dtype=check_dtype, path=f"{path}.{field}")
        return
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), f"{where}: expected mapping, got {type(actual)}"
        assert set(actual.keys()) == set(expected.keys()), f"{where}: keys {set(actual)} != {set(expected)}"
        for key in expected:
            assert_tree_equal(actual[key], expected[key], rtol=rtol, atol=atol,
                              check_dtype=check_dtype, path=f"{path}[{key!r}]")
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)), f"{where}: expected sequence, got {type(actual)}"
        assert len(actual) == len(expected), f"{where}: length {len(actual)} != {len(expected)}"
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_tree_equal(a, b, rtol=rtol, atol=atol, check_dtype=check_dtype, path=f"{path}[{i}]")
        return
    if expected is None or isinstance(expected, (str, bytes)):
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"
        return

    a, b = to_numpy(actual), to_numpy(expected)
    if isinstance(b, np.ndarray) and b.dtype == object:
        assert isinstance(a, np.ndarray) and a.shape == b.shape, f"{where}: object array shape mismatch"
        for i, (x, y) in enumerate(zip(a.flat, b.flat)):
            assert_tree_equal(x, y, rtol=rtol, atol=atol, check_dtype=check_dtype, path=f"{path}.flat[{i}]")
        return
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    assert a_arr.shape == b_arr.shape, f"{where}: shape {a_arr.shape} != {b_arr.shape}"
    if check_dtype and isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        assert a_arr.dtype == b_arr.dtype, f"{where}: dtype {a_arr.dtype} != {b_arr.dtype}"
    if rtol == 0.0 and atol == 0.0:
        np.testing.assert_array_equal(a_arr, b_arr, err_msg=where)
    else:
        np.testing.assert_allclose(a_arr, b_arr, rtol=rtol, atol=atol, err_msg=where)
