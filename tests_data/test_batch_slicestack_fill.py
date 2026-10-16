"""SliceStackedBatch invalid-frame filling, checked against a reference implementation.

The view only re-gathers the rows holding invalid frames, in place; the reference below is
the full ``data[batch_idx, time_idx]`` gather it replaced, applied to the unfilled output
(``fill_invalid_data=False``) of an otherwise identical view.
"""

import itertools

import numpy as np
import pytest

from unienv_data.batches import SliceStackedBatch

from tests_common.batch_helpers import make_rb, numpy_array_batch
from tests_common.trees import assert_tree_equal, to_numpy


def scattered_mask(batch, expanded_idx, data, metadata):
    """Deterministic, non-contiguous valid mask (the anchor frame is always valid)."""
    idx = to_numpy(expanded_idx)
    t = np.arange(idx.shape[1])[None, :]
    mask = ((idx * 7 + t * 3) % 10) < 6
    mask[:, batch._idx_zero] = True
    return batch.backend.from_numpy(mask)


def reference_fill(x, mask, idx_zero):
    """Fill (B, T, ...) ``x`` as the original full-gather implementation did."""
    B, T = mask.shape
    first = idx_zero - mask[:, :idx_zero].sum(axis=1)
    last = idx_zero + mask[:, idx_zero + 1:].sum(axis=1)
    time_idx = np.repeat(np.arange(T)[None, :], B, axis=0)
    time_idx[:, :idx_zero] = np.where(mask[:, :idx_zero], time_idx[:, :idx_zero], first[:, None])
    time_idx[:, idx_zero + 1:] = np.where(mask[:, idx_zero + 1:], time_idx[:, idx_zero + 1:], last[:, None])
    return x[np.arange(B)[:, None], time_idx]


def map_tree(fn, tree):
    if isinstance(tree, dict):
        return {k: map_tree(fn, v) for k, v in tree.items()}
    return fn(tree)


SOURCES = {"numpy+metadata": lambda: numpy_array_batch(40), "torch replay buffer": lambda: make_rb(40)}
OFFSETS = {"prefetch": [-2, -1, 0], "postfetch": [0, 1, 2, 3], "sparse": [-3, -1, 0, 2, 5]}
MASKS = {"scattered": scattered_mask,
         "episode_id": SliceStackedBatch.get_valid_mask_function_with_episodeid_key("episode_id")}
CONFIGS = list(itertools.product(SOURCES, OFFSETS, [False, True], MASKS, [False, True], ["array", "bool"]))
assert len(CONFIGS) == 96


def _index(src, kind):
    rng = np.random.default_rng(0)
    if kind == "array":
        idx = rng.integers(-len(src), len(src), 23)  # negatives and duplicates
    else:
        idx = rng.random(len(src)) < 0.5
    return src.backend.from_numpy(idx)


@pytest.mark.parametrize("source,offsets,stack_metadata,mask,flat,index", CONFIGS)
def test_fill_matches_full_gather_reference(source, offsets, stack_metadata, mask, flat, index):
    src = SOURCES[source]()
    be = src.backend
    before = to_numpy(src.get_at(...))
    kwargs = dict(get_valid_mask_function=MASKS[mask], stack_metadata=stack_metadata)
    offs = be.from_numpy(np.array(OFFSETS[offsets]))
    filled = SliceStackedBatch(src, offs, **kwargs)
    raw = SliceStackedBatch(src, offs, fill_invalid_data=False, **kwargs)
    idx = _index(src, index)
    fn = "get_flattened_at_with_metadata" if flat else "get_at_with_metadata"
    data, meta = getattr(filled, fn)(idx)
    raw_data, raw_meta = getattr(raw, fn)(idx)

    valid = to_numpy(raw_meta["slice_valid_mask"])
    assert (~valid).any(), "the configuration should exercise the fill"
    z = filled._idx_zero
    assert_tree_equal(data, map_tree(lambda x: reference_fill(to_numpy(x), valid, z), raw_data))
    expected_meta = {k: v for k, v in raw_meta.items() if k != "slice_valid_mask"}
    if stack_metadata:
        expected_meta = map_tree(lambda x: reference_fill(to_numpy(x), valid, z), expected_meta)
    assert_tree_equal({k: v for k, v in meta.items() if k != "slice_valid_mask"}, expected_meta)
    assert_tree_equal(meta["slice_valid_mask"], valid)
    # Filling in place never writes through to the source.
    assert_tree_equal(src.get_at(...), before)


def test_fill_on_immutable_jax_arrays_matches_reference():
    jnp = pytest.importorskip("jax.numpy")
    from unienv_interface.backends.jax import JaxComputeBackend as JB
    from unienv_interface.space import BinarySpace, BoxSpace, DictSpace
    from tests_common.batch_helpers import ArrayBatch

    space = DictSpace(JB, {"obs": BoxSpace(JB, -1e6, 1e6, jnp.float32, shape=(2,)),
                           "episode_id": BoxSpace(JB, 0, 100, jnp.int32, shape=())})
    i = jnp.arange(30)
    src = ArrayBatch(space, {"obs": jnp.stack([2 * i, 2 * i + 1], -1).astype(jnp.float32), "episode_id": i // 4})
    offs = jnp.array([-2, -1, 0, 1])
    filled = SliceStackedBatch(src, offs, get_valid_mask_function=scattered_mask)
    raw = SliceStackedBatch(src, offs, get_valid_mask_function=scattered_mask, fill_invalid_data=False)
    idx = jnp.array([0, 3, 7, 12, 29, 15])
    for fn in ("get_at_with_metadata", "get_flattened_at_with_metadata"):
        data, _ = getattr(filled, fn)(idx)
        raw_data, raw_meta = getattr(raw, fn)(idx)
        valid = to_numpy(raw_meta["slice_valid_mask"])
        assert_tree_equal(data, map_tree(lambda x: reference_fill(to_numpy(x), valid, filled._idx_zero), raw_data))


def test_rows_without_invalid_frames_are_returned_as_gathered():
    rb = make_rb(40)
    view = SliceStackedBatch(rb, rb.backend.from_numpy(np.array([-1, 0, 1])))
    out, meta = view.get_at_with_metadata(rb.backend.from_numpy(np.array([5, 9, 20])))
    assert to_numpy(meta["slice_valid_mask"]).all()
    assert_tree_equal(out["obs"][:, :, 0], np.array([[8, 10, 12], [16, 18, 20], [38, 40, 42]], dtype=np.float32))
