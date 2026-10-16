"""Default implementations of BatchBase / BatchSampler in unienv_data.base.common."""

import numpy as np
import pytest

from unienv_data.base import BatchBase, BatchSampler
from unienv_data.base.common import convert_index_to_backendarray
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.space.space_utils import flatten_utils as sfu

from tests_common.trees import assert_tree_equal

B = NumpyComputeBackend


def _space() -> DictSpace:
    return DictSpace(B, {
        "x": BoxSpace(B, -100.0, 100.0, np.float32, shape=(2,)),
        "y": BoxSpace(B, -100.0, 100.0, np.float32, shape=()),
    })


class FlatArrayBatch(BatchBase):
    """Implements only the flattened primitives; everything else uses the BatchBase defaults."""

    def __init__(self, space):
        super().__init__(space)
        self.data = np.zeros((0, sfu.flatten_space(space).shape[0]), np.float32)

    def __len__(self):
        return self.data.shape[0]

    def get_flattened_at(self, idx):
        return self.data[idx].copy()

    def set_flattened_at(self, idx, value):
        self.data[idx] = value

    def extend_flattened(self, value):
        self.data = np.concatenate([self.data, value], axis=0)


class StructuredListBatch(BatchBase):
    """Implements only the structured primitives."""

    def __init__(self, space, rows):
        super().__init__(space)
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def get_at(self, idx):
        if isinstance(idx, int):
            return self.rows[idx]
        idx = np.arange(len(self.rows))[idx]
        return {
            "x": np.stack([self.rows[i]["x"] for i in idx]),
            "y": np.stack([self.rows[i]["y"] for i in idx]),
        }


def _row(v: float) -> dict:
    return {"x": np.array([v, -v], np.float32), "y": np.float32(v * 10)}


def _rows(values) -> dict:
    return {"x": np.stack([_row(v)["x"] for v in values]), "y": np.stack([_row(v)["y"] for v in values])}


@pytest.fixture
def flat_batch():
    batch = FlatArrayBatch(_space())
    batch.extend(_rows([0.0, 1.0, 2.0]))
    batch.append(_row(3.0))
    return batch


def test_flat_primitives_drive_structured_defaults(flat_batch):
    assert len(flat_batch) == 4
    assert_tree_equal(flat_batch.get_at(1), _row(1.0))
    assert_tree_equal(flat_batch[slice(1, 3)], _rows([1.0, 2.0]))
    assert_tree_equal(flat_batch.get_at(np.array([3, 0])), _rows([3.0, 0.0]))
    flat_batch.set_at(0, _row(9.0))
    flat_batch[slice(1, 3)] = _rows([7.0, 8.0])
    assert_tree_equal(flat_batch.get_at(slice(None)), _rows([9.0, 7.0, 8.0, 3.0]))
    flat_batch.append_flattened(np.array([1.0, 2.0, 3.0], np.float32))
    assert_tree_equal(flat_batch.get_at(-1), {"x": np.array([1.0, 2.0], np.float32), "y": np.float32(3.0)})


def test_with_metadata_defaults_do_not_recurse(flat_batch):
    value, metadata = flat_batch.get_at_with_metadata(2)
    assert_tree_equal(value, _row(2.0))
    assert metadata is None
    flat, metadata = flat_batch.get_flattened_at_with_metadata(slice(0, 2))
    np.testing.assert_array_equal(flat, flat_batch.get_flattened_at(slice(0, 2)))
    assert metadata is None

    structured = StructuredListBatch(_space(), [_row(1.0), _row(2.0)])
    value, metadata = structured.get_at_with_metadata(slice(None))
    assert_tree_equal(value, _rows([1.0, 2.0]))
    assert metadata is None
    flat, metadata = structured.get_flattened_at_with_metadata(1)
    np.testing.assert_array_equal(flat, np.array([2.0, -2.0, 20.0], np.float32))
    assert metadata is None
    np.testing.assert_array_equal(structured.get_flattened_at(slice(None)),
                                  np.array([[1.0, -1.0, 10.0], [2.0, -2.0, 20.0]], np.float32))


def test_unimplemented_mutators_raise(flat_batch):
    structured = StructuredListBatch(_space(), [_row(1.0)])
    with pytest.raises(NotImplementedError):
        structured.set_at(0, _row(2.0))
    with pytest.raises(NotImplementedError):
        del flat_batch[0]


def test_extend_from_copies_in_chunks(flat_batch):
    target = FlatArrayBatch(_space())
    target.extend_from(flat_batch, chunk_size=3)
    np.testing.assert_array_equal(target.data, flat_batch.data)
    target.extend_from(flat_batch, chunk_size=10, tqdm=True)
    assert len(target) == 8


def test_get_slice_and_get_column_views(flat_batch):
    sliced = flat_batch.get_slice(slice(1, 3))
    assert len(sliced) == 2
    assert_tree_equal(sliced.get_at(slice(None)), _rows([1.0, 2.0]))
    column = flat_batch.get_column(["y"])
    np.testing.assert_array_equal(column.get_at(slice(None)), np.array([0.0, 10.0, 20.0, 30.0], np.float32))


def test_get_slice_single_index_is_unbatched(flat_batch):
    sliced = flat_batch.get_slice(slice(1, 3))
    assert_tree_equal(sliced.get_at(0), _row(1.0))


def test_batched_space_properties(flat_batch):
    assert flat_batch.backend is B and flat_batch.device is None
    assert flat_batch._batched_metadata_space is None
    meta_space = DictSpace(B, {"w": BoxSpace(B, 0.0, 1.0, np.float32, shape=())})
    with_meta = FlatArrayBatch(_space())
    with_meta.single_metadata_space = meta_space
    assert with_meta._batched_metadata_space is not None


@pytest.mark.parametrize("index,expected", [
    (2, [2]), (slice(1, 5, 2), [1, 3]), (Ellipsis, [0, 1, 2, 3, 4]),
])
def test_convert_index_to_backendarray(index, expected):
    result = convert_index_to_backendarray(B, index, 5)
    np.testing.assert_array_equal(result, np.array(expected))
    assert B.dtype_is_real_integer(result.dtype)


def test_convert_index_to_backendarray_rejects_arrays():
    with pytest.raises(ValueError):
        convert_index_to_backendarray(B, np.array([1]), 5)


class SimpleSampler(BatchSampler):
    def __init__(self, data, batch_size, seed=0, data_rng=False):
        super().__init__(data.single_space, None, batch_size=batch_size)
        self.data = data
        self.rng = B.random.random_number_generator(seed)
        if data_rng:
            self.data_rng = B.random.random_number_generator(seed + 1)

    def get_flattened_at(self, idx):
        return self.data.get_flattened_at(idx)


@pytest.fixture
def big_batch():
    batch = FlatArrayBatch(_space())
    batch.extend(_rows([float(i) for i in range(7)]))
    return batch


@pytest.mark.parametrize("data_rng", [False, True], ids=["shared-rng", "data-rng"])
def test_sampler_sampling_and_epochs(big_batch, data_rng):
    sampler = SimpleSampler(big_batch, batch_size=3, data_rng=data_rng)
    assert len(sampler) == 7
    assert sampler.sampled_space == sampler._batched_space
    assert sampler.sampled_metadata_space is None

    idx = sampler.sample_index()
    assert idx.shape == (3,) and np.all((idx >= 0) & (idx < 7))
    sample = sampler.sample()
    assert sample["x"].shape == (3, 2)
    flat = sampler.sample_flat()
    assert flat.shape == (3, 3)
    value, metadata = sampler.sample_with_metadata()
    assert value["y"].shape == (3,) and metadata is None
    flat, metadata = sampler.sample_flat_with_metadata()
    assert flat.shape == (3, 3) and metadata is None

    seen = np.concatenate([batch["y"] for batch in sampler])
    assert sorted(seen.tolist()) == [float(i) * 10 for i in range(7)]
    sizes = [b["y"].shape[0] for b in sampler.epoch_iter()]
    assert sizes == [3, 3, 1]
    flat_rows = np.concatenate(list(sampler.epoch_flat_iter()))
    assert sorted(flat_rows[:, 2].tolist()) == [float(i) * 10 for i in range(7)]
    with_meta = list(sampler.epoch_iter_with_metadata())
    assert [m for _, m in with_meta] == [None, None, None]
    flat_meta = list(sampler.epoch_flat_iter_with_metadata())
    assert sum(f.shape[0] for f, _ in flat_meta) == 7


def test_sampler_manual_seed_is_reproducible(big_batch):
    for data_rng in (False, True):
        sampler = SimpleSampler(big_batch, batch_size=4, data_rng=data_rng)
        sampler.manual_seed(123)
        first = [sampler.sample_index() for _ in range(3)]
        sampler.manual_seed(123)
        second = [sampler.sample_index() for _ in range(3)]
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
    sampler.close()
