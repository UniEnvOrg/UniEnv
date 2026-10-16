"""PyTorch Dataset <-> UniEnv batch adapters."""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_data.integrations.pytorch import PyTorchAsUniEnvDataset, UniEnvAsPyTorchDataset

from tests_common.batch_helpers import make_rb
from tests_common.trees import assert_tree_equal


def test_unienv_as_torch_dataset_items():
    rb = make_rb(10)
    ds = UniEnvAsPyTorchDataset(rb)
    assert len(ds) == 10
    assert_tree_equal(ds[3], rb.get_at(3))
    assert_tree_equal(ds.__getitems__([1, 4]), rb.get_at(torch.tensor([1, 4])))
    with pytest.raises(AssertionError):
        ds[slice(0, 2)]


def test_dataloader_with_batched_fetch():
    rb = make_rb(10)
    ds = UniEnvAsPyTorchDataset(rb)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=UniEnvAsPyTorchDataset.collate_fn)
    batches = list(loader)
    assert [b["obs"].shape[0] for b in batches] == [4, 4, 2]
    assert_tree_equal(batches[1], rb.get_at(torch.arange(4, 8)))
    # Without batching the loader yields single samples.
    first = next(iter(DataLoader(ds, batch_size=None)))
    assert_tree_equal(first, rb.get_at(0))


def test_dataloader_with_metadata():
    rb = make_rb(6)
    ds = UniEnvAsPyTorchDataset(rb, include_metadata=True)
    data, meta = ds[2]
    assert_tree_equal(data, rb.get_at(2)) and meta is None
    loader = DataLoader(ds, batch_size=3, collate_fn=ds.collate_fn)
    data, meta = next(iter(loader))
    assert_tree_equal(data, rb.get_at(torch.arange(3)))


class _DictDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, i):
        return {"x": torch.full((2,), float(i)), "y": torch.tensor(i)}


class _BatchedDictDataset(_DictDataset):
    def __init__(self):
        self.getitems_calls = 0

    def __getitems__(self, indices):
        self.getitems_calls += 1
        return [self[i] for i in indices]


def _expected(ids):
    ids = torch.as_tensor(ids)
    return {"x": ids[:, None].float().repeat(1, 2), "y": ids}


@pytest.mark.parametrize("dataset_cls", [_DictDataset, _BatchedDictDataset])
@pytest.mark.parametrize("idx,ids", [
    (slice(1, 4), [1, 2, 3]),
    (..., [0, 1, 2, 3, 4, 5]),
    (torch.tensor([5, 0, 2]), [5, 0, 2]),
    (torch.tensor([-1, -6]), [5, 0]),
    (torch.tensor([True, False, True, False, False, True]), [0, 2, 5]),
], ids=["slice", "ellipsis", "int-array", "negative", "bool-mask"])
def test_torch_dataset_as_batch(dataset_cls, idx, ids):
    dataset = dataset_cls()
    batch = PyTorchAsUniEnvDataset(dataset)
    assert len(batch) == 6 and not batch.is_mutable
    out = batch.get_at(idx)
    assert_tree_equal(out, _expected(ids))
    assert sbu.batch_space(batch.single_space, len(ids)).contains(out)
    data, meta = batch.get_at_with_metadata(idx)
    assert meta == {}
    if dataset_cls is _BatchedDictDataset:
        assert dataset.getitems_calls == 2


def test_torch_dataset_single_items_and_flattening():
    batch = PyTorchAsUniEnvDataset(_DictDataset())
    assert_tree_equal(batch.get_at(4), {"x": torch.tensor([4.0, 4.0]), "y": torch.tensor(4)})
    assert batch.single_space.contains(batch.get_at(4))
    flat = batch.get_flattened_at(torch.tensor([1, 2]))
    assert flat.shape == (2, 3)
    assert_tree_equal(flat[:, -1], torch.tensor([1.0, 2.0]), check_dtype=False)
