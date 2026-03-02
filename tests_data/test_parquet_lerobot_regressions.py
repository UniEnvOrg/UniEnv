import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace
from unienv_data.storages.parquet import ParquetStorage
from unienv_data.integrations.lerobot import LeRobotAsUniEnvDataset

try:
    import torch
except ImportError:
    torch = None


def _float_space(shape):
    return BoxSpace(
        NumpyComputeBackend,
        low=-np.inf,
        high=np.inf,
        dtype=np.float32,
        shape=shape,
    )


def _write_lerobot_v2_dataset(root_dir: Path) -> Path:
    dataset_dir = root_dir / "lerobot_v2"
    (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "testbot",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 1,
        "total_tasks": 1,
        "chunks_size": 1000,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [1]},
            "observation.image": {
                "dtype": "video",
                "shape": [2, 2, 3],
                "video_info": {"video.codec": "h264"},
            },
        },
    }
    with open(dataset_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f)

    with open(dataset_dir / "meta" / "episodes.jsonl", "w") as f:
        f.write(json.dumps({"episode_index": 0, "length": 1, "task_index": 0}) + "\n")

    table = pa.table({
        "observation.state": pa.array([[123.0]], type=pa.list_(pa.float32(), 1)),
    })
    pq.write_table(table, dataset_dir / "data" / "chunk-000" / "episode_000000.parquet")
    return dataset_dir


def _write_lerobot_v3_packed_dataset(root_dir: Path) -> Path:
    dataset_dir = root_dir / "lerobot_v3"
    (dataset_dir / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "testbot",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 2,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:06d}.parquet",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [1]},
        },
    }
    with open(dataset_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f)

    episodes_table = pa.table({
        "episode_index": pa.array([0], type=pa.int64()),
        "length": pa.array([2], type=pa.int64()),
        "task_index": pa.array([0], type=pa.int64()),
        "data/chunk_index": pa.array([0], type=pa.int64()),
        "data/file_index": pa.array([0], type=pa.int64()),
        "dataset_from_index": pa.array([3], type=pa.int64()),
        "dataset_to_index": pa.array([5], type=pa.int64()),
    })
    pq.write_table(episodes_table, dataset_dir / "meta" / "episodes" / "part-000.parquet")

    packed_values = np.arange(6, dtype=np.float32)
    packed_table = pa.table({
        "observation.state": pa.array(
            [[float(v)] for v in packed_values],
            type=pa.list_(pa.float32(), 1),
        ),
    })
    pq.write_table(packed_table, dataset_dir / "data" / "chunk-000" / "file-000000.parquet")
    return dataset_dir


def test_parquet_build_space_from_empty_storage(tmp_path: Path):
    storage_path = tmp_path / "parquet_empty"
    space = _float_space((2,))
    storage = ParquetStorage.create(space, capacity=None, cache_path=storage_path)
    storage.close()

    count, capacity, inferred_space = ParquetStorage.build_space_from_parquet_file(storage_path)
    assert count == 0
    assert capacity == 0
    assert isinstance(inferred_space, BoxSpace)
    assert inferred_space.shape == (2,)
    assert np.dtype(inferred_space.dtype) == np.dtype(np.float32)


def test_parquet_read_only_blocks_length_mutation_for_unbounded_storage(tmp_path: Path):
    storage_path = tmp_path / "parquet_rw"
    space = _float_space((2,))
    storage = ParquetStorage.create(space, capacity=None, cache_path=storage_path, piece_size=2)
    storage.extend_length(2)
    storage.set(slice(0, 2), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    storage.close()

    read_only_storage = ParquetStorage.load_from(
        storage_path,
        space,
        capacity=None,
        read_only=True,
    )
    assert not read_only_storage.is_mutable
    with pytest.raises(AssertionError):
        read_only_storage.extend_length(1)
    with pytest.raises(AssertionError):
        read_only_storage.shrink_length(1)
    read_only_storage.close()


def test_parquet_load_ignores_stale_piece_files(tmp_path: Path):
    storage_path = tmp_path / "parquet_stale"
    space = _float_space((1,))
    storage = ParquetStorage.create(space, capacity=None, cache_path=storage_path, piece_size=2)
    storage.extend_length(2)
    storage.set(slice(0, 2), np.array([[1.0], [2.0]], dtype=np.float32))
    storage.close()

    shutil.copy2(storage_path / "part-000000.parquet", storage_path / "part-000005.parquet")

    loaded = ParquetStorage.load_from(storage_path, space, capacity=None, read_only=False)
    loaded.extend_length(1)
    loaded.set(2, np.array([3.0], dtype=np.float32))
    values = loaded.get(slice(0, 3)).reshape(-1)
    np.testing.assert_allclose(values, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    loaded.close()


def test_parquet_negative_indices_supported(tmp_path: Path):
    storage_path = tmp_path / "parquet_negative_index"
    space = _float_space((1,))
    storage = ParquetStorage.create(space, capacity=3, cache_path=storage_path, piece_size=2)
    storage.set(slice(0, 3), np.array([[10.0], [20.0], [30.0]], dtype=np.float32))

    np.testing.assert_allclose(storage.get(-1), np.array([30.0], dtype=np.float32))
    np.testing.assert_allclose(storage.get(-2), np.array([20.0], dtype=np.float32))
    storage.set(-1, np.array([99.0], dtype=np.float32))
    np.testing.assert_allclose(storage.get(2), np.array([99.0], dtype=np.float32))
    storage.close()


def test_parquet_read_only_get_uses_arrow_piece_cache(tmp_path: Path):
    storage_path = tmp_path / "parquet_read_only_arrow_cache"
    space = _float_space((2,))
    storage = ParquetStorage.create(space, capacity=None, cache_path=storage_path, piece_size=2)
    storage.extend_length(4)
    storage.set(
        slice(0, 4),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    )
    storage.close()

    read_only_storage = ParquetStorage.load_from(storage_path, space, capacity=None, read_only=True)
    values = read_only_storage.get(np.array([0, 3], dtype=np.int64))
    np.testing.assert_allclose(values, np.array([[1.0, 2.0], [7.0, 8.0]], dtype=np.float32))
    assert read_only_storage._cached_pieces == {}
    assert len(read_only_storage._cached_piece_tables) > 0
    read_only_storage.close()


def test_parquet_column_compression_dict_respected(tmp_path: Path):
    storage_path = tmp_path / "parquet_compression"
    space = _float_space((2,))
    storage = ParquetStorage.create(
        space,
        capacity=2,
        cache_path=storage_path,
        compression={"data": "gzip"},
    )
    storage.close()

    metadata = pq.ParquetFile(storage_path / "part-000000.parquet").metadata
    compression = metadata.row_group(0).column(0).compression
    assert str(compression).upper() == "GZIP"


def test_lerobot_decode_video_false_excludes_video_features(tmp_path: Path):
    dataset_dir = _write_lerobot_v2_dataset(tmp_path)
    dataset = LeRobotAsUniEnvDataset.from_local(
        str(dataset_dir),
        backend=NumpyComputeBackend,
        decode_video=False,
    )

    assert "observation.image" not in dataset.single_space.spaces
    frame = dataset.get_at(0)
    assert "observation.image" not in frame
    np.testing.assert_allclose(frame["observation.state"], np.array([123.0], dtype=np.float32))


def test_lerobot_out_of_bounds_indices_raise(tmp_path: Path):
    dataset_dir = _write_lerobot_v2_dataset(tmp_path)
    dataset = LeRobotAsUniEnvDataset.from_local(
        str(dataset_dir),
        backend=NumpyComputeBackend,
        decode_video=False,
    )

    np.testing.assert_allclose(dataset.get_at(-1)["observation.state"], np.array([123.0], dtype=np.float32))
    with pytest.raises(IndexError):
        dataset.get_at(1)
    with pytest.raises(IndexError):
        dataset.get_at(-2)
    with pytest.raises(IndexError):
        dataset.get_at(np.array([0, 1], dtype=np.int64))


def test_lerobot_v3_packed_episode_uses_dataset_offset(tmp_path: Path):
    dataset_dir = _write_lerobot_v3_packed_dataset(tmp_path)
    dataset = LeRobotAsUniEnvDataset.from_local(
        str(dataset_dir),
        backend=NumpyComputeBackend,
    )

    episode = dataset.get_episode(0)
    np.testing.assert_allclose(
        np.asarray(episode["observation.state"]).reshape(-1),
        np.array([3.0, 4.0], dtype=np.float32),
    )


@pytest.mark.skipif(torch is None, reason="PyTorch is required for this test")
def test_lerobot_torch_backend_uses_torch_fastpath(tmp_path: Path):
    dataset_dir = _write_lerobot_v2_dataset(tmp_path)
    dataset = LeRobotAsUniEnvDataset.from_local(
        str(dataset_dir),
        backend=PyTorchComputeBackend,
        decode_video=False,
    )
    assert dataset._use_torch_fastpath

    batch = dataset.get_at(slice(0, 1))
    assert isinstance(batch["observation.state"], torch.Tensor)
    np.testing.assert_allclose(
        batch["observation.state"].cpu().numpy().reshape(-1),
        np.array([123.0], dtype=np.float32),
    )

    cached_episode = dataset._parquet_cache[0]
    assert isinstance(cached_episode["observation.state"], torch.Tensor)
