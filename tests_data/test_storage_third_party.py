"""Vendored third-party helpers used by unienv_data (tensordict memmap, LeRobot metadata)."""

import json
import os
import pickle

import numpy as np
import pytest
import torch

from unienv_data.third_party.tensordict.memmap_tensor import MemoryMappedTensor


# ---------------------------------------------------------------------------
# MemoryMappedTensor (backs PytorchTensorStorage)
# ---------------------------------------------------------------------------

@pytest.fixture
def memmap_file(tmp_path):
    path = str(tmp_path / "data.memmap")
    MemoryMappedTensor.from_tensor(torch.arange(12.0).reshape(6, 2), filename=path)
    return path


def _read(path):
    return MemoryMappedTensor.from_filename(path, dtype=torch.float32, shape=(6, 2), readonly=True).clone()


@pytest.mark.parametrize("index", [slice(1, 3), 4, (slice(None), 1)], ids=["slice", "int", "column"])
def test_pickled_view_keeps_its_index(memmap_file, index):
    full = MemoryMappedTensor.from_filename(memmap_file, dtype=torch.float32, shape=(6, 2))
    view = full[index]
    assert isinstance(view, MemoryMappedTensor)
    clone = pickle.loads(pickle.dumps(view))
    assert clone.shape == view.shape
    torch.testing.assert_close(torch.as_tensor(clone), torch.as_tensor(view))


def test_pickled_read_only_tensor_stays_read_only(memmap_file):
    read_only = MemoryMappedTensor.from_filename(memmap_file, dtype=torch.float32, shape=(6, 2), readonly=True)
    clone = pickle.loads(pickle.dumps(read_only))
    clone[0] = 100.0  # private copy-on-write mapping
    torch.testing.assert_close(_read(memmap_file)[0], torch.tensor([0.0, 1.0]))


def test_pickled_writable_tensor_shares_the_file(memmap_file):
    writable = MemoryMappedTensor.from_filename(memmap_file, dtype=torch.float32, shape=(6, 2))
    clone = pickle.loads(pickle.dumps(writable))
    clone[5] = -1.0
    torch.testing.assert_close(_read(memmap_file)[5], torch.tensor([-1.0, -1.0]))
    state = writable.__getstate__()
    assert state["filename"] == os.path.abspath(memmap_file)


def test_memmap_constructors_used_by_storages(tmp_path):
    path = str(tmp_path / "empty.memmap")
    empty = MemoryMappedTensor.empty((3, 2), dtype=torch.float32, filename=path)
    assert empty.filename == os.path.abspath(path) and empty.shape == (3, 2)
    with pytest.raises(RuntimeError):
        MemoryMappedTensor.empty((3, 2), dtype=torch.float32, filename=path)
    again = MemoryMappedTensor.empty((3, 2), dtype=torch.float32, filename=path, existsok=True)
    again.copy_(torch.ones(3, 2))
    torch.testing.assert_close(MemoryMappedTensor.from_filename(path, torch.float32, (3, 2)), torch.ones(3, 2))
    # from_tensor of a memmap: same location returns it, a new location needs copy_existing
    assert MemoryMappedTensor.from_tensor(again, filename=path) is again
    with pytest.raises(RuntimeError):
        MemoryMappedTensor.from_tensor(again, filename=str(tmp_path / "other.memmap"))
    copied = MemoryMappedTensor.from_tensor(again, filename=str(tmp_path / "other.memmap"), copy_existing=True)
    torch.testing.assert_close(copied, torch.ones(3, 2))
    with pytest.raises(TypeError):
        MemoryMappedTensor.from_tensor(np.zeros(2))
    in_memory = MemoryMappedTensor.from_tensor(torch.arange(4, dtype=torch.int32))
    with pytest.raises(RuntimeError):
        in_memory.filename
    torch.testing.assert_close(MemoryMappedTensor.zeros(2, 2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64))
    torch.testing.assert_close(MemoryMappedTensor.ones((2,), dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(MemoryMappedTensor.full((2,), fill_value=3, dtype=torch.int64),
                               torch.full((2,), 3, dtype=torch.int64))
    with pytest.raises(RuntimeError):
        MemoryMappedTensor.empty((2,), dtype=torch.float32, device="meta")


# ---------------------------------------------------------------------------
# LeRobot dataset metadata helpers
# ---------------------------------------------------------------------------

from unienv_data.third_party.lerobot import (  # noqa: E402
    EpisodeMetadata,
    LeRobotIndex,
    LeRobotVersion,
    build_index,
    get_data_file_path,
    get_video_file_path,
    parse_info_json,
)


def _write_info(root, version, **extra):
    info = {
        "codebase_version": version,
        "fps": 10,
        "total_episodes": 3,
        "total_frames": 7,
        "chunks_size": 2,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "observation.image": {"dtype": "video", "shape": [4, 4, 3], "video_info": {
                "video.codec": "av1", "video.pix_fmt": "yuv420p", "video.fps": 10, "video.is_depth_map": False}},
            "observation.wrist": {"dtype": "image", "shape": [4, 4, 3]},
        },
        **extra,
    }
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)
    with open(os.path.join(root, "meta", "info.json"), "w") as f:
        json.dump(info, f)


def test_parse_info_json_features(tmp_path):
    _write_info(str(tmp_path), "v2.1")
    schema = parse_info_json(str(tmp_path))
    assert schema.codebase_version is LeRobotVersion.V2_1
    assert schema.robot_type == "unknown" and schema.total_tasks == 0
    assert set(schema.video_features) == {"observation.image"}
    assert set(schema.image_features) == {"observation.wrist"}
    assert set(schema.tabular_features) == {"observation.state"}
    video = schema.features["observation.image"]
    assert video.video_info.video_codec == "av1" and video.shape == (4, 4, 3)


def test_v2_paths_and_episodes(tmp_path):
    root = str(tmp_path)
    _write_info(root, "v2.0")
    with open(os.path.join(root, "meta", "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "pick"}) + "\n\n")
        f.write(json.dumps({"task_index": 1, "task": "place"}) + "\n")
    with open(os.path.join(root, "meta", "episodes.jsonl"), "w") as f:
        f.write(json.dumps({"episode_index": 2, "length": 3, "tasks": ["place"]}) + "\n")
        f.write(json.dumps({"episode_index": 0, "length": 4, "task_index": 1}) + "\n")
        f.write("\n")
        f.write(json.dumps({"episode_index": 1, "length": 0, "tasks": [1]}) + "\n")
    schema = parse_info_json(root)
    index = build_index(root, schema)
    assert [e.index for e in index.episodes] == [0, 1, 2]
    assert [e.task for e in index.episodes] == ["place", "place", "pick"]
    assert index.total_frames == 7 and index.total_episodes == 3
    assert index.episode_frame_range(2) == (4, 7)
    # the zero-length episode is skipped by the lookup
    assert index.global_index_to_episode_and_local(4) == (2, 0)
    groups = index.global_indices_to_episode_and_local(np.array([6, 0, 3, 4]))
    assert set(groups) == {0, 2}
    np.testing.assert_array_equal(groups[0][0], [0, 3])
    np.testing.assert_array_equal(groups[0][1], [1, 2])
    np.testing.assert_array_equal(groups[2][0], [2, 0])
    assert get_data_file_path(root, schema, 3) == os.path.join(root, "data", "chunk-001", "episode_000003.parquet")
    assert get_video_file_path(root, schema, "observation.image", 3) == os.path.join(
        root, "videos", "chunk-001", "observation.image", "episode_000003.mp4")


def test_v3_paths_and_packed_episodes(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = str(tmp_path)
    _write_info(root, "v3.0",
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    schema = parse_info_json(root)
    with pytest.raises(FileNotFoundError):
        build_index(root, schema)
    os.makedirs(os.path.join(root, "meta", "episodes"))
    pq.write_table(pa.table({
        "episode_index": [1, 0],
        "length": [3, 4],
        "data/chunk_index": [0, 0],
        "data/file_index": [0, 0],
        "dataset_from_index": [4, 0],
        "dataset_to_index": [7, 4],
    }), os.path.join(root, "meta", "episodes", "file-000.parquet"))
    open(os.path.join(root, "meta", "episodes", "README.txt"), "w").close()
    index = build_index(root, schema)
    assert [(e.index, e.dataset_from_index, e.data_file_index) for e in index.episodes] == [(0, 0, 0), (1, 4, 0)]
    assert index.episodes[0].task_index == 0
    assert get_data_file_path(root, schema, 1, file_index=0, chunk_index=0) == os.path.join(
        root, "data", "chunk-000", "file-000.parquet")
    assert get_data_file_path(root, schema, 5) == os.path.join(root, "data", "chunk-002", "file-005.parquet")
    assert get_video_file_path(root, schema, "cam", 5) == os.path.join(
        root, "videos", "cam", "chunk-002", "file-005.mp4")
    assert get_video_file_path(root, schema, "cam", 1, file_index=0, chunk_index=0) == os.path.join(
        root, "videos", "cam", "chunk-000", "file-000.mp4")


def test_index_with_no_episodes():
    index = LeRobotIndex(episodes=[], tasks={})
    assert index.total_frames == 0 and index.total_episodes == 0
    single = LeRobotIndex(episodes=[EpisodeMetadata(index=0, length=2)], tasks={})
    assert single.global_index_to_episode_and_local(1) == (0, 1)
