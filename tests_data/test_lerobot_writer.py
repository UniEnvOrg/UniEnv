"""Writing LeRobot datasets (v3.0): read back by UniEnv's reader (and by LeRobot itself when installed)."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace, TextSpace
from unienv_data.integrations.lerobot import (LeRobotAsUniEnvDataset, LeRobotDatasetWriter, export_lerobot,
                                              lerobot_features)

av = pytest.importorskip("av")

SPACE = DictSpace(B, {
    "obs": DictSpace(B, {
        "state": BoxSpace(B, -100, 100, np.float32, shape=(3,), metadata={"names": ["x", "y", "z"]}),
        "front": BoxSpace(B, 0, 255, np.uint8, shape=(32, 48, 3), metadata={"kind": "image"}),
        "note": TextSpace(B, 16),
    }),
    "action": BoxSpace(B, -1, 1, np.float32, shape=(2,)),
    "count": BoxSpace(B, 0, 100, np.int64, shape=()),
    "grid": BoxSpace(B, 0, 9, np.int64, shape=(2, 2)),
})
NAMES = {"observation.state": "obs/state", "observation.images.front": "obs/front", "action": "action",
         "count": "count", "grid": "grid"}


def frame(episode, t):
    return {"obs": {"state": np.full(3, 10 * episode + t, np.float32), "front": np.full((32, 48, 3), 20 * t, np.uint8),
                    "note": "x"},
            "action": np.full(2, -t / 10, np.float32), "count": np.int64(t), "grid": np.full((2, 2), t, np.int64)}


def write(root, episodes=(5, 6, 4), tasks=("pick", "place", "pick"), **kwargs):
    kwargs.setdefault("features", NAMES)
    with LeRobotDatasetWriter.create(root, SPACE, fps=10, **kwargs) as writer:
        for episode, (length, task) in enumerate(zip(episodes, tasks)):
            for t in range(length):
                writer.add_frame(frame(episode, t), task=task)
            assert writer.save_episode() == episode
    return writer


@pytest.mark.parametrize("videos", [True, False], ids=["video", "png"])
def test_round_trip_through_the_reader(tmp_path, videos):
    writer = write(tmp_path / "ds", videos=videos, robot_type="testbot")
    assert writer.num_episodes == 3 and writer.num_frames == 15
    ds = LeRobotAsUniEnvDataset.from_local(str(tmp_path / "ds"), backend=B)
    assert len(ds) == 15 and ds.get_episode_segments() == [(0, 5, 0), (5, 11, 1), (11, 15, 2)]
    picks = np.array([0, 4, 5, 10, 14])
    data, metadata = ds.get_at_with_metadata(picks)
    np.testing.assert_array_equal(data["observation.state"][:, 0], [0, 4, 10, 15, 23])
    np.testing.assert_allclose(data["action"][:, 0], [0, -0.4, 0, -0.5, -0.3], rtol=1e-6)
    np.testing.assert_array_equal(data["count"].reshape(-1), [0, 4, 0, 5, 3])
    np.testing.assert_array_equal(data["grid"][:, 1, 1], [0, 4, 0, 5, 3])
    images = data["observation.images.front"].reshape(len(picks), -1).astype(float).mean(axis=1)
    np.testing.assert_allclose(images, [0, 80, 0, 100, 60], atol=3 if videos else 0)  # PNG is lossless
    assert list(metadata["episode"]["task"]) == ["pick", "pick", "place", "place", "pick"]

    info = json.loads((tmp_path / "ds/meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["total_episodes"] == 3 and info["total_tasks"] == 2
    assert info["robot_type"] == "testbot" and info["splits"] == {"train": "0:3"}
    front = info["features"]["observation.images.front"]
    assert front["dtype"] == ("video" if videos else "image") and front["shape"] == [32, 48, 3]
    assert info["features"]["observation.state"]["names"] == ["x", "y", "z"]
    assert info["features"]["count"]["shape"] == [1] and info["features"]["grid"]["shape"] == [2, 2]
    assert "obs.note" not in info["features"] and writer.skipped == {}  # explicit mapping: nothing else
    stats = json.loads((tmp_path / "ds/meta/stats.json").read_text())
    assert np.asarray(stats["observation.images.front"]["mean"]).shape == (3, 1, 1)
    assert stats["observation.state"]["count"] == [15] and stats["observation.state"]["max"] == [23.0] * 3
    episodes = pq.read_table(tmp_path / "ds/meta/episodes/chunk-000/file-000.parquet").to_pydict()
    assert episodes["tasks"] == [["pick"], ["place"], ["pick"]] and episodes["length"] == [5, 6, 4]
    assert episodes["stats/observation.state/mean"][1] == [12.5] * 3
    if videos:
        assert episodes["videos/observation.images.front/from_timestamp"] == [0.0, 0.5, 1.1]
    assert not (tmp_path / "ds/.unienv-staging").exists()


def test_files_rotate_across_chunks(tmp_path):
    write(tmp_path / "ds", episodes=(3, 3, 3, 3, 3), tasks=("a", "b", "c", "d", "e"), chunks_size=2,
          data_files_size_in_mb=1e-6, video_files_size_in_mb=1e-6)
    files = sorted(str(p.relative_to(tmp_path / "ds")) for p in (tmp_path / "ds").rglob("*") if p.suffix in (".parquet", ".mp4"))
    assert "data/chunk-002/file-000.parquet" in files and "videos/observation.images.front/chunk-002/file-000.mp4" in files
    ds = LeRobotAsUniEnvDataset.from_local(str(tmp_path / "ds"), backend=B)
    data = ds.get_at(np.arange(15))
    np.testing.assert_array_equal(data["observation.state"][:, 0], [10 * e + t for e in range(5) for t in range(3)])
    means = data["observation.images.front"].reshape(15, -1).astype(float).mean(axis=1)
    np.testing.assert_allclose(means, [20 * t for _ in range(5) for t in range(3)], atol=3)


def test_discarding_and_finishing(tmp_path):
    with LeRobotDatasetWriter.create(tmp_path / "ds", SPACE, fps=10, features=NAMES) as writer:
        writer.add_frame(frame(0, 0), task="dropped")
        writer.add_frame(frame(0, 1), task="dropped")
        writer.clear_episode()
        assert writer.episode_frames == 0
        with pytest.raises(ValueError, match="no frames"):
            writer.save_episode()
        batch = {"obs": {"state": np.zeros((4, 3), np.float32), "front": np.zeros((4, 32, 48, 3), np.uint8),
                         "note": np.array(["x"] * 4, dtype=object)},
                 "action": np.zeros((4, 2), np.float32), "count": np.arange(4), "grid": np.zeros((4, 2, 2), np.int64)}
        writer.add_frames(batch, task=["a", "a", "b", "b"])
        writer.save_episode()
        writer.add_frame(frame(1, 0), task="unsaved")  # discarded by finalize
    assert writer.num_episodes == 1 and writer.num_frames == 4
    tasks = pq.read_table(tmp_path / "ds/meta/tasks.parquet").to_pydict()
    assert tasks["task"] == ["a", "b"]  # "dropped" and "unsaved" never made it
    with pytest.raises(RuntimeError, match="finalized"):
        writer.add_frame(frame(0, 0), task="a")
    with pytest.raises(FileExistsError):
        LeRobotDatasetWriter.create(tmp_path / "ds", SPACE, fps=10)


def test_feature_naming_and_errors():
    derived = lerobot_features(SPACE)
    assert set(derived) == {"obs.state", "obs.front", "action", "count", "grid"}  # dots; the text leaf is skipped
    assert lerobot_features(SPACE, videos=False)["obs.front"]["dtype"] == "image"
    with pytest.raises(KeyError, match="not a leaf"):
        lerobot_features(SPACE, features={"x": "obs/nope"})
    with pytest.raises(ValueError, match="column every LeRobot dataset has"):
        lerobot_features(SPACE, features={"timestamp": "count"})
    with pytest.raises(ValueError, match="text"):
        lerobot_features(SPACE, features={"note": "obs/note"})
    odd = DictSpace(B, {"cam": BoxSpace(B, 0, 255, np.uint8, shape=(17, 32, 3), metadata={"kind": "image"})})
    with pytest.raises(ValueError, match="even height and width"):
        lerobot_features(odd, features={"cam": "cam"})
    assert lerobot_features(odd, features={"cam": "cam"}, videos=False)["cam"]["dtype"] == "image"
    with pytest.raises(ValueError, match="fps"):
        LeRobotDatasetWriter.create("/tmp/never", SPACE, fps=0)


def test_export_a_replay_buffer_with_episode_records(tmp_path):
    from unienv_data.replay_buffer import ReplayBuffer
    from unienv_data.storages.hdf5 import HDF5Storage

    step = DictSpace(B, {"state": BoxSpace(B, -100, 100, np.float32, shape=(3,)),
                         "front": BoxSpace(B, 0, 255, np.uint8, shape=(32, 48, 3), metadata={"kind": "image"})})
    rb = ReplayBuffer.create(HDF5Storage, step, cache_path=str(tmp_path / "rb"),
                             episode_space=DictSpace(B, {"instruction": TextSpace(B, 32)}),
                             episode_storage_cls=HDF5Storage)
    for episode, (length, text) in enumerate([(4, "open the drawer"), (3, "close it")]):
        rb.extend({"state": np.tile(np.arange(length, dtype=np.float32)[:, None], 3) + 10 * episode,
                   "front": np.zeros((length, 32, 48, 3), np.uint8)}, episode_data={"instruction": text})
    writer = export_lerobot(rb, tmp_path / "ds", fps=5, task=lambda record: record["instruction"], batch_frames=2)
    ds = LeRobotAsUniEnvDataset.from_local(str(tmp_path / "ds"), backend=B)
    assert ds.get_episode_segments() == [(0, 4, 0), (4, 7, 1)] and writer.fps == 5
    assert [ds.get_episode_data(i)["task"] for i in range(2)] == ["open the drawer", "close it"]
    np.testing.assert_array_equal(ds.get_at(np.arange(7))["state"][:, 0], [0, 1, 2, 3, 10, 11, 12])
    with pytest.raises(ValueError, match="pass episodes"):
        export_lerobot(ReplayBuffer.create(HDF5Storage, step, cache_path=str(tmp_path / "plain")), tmp_path / "x", fps=5)


def test_export_a_lerobot_dataset_round_trips(tmp_path):
    write(tmp_path / "first")
    source = LeRobotAsUniEnvDataset.from_local(str(tmp_path / "first"), backend=B)
    export_lerobot(source, tmp_path / "second", fps=10)  # its own names, tasks and episodes; standard columns skipped
    copy = LeRobotAsUniEnvDataset.from_local(str(tmp_path / "second"), backend=B)
    assert copy.get_episode_segments() == source.get_episode_segments()
    assert [copy.get_episode_data(i)["task"] for i in range(3)] == ["pick", "place", "pick"]
    a, b = source.get_at(np.arange(15)), copy.get_at(np.arange(15))
    for key in ("observation.state", "action", "count", "grid", "index", "frame_index"):
        np.testing.assert_array_equal(a[key], b[key])


def test_push_to_hub(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    class FakeApi:
        def __init__(self, token=None):
            calls.append(("token", token))

        def create_repo(self, repo_id, **kwargs):
            calls.append(("create_repo", repo_id, kwargs["repo_type"], kwargs["private"]))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs["repo_id"], kwargs["repo_type"], sorted(
                p.name for p in Path(kwargs["folder_path"]).iterdir())))

        def create_tag(self, repo_id, **kwargs):
            calls.append(("create_tag", repo_id, kwargs["tag"]))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    writer = write(tmp_path / "ds")
    url = writer.push_to_hub("me/pick-place", private=True, token="t")
    assert url == "https://huggingface.co/datasets/me/pick-place"
    assert ("create_repo", "me/pick-place", "dataset", True) in calls
    assert ("upload_folder", "me/pick-place", "dataset", ["README.md", "data", "meta", "videos"]) in calls
    assert ("create_tag", "me/pick-place", "v3.0") in calls
    card = (tmp_path / "ds/README.md").read_text()
    assert "- LeRobot" in card and "data/*/*.parquet" in card
    unfinished = LeRobotDatasetWriter.create(tmp_path / "other", SPACE, fps=10, features=NAMES)
    with pytest.raises(RuntimeError, match="finalize"):
        unfinished.push_to_hub("me/x")


def test_lerobot_reads_what_unienv_writes(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    write(tmp_path / "ds")
    ds = LeRobotDataset("local/written", root=tmp_path / "ds", video_backend="pyav")  # torchcodec builds vary
    assert len(ds) == 15 and ds.num_episodes == 3 and list(ds.meta.tasks.index) == ["pick", "place"]
    item = ds[7]  # episode 1, frame 2
    assert item["task"] == "place" and int(item["episode_index"]) == 1 and int(item["frame_index"]) == 2
    np.testing.assert_allclose(item["observation.state"].numpy(), [12, 12, 12])
    assert abs(float(item["observation.images.front"].mean()) * 255 - 40) < 3


def test_auto_backend_falls_back_to_pyav_when_torchcodec_cannot_load(tmp_path, monkeypatch):
    import importlib.util as iu
    import unienv_data.integrations.lerobot as module
    import unienv_data.storages.video_storage as storage

    class Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Could not load libtorchcodec.")

    monkeypatch.setattr(storage, "TorchCodecVideoReader", Broken)
    monkeypatch.setattr(module, "_torchcodec_usable", None)
    real_find_spec = iu.find_spec
    monkeypatch.setattr(iu, "find_spec", lambda name, *a: object() if name == "torchcodec" else real_find_spec(name, *a))
    write(tmp_path / "ds")
    ds = LeRobotAsUniEnvDataset.from_local(str(tmp_path / "ds"), backend=B)
    with pytest.warns(RuntimeWarning, match="PyAV"):
        first = ds.get_at(np.array([3]))["observation.images.front"]
    assert abs(float(first.mean()) - 60) < 3 and module._torchcodec_usable is False
    ds.get_at(np.array([4]))  # settled on PyAV: no second attempt, no second warning
    explicit = LeRobotAsUniEnvDataset.from_local(str(tmp_path / "ds"), backend=B, video_backend="torchcodec")
    with pytest.raises(RuntimeError, match="libtorchcodec"):
        explicit.get_at(np.array([0]))
