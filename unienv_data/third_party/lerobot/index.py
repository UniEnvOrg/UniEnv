from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import os
import json
import numpy as np

from .schema import LeRobotSchema, LeRobotVersion


@dataclass
class EpisodeMetadata:
    index: int
    length: int
    task_index: int = 0
    task: str = ""
    # v3.0 packed-file metadata (optional)
    data_chunk_index: Optional[int] = None
    data_file_index: Optional[int] = None
    dataset_from_index: Optional[int] = None
    dataset_to_index: Optional[int] = None


class LeRobotIndex:
    """
    Frame-level index for a LeRobot dataset.

    Maps global frame indices (0..total_frames-1) to (episode_index, local_frame_index).
    Uses a cumulative-length array for O(log N) lookup via np.searchsorted.
    """

    def __init__(
        self,
        episodes: List[EpisodeMetadata],
        tasks: Dict[int, str],
    ):
        self.episodes = episodes
        self.tasks = tasks

        lengths = [ep.length for ep in episodes]
        self.cumulative_lengths = np.zeros(len(lengths) + 1, dtype=np.int64)
        self.cumulative_lengths[1:] = np.cumsum(lengths)

    @property
    def total_frames(self) -> int:
        return int(self.cumulative_lengths[-1])

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    def global_index_to_episode_and_local(
        self, global_index: int
    ) -> Tuple[int, int]:
        ep_idx = int(np.searchsorted(self.cumulative_lengths[1:], global_index, side='right'))
        local_idx = global_index - int(self.cumulative_lengths[ep_idx])
        return ep_idx, local_idx

    def global_indices_to_episode_and_local(
        self, global_indices: np.ndarray
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert an array of global frame indices to per-episode groups.

        Returns:
            Dict mapping episode_index -> (local_indices, output_positions)
            where local_indices are frame offsets within the episode
            and output_positions are indices into the original array.
        """
        ep_indices = np.searchsorted(self.cumulative_lengths[1:], global_indices, side='right')
        local_indices = global_indices - self.cumulative_lengths[ep_indices]

        result = {}
        for ep_idx in np.unique(ep_indices):
            mask = ep_indices == ep_idx
            result[int(ep_idx)] = (
                local_indices[mask].astype(np.int64),
                np.where(mask)[0],
            )
        return result

    def episode_frame_range(self, episode_index: int) -> Tuple[int, int]:
        """Return (start_global, end_global_exclusive) for an episode."""
        start = int(self.cumulative_lengths[episode_index])
        end = int(self.cumulative_lengths[episode_index + 1])
        return start, end


def build_index(dataset_dir: str, schema: LeRobotSchema) -> LeRobotIndex:
    tasks = _load_tasks(dataset_dir)
    episodes = _load_episodes(dataset_dir, schema, tasks)
    return LeRobotIndex(episodes=episodes, tasks=tasks)


def _load_tasks(dataset_dir: str) -> Dict[int, str]:
    tasks_path = os.path.join(dataset_dir, "meta", "tasks.jsonl")
    tasks: Dict[int, str] = {}
    if os.path.exists(tasks_path):
        with open(tasks_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    tasks[entry["task_index"]] = entry["task"]
    return tasks


def _load_episodes(
    dataset_dir: str,
    schema: LeRobotSchema,
    tasks: Dict[int, str],
) -> List[EpisodeMetadata]:
    if schema.codebase_version == LeRobotVersion.V3_0:
        return _load_episodes_v3(dataset_dir, schema, tasks)
    else:
        return _load_episodes_v2(dataset_dir, tasks)


def _load_episodes_v2(
    dataset_dir: str,
    tasks: Dict[int, str],
) -> List[EpisodeMetadata]:
    episodes_path = os.path.join(dataset_dir, "meta", "episodes.jsonl")
    episodes: List[EpisodeMetadata] = []
    with open(episodes_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            ep_idx = entry["episode_index"]
            length = entry["length"]

            # task_index can come from different fields depending on dataset
            if "task_index" in entry:
                task_idx = entry["task_index"]
            elif "tasks" in entry:
                task_list = entry["tasks"]
                # tasks field may be a list of strings or indices
                if task_list and isinstance(task_list[0], int):
                    task_idx = task_list[0]
                else:
                    task_idx = 0
            else:
                task_idx = 0

            episodes.append(EpisodeMetadata(
                index=ep_idx,
                length=length,
                task_index=task_idx,
                task=tasks.get(task_idx, ""),
            ))

    episodes.sort(key=lambda e: e.index)
    return episodes


def _load_episodes_v3(
    dataset_dir: str,
    schema: LeRobotSchema,
    tasks: Dict[int, str],
) -> List[EpisodeMetadata]:
    import pyarrow.parquet as pq

    episodes_dir = os.path.join(dataset_dir, "meta", "episodes")
    episodes: List[EpisodeMetadata] = []

    if not os.path.isdir(episodes_dir):
        raise FileNotFoundError(
            f"v3.0 dataset missing meta/episodes/ directory: {episodes_dir}"
        )

    for fname in sorted(os.listdir(episodes_dir)):
        if not fname.endswith(".parquet"):
            continue
        table = pq.read_table(os.path.join(episodes_dir, fname))
        col_names = set(table.column_names)

        for i in range(table.num_rows):
            ep_idx = int(table.column("episode_index")[i].as_py())
            length = int(table.column("length")[i].as_py())
            task_idx = int(table.column("task_index")[i].as_py()) if "task_index" in col_names else 0

            # Extract packed-file metadata if available
            data_chunk_index = None
            data_file_index = None
            dataset_from = None
            dataset_to = None
            if "data/chunk_index" in col_names:
                data_chunk_index = int(table.column("data/chunk_index")[i].as_py())
            if "data/file_index" in col_names:
                data_file_index = int(table.column("data/file_index")[i].as_py())
            if "dataset_from_index" in col_names:
                dataset_from = int(table.column("dataset_from_index")[i].as_py())
            if "dataset_to_index" in col_names:
                dataset_to = int(table.column("dataset_to_index")[i].as_py())

            episodes.append(EpisodeMetadata(
                index=ep_idx,
                length=length,
                task_index=task_idx,
                task=tasks.get(task_idx, ""),
                data_chunk_index=data_chunk_index,
                data_file_index=data_file_index,
                dataset_from_index=dataset_from,
                dataset_to_index=dataset_to,
            ))

    episodes.sort(key=lambda e: e.index)
    return episodes
