from typing import Optional
import os

from .schema import LeRobotSchema, LeRobotVersion


def get_data_file_path(
    dataset_dir: str,
    schema: LeRobotSchema,
    episode_index: int,
    file_index: Optional[int] = None,
    chunk_index: Optional[int] = None,
) -> str:
    """
    Get the Parquet file path for a given episode.

    v2.0/v2.1: data/chunk-{CCC}/episode_{EEEEEE}.parquet (one file per episode)
    v3.0: uses data_path template from info.json (may pack multiple episodes per file)

    For v3.0, file_index and chunk_index should be provided from episode metadata.
    If not provided, falls back to episode_index-based calculation.
    """
    if schema.codebase_version == LeRobotVersion.V3_0 and schema.data_path is not None:
        if chunk_index is None:
            chunk_index = episode_index // schema.chunks_size
        if file_index is None:
            file_index = episode_index
        return os.path.join(dataset_dir, schema.data_path.format(
            chunk_index=chunk_index,
            file_index=file_index,
            episode_chunk=chunk_index,
            episode_index=episode_index,
        ))
    else:
        chunk_index = episode_index // schema.chunks_size
        return os.path.join(
            dataset_dir, "data",
            f"chunk-{chunk_index:03d}",
            f"episode_{episode_index:06d}.parquet",
        )


def get_video_file_path(
    dataset_dir: str,
    schema: LeRobotSchema,
    feature_name: str,
    episode_index: int,
    file_index: Optional[int] = None,
    chunk_index: Optional[int] = None,
) -> str:
    """
    Get the MP4 file path for a given video feature and episode.

    v2.0/v2.1: videos/chunk-{CCC}/{feature_name}/episode_{EEEEEE}.mp4
    v3.0: uses video_path template from info.json

    For v3.0, file_index and chunk_index should be provided from episode metadata.
    If not provided, falls back to episode_index-based calculation.
    """
    if schema.codebase_version == LeRobotVersion.V3_0 and schema.video_path is not None:
        if chunk_index is None:
            chunk_index = episode_index // schema.chunks_size
        if file_index is None:
            file_index = episode_index
        return os.path.join(dataset_dir, schema.video_path.format(
            video_key=feature_name,
            chunk_index=chunk_index,
            file_index=file_index,
            episode_chunk=chunk_index,
            episode_index=episode_index,
        ))
    else:
        chunk_index = episode_index // schema.chunks_size
        return os.path.join(
            dataset_dir, "videos",
            f"chunk-{chunk_index:03d}",
            feature_name,
            f"episode_{episode_index:06d}.mp4",
        )
