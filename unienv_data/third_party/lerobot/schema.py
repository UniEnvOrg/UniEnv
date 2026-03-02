from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import os
import json

class LeRobotVersion(Enum):
    V2_0 = "v2.0"
    V2_1 = "v2.1"
    V3_0 = "v3.0"

@dataclass
class LeRobotVideoInfo:
    video_codec: Optional[str] = None
    pixel_format: Optional[str] = None
    fps: Optional[float] = None
    is_depth: bool = False

@dataclass
class LeRobotFeature:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    names: Optional[Any] = None
    video_info: Optional[LeRobotVideoInfo] = None

    @property
    def is_video(self) -> bool:
        return self.dtype == "video"

    @property
    def is_image(self) -> bool:
        return self.dtype == "image"

@dataclass
class LeRobotSchema:
    codebase_version: LeRobotVersion
    robot_type: str
    fps: int
    total_episodes: int
    total_frames: int
    total_tasks: int
    chunks_size: int
    features: Dict[str, LeRobotFeature]
    splits: Optional[Dict[str, str]] = None
    data_path: Optional[str] = None
    video_path: Optional[str] = None
    total_chunks: Optional[int] = None
    encoding: Optional[Dict[str, Any]] = None

    @property
    def video_features(self) -> Dict[str, LeRobotFeature]:
        return {k: v for k, v in self.features.items() if v.is_video}

    @property
    def image_features(self) -> Dict[str, LeRobotFeature]:
        return {k: v for k, v in self.features.items() if v.is_image}

    @property
    def tabular_features(self) -> Dict[str, LeRobotFeature]:
        return {k: v for k, v in self.features.items() if not v.is_video and not v.is_image}


def parse_info_json(dataset_dir: str) -> LeRobotSchema:
    info_path = os.path.join(dataset_dir, "meta", "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    version = LeRobotVersion(info["codebase_version"])

    features: Dict[str, LeRobotFeature] = {}
    for feat_name, feat_def in info.get("features", {}).items():
        video_info = None
        if feat_def.get("dtype") == "video" and "video_info" in feat_def:
            vi = feat_def["video_info"]
            video_info = LeRobotVideoInfo(
                video_codec=vi.get("video.codec"),
                pixel_format=vi.get("video.pix_fmt"),
                fps=vi.get("video.fps"),
                is_depth=vi.get("video.is_depth_map", False),
            )
        features[feat_name] = LeRobotFeature(
            name=feat_name,
            dtype=feat_def["dtype"],
            shape=tuple(feat_def["shape"]),
            names=feat_def.get("names"),
            video_info=video_info,
        )

    return LeRobotSchema(
        codebase_version=version,
        robot_type=info.get("robot_type", "unknown"),
        fps=info["fps"],
        total_episodes=info["total_episodes"],
        total_frames=info["total_frames"],
        total_tasks=info.get("total_tasks", 0),
        chunks_size=info.get("chunks_size", 1000),
        features=features,
        splits=info.get("splits"),
        data_path=info.get("data_path"),
        video_path=info.get("video_path"),
        total_chunks=info.get("total_chunks"),
        encoding=info.get("encoding"),
    )
