"""Video wrappers pick and convert image leaves from ``Space.metadata`` (and fall back without it)."""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.env_base.env import Env
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.wrapper.video_record import (
    EpisodeRenderStackWrapper,
    EpisodeVideoWrapper,
    EpisodeWandbVideoWrapper,
    image_specs_from_space,
    image_to_hwc_rgb,
)

H, W = 4, 6


def rgb_image(seed: int) -> np.ndarray:
    """An (H, W, 3) RGB image whose channels differ, so channel swaps are visible."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)


def observation(seed: int) -> Dict[str, Any]:
    rgb = rgb_image(seed)
    return {
        "cameras": {
            "front": rgb,                                        # HWC RGB
            "wrist": np.ascontiguousarray(rgb[..., ::-1].transpose(2, 0, 1)),  # CHW BGR of the same image
        },
        "depth": rgb[..., 0].astype(np.float32) / 255.0,         # not an image for the recorder
        "state": np.full((7,), seed, np.float32),
    }


def observation_space(with_metadata: bool) -> DictSpace:
    def md(**kwargs):
        return kwargs if with_metadata else None
    return DictSpace(B, {
        "cameras": DictSpace(B, {
            "front": BoxSpace(B, 0, 255, np.uint8, shape=(H, W, 3), metadata=md(kind="image", format="rgb", layout="HWC")),
            "wrist": BoxSpace(B, 0, 255, np.uint8, shape=(3, H, W), metadata=md(kind="image", format="bgr", layout="CHW")),
        }),
        "depth": BoxSpace(B, 0.0, 1.0, np.float32, shape=(H, W), metadata=md(kind="depth", units="m")),
        "state": BoxSpace(B, -np.inf, np.inf, np.float32, shape=(7,), metadata=md(kind="proprio")),
    })


class ObsEnv(Env):
    """Returns ``observation(i)`` at step ``i``; ``render()`` returns the same observation dict."""
    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"
    render_fps = 5
    backend = B
    device = None
    batch_size = None
    action_space = None
    context_space = None

    def __init__(self, with_metadata: bool = True):
        self.observation_space = observation_space(with_metadata)
        self.t = 0

    def reset(self, *, mask=None, seed=None, **kwargs):
        self.t = 0
        return None, observation(self.t), {}

    def step(self, action):
        self.t += 1
        return observation(self.t), 0.0, False, False, {}

    def render(self):
        return observation(self.t)

    def close(self):
        pass


@pytest.fixture
def fake_wandb(monkeypatch):
    module = types.ModuleType("wandb")
    module.logged = []

    class Video:
        def __init__(self, data, fps, format):
            self.data, self.fps, self.format = data, fps, format

    module.Video = Video
    module.log = lambda data, **kwargs: module.logged.append((data, kwargs))
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def _run_episode(wrapper, steps: int = 2) -> None:
    wrapper.reset()
    for _ in range(steps):
        wrapper.step(None)
    wrapper.close()


def _expected_video(steps: int = 2) -> np.ndarray:
    return np.stack([rgb_image(t) for t in range(steps + 1)]).transpose(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Metadata path
# ---------------------------------------------------------------------------

def test_observation_frames_use_metadata_to_select_and_convert(fake_wandb):
    wrapper = EpisodeWandbVideoWrapper(ObsEnv(), wandb_log_key="video", frame_source="observation", log_wandb_episode_id=False)
    _run_episode(wrapper)
    (data, _), = fake_wandb.logged
    assert set(data) == {"video/cameras.front", "video/cameras.wrist"}
    # The CHW / BGR wrist camera is recorded as the same RGB video as the front camera
    np.testing.assert_array_equal(data["video/cameras.front"].data, _expected_video())
    np.testing.assert_array_equal(data["video/cameras.wrist"].data, _expected_video())


def test_render_frames_use_an_explicit_frame_space(fake_wandb):
    wrapper = EpisodeWandbVideoWrapper(
        ObsEnv(), wandb_log_key="video", frame_space=observation_space(True), log_wandb_episode_id=False,
    )
    _run_episode(wrapper)
    (data, _), = fake_wandb.logged
    assert set(data) == {"video/cameras.front", "video/cameras.wrist"}
    np.testing.assert_array_equal(data["video/cameras.wrist"].data, _expected_video())


def test_render_stack_keeps_only_image_leaves_with_metadata():
    wrapper = EpisodeRenderStackWrapper(ObsEnv(), frame_source="observation", nested_separator="/")
    wrapper.reset()
    wrapper.step(None)
    assert set(wrapper.episodic_frames) == {"cameras/front", "cameras/wrist"}
    assert len(wrapper.episodic_frames["cameras/wrist"]) == 2
    # Observation mode leaves render() alone
    assert set(wrapper.render()) == {"cameras", "depth", "state"}


def test_video_wrapper_writes_one_file_per_image_leaf(tmp_path):
    pytest.importorskip("moviepy")
    wrapper = EpisodeVideoWrapper(ObsEnv(), store_dir=tmp_path, format="mp4", frame_source="observation")
    _run_episode(wrapper)
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["episode_0_step_0_key_cameras.front.mp4", "episode_0_step_0_key_cameras.wrist.mp4"]


def test_single_image_frame_space_converts_the_frame(fake_wandb):
    class ChwEnv(ObsEnv):
        def render(self):
            return np.ascontiguousarray(rgb_image(self.t).transpose(2, 0, 1))

    space = BoxSpace(B, 0, 255, np.uint8, shape=(3, H, W), metadata={"kind": "image", "layout": "CHW"})
    wrapper = EpisodeWandbVideoWrapper(ChwEnv(), wandb_log_key="video", frame_space=space, log_wandb_episode_id=False)
    _run_episode(wrapper)
    (data, _), = fake_wandb.logged
    np.testing.assert_array_equal(data["video"].data, _expected_video())


def test_batched_observations_are_indexed_before_conversion(fake_wandb):
    class BatchedEnv(ObsEnv):
        batch_size = 2

        def __init__(self):
            super().__init__()
            from unienv_interface.space.space_utils import batch_utils as sbu
            self.observation_space = sbu.batch_space(self.observation_space, 2)

        def _batched(self, obs):
            return {"cameras": {k: np.stack([v, v]) for k, v in obs["cameras"].items()},
                    "depth": np.stack([obs["depth"]] * 2), "state": np.stack([obs["state"]] * 2)}

        def reset(self, *, mask=None, seed=None, **kwargs):
            _, obs, info = super().reset()
            return None, self._batched(obs), info

        def step(self, action):
            obs, *_ = super().step(action)
            return self._batched(obs), np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), {}

    wrapper = EpisodeWandbVideoWrapper(
        BatchedEnv(), wandb_log_key="video", frame_source="observation", rendered_index=1, log_wandb_episode_id=False,
    )
    _run_episode(wrapper)
    (data, _), = fake_wandb.logged
    np.testing.assert_array_equal(data["video/cameras.wrist"].data, _expected_video())


# ---------------------------------------------------------------------------
# Fallback without metadata
# ---------------------------------------------------------------------------

def test_render_frames_without_metadata_keep_current_behaviour():
    wrapper = EpisodeRenderStackWrapper(ObsEnv(with_metadata=False), frame_space=observation_space(False))
    wrapper.reset()
    assert set(wrapper.episodic_frames) == {"cameras.front", "cameras.wrist", "depth", "state"}
    with pytest.raises(AssertionError, match="Only RGB frames"):
        wrapper._frame_sequence_to_numpy_rgb(wrapper.episodic_frames["cameras.wrist"], "cameras.wrist")


def test_observation_frames_without_metadata_record_hwc_rgb_leaves(fake_wandb):
    wrapper = EpisodeWandbVideoWrapper(
        ObsEnv(with_metadata=False), wandb_log_key="video", frame_source="observation", log_wandb_episode_id=False,
    )
    _run_episode(wrapper)
    (data, _), = fake_wandb.logged
    # Only the (H, W, 3) leaf looks like an image; nothing is converted
    assert set(data) == {"video/cameras.front"}
    np.testing.assert_array_equal(data["video/cameras.front"].data, _expected_video())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_image_specs_from_space():
    assert image_specs_from_space(None) is None
    assert image_specs_from_space(observation_space(False)) is None
    # Metadata without any ``kind`` is no description either
    assert image_specs_from_space(BoxSpace(B, 0, 1, np.float32, shape=(2,), metadata={"units": "m"})) is None
    assert image_specs_from_space(observation_space(True), "/") == {
        "cameras/front": ("HWC", "rgb"),
        "cameras/wrist": ("CHW", "bgr"),
    }
    assert image_specs_from_space(BoxSpace(B, 0, 255, np.uint8, shape=(H, W, 3), metadata={"kind": "image"})) == {None: (None, None)}


@pytest.mark.parametrize("layout,fmt,make", [
    (None, None, lambda rgb: rgb),
    ("HWC", "rgb", lambda rgb: rgb),
    ("hwc", "BGR", lambda rgb: rgb[..., ::-1]),
    ("CHW", "rgb", lambda rgb: rgb.transpose(2, 0, 1)),
    ("CHW", "bgr", lambda rgb: rgb[..., ::-1].transpose(2, 0, 1)),
    ("HWC", "rgba", lambda rgb: np.concatenate([rgb, np.full((H, W, 1), 255, np.uint8)], -1)),
    ("HWC", "bgra", lambda rgb: np.concatenate([rgb[..., ::-1], np.full((H, W, 1), 255, np.uint8)], -1)),
])
def test_image_to_hwc_rgb(layout, fmt, make):
    rgb = rgb_image(3)
    converted = image_to_hwc_rgb(make(rgb), layout, fmt)
    np.testing.assert_array_equal(converted, rgb)
    assert converted.flags["C_CONTIGUOUS"]


def test_image_to_hwc_rgb_expands_gray():
    gray = rgb_image(4)[..., 0]
    expected = np.repeat(gray[..., None], 3, -1)
    np.testing.assert_array_equal(image_to_hwc_rgb(gray, None, "gray"), expected)
    np.testing.assert_array_equal(image_to_hwc_rgb(gray[..., None], "HWC", "gray"), expected)
    np.testing.assert_array_equal(image_to_hwc_rgb(gray[None], "CHW", "gray"), expected)
