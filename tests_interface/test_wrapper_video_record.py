"""Tests for mapping-aware video recording wrappers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.env_base.env import Env
from unienv_interface.wrapper.video_record import (
    EpisodeRenderStackWrapper,
    EpisodeVideoWrapper,
)


def make_rgb_frame(value: int, shape: tuple[int, int, int] = (16, 16, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def make_batched_rgb_frame(value: int, batch_size: int = 2) -> np.ndarray:
    return np.stack([make_rgb_frame(value + i) for i in range(batch_size)], axis=0)


def make_nested_render(value: int) -> dict[str, Any]:
    return {
        "camera": {
            "left": make_rgb_frame(value),
            "right": make_rgb_frame(value + 1),
        },
        "overview": make_rgb_frame(value + 2),
    }


def make_batched_nested_render(value: int, batch_size: int = 2) -> dict[str, Any]:
    return {
        "camera": {
            "left": make_batched_rgb_frame(value, batch_size=batch_size),
            "right": make_batched_rgb_frame(value + 10, batch_size=batch_size),
        },
        "overview": make_batched_rgb_frame(value + 20, batch_size=batch_size),
    }


class DummyRenderEnv(
    Env[
        np.ndarray,
        None,
        Any,
        Any,
        Any,
        Any,
        np.dtype,
        np.random.Generator,
    ]
):
    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"
    render_fps = 12
    backend = NumpyComputeBackend
    device = None
    batch_size = None
    action_space = None
    observation_space = None
    context_space = None

    def __init__(self, render_frames: list[Any]):
        self._render_frames = list(render_frames)
        if len(self._render_frames) == 0:
            raise ValueError("render_frames must not be empty")
        self._render_index = 0

    def _next_render(self) -> Any:
        idx = min(self._render_index, len(self._render_frames) - 1)
        self._render_index += 1
        return self._render_frames[idx]

    def step(self, action: Any):
        return None, 0.0, False, False, {}

    def reset(self, *, mask=None, seed=None, **kwargs):
        return None, None, {}

    def render(self):
        return self._next_render()

    def close(self):
        pass


class DummyBatchedDoneEnv(DummyRenderEnv):
    batch_size = 2

    def __init__(
        self,
        render_frames: list[Any],
        terminations: list[np.ndarray],
        truncations: list[np.ndarray] | None = None,
    ):
        super().__init__(render_frames=render_frames)
        self._terminations = list(terminations)
        if truncations is None:
            self._truncations = [np.array([False, False])] * len(self._terminations)
        else:
            self._truncations = list(truncations)
        self._done_index = 0

    def step(self, action: Any):
        idx = min(self._done_index, len(self._terminations) - 1)
        self._done_index += 1
        return None, 0.0, self._terminations[idx], self._truncations[idx], {}


def test_render_stack_flattens_mapping_and_accumulates_history():
    env = DummyRenderEnv([make_nested_render(10), make_nested_render(20)])
    wrapper = EpisodeRenderStackWrapper(env)

    wrapper.reset()

    assert isinstance(wrapper.episodic_frames, dict)
    assert set(wrapper.episodic_frames.keys()) == {
        "camera.left",
        "camera.right",
        "overview",
    }
    np.testing.assert_array_equal(wrapper.episodic_frames["camera.left"][0], make_rgb_frame(10))
    np.testing.assert_array_equal(wrapper.episodic_frames["camera.right"][0], make_rgb_frame(11))
    np.testing.assert_array_equal(wrapper.episodic_frames["overview"][0], make_rgb_frame(12))

    wrapper.step(None)
    assert isinstance(wrapper.episodic_frames, dict)
    assert len(wrapper.episodic_frames["camera.left"]) == 2
    np.testing.assert_array_equal(wrapper.episodic_frames["camera.left"][1], make_rgb_frame(20))
    np.testing.assert_array_equal(wrapper.episodic_frames["camera.right"][1], make_rgb_frame(21))
    np.testing.assert_array_equal(wrapper.episodic_frames["overview"][1], make_rgb_frame(22))


def test_render_stack_uses_custom_nested_separator():
    env = DummyRenderEnv([make_nested_render(10)])
    wrapper = EpisodeRenderStackWrapper(env, nested_separator="/")

    wrapper.reset()

    assert isinstance(wrapper.episodic_frames, dict)
    assert set(wrapper.episodic_frames.keys()) == {
        "camera/left",
        "camera/right",
        "overview",
    }


def test_render_stack_rendered_index_indexes_single_batched_render():
    env = DummyRenderEnv([make_batched_rgb_frame(100), make_batched_rgb_frame(200)])
    wrapper = EpisodeRenderStackWrapper(env, rendered_index=1)

    wrapper.reset()
    assert isinstance(wrapper.episodic_frames, list)
    np.testing.assert_array_equal(wrapper.episodic_frames[0], make_rgb_frame(101))

    wrapper.step(None)
    assert isinstance(wrapper.episodic_frames, list)
    np.testing.assert_array_equal(wrapper.episodic_frames[1], make_rgb_frame(201))


def test_render_stack_rendered_index_indexes_mapping_batched_render():
    env = DummyRenderEnv([make_batched_nested_render(30), make_batched_nested_render(40)])
    wrapper = EpisodeRenderStackWrapper(env, rendered_index=1)

    wrapper.reset()
    assert isinstance(wrapper.episodic_frames, dict)
    np.testing.assert_array_equal(wrapper.episodic_frames["camera.left"][0], make_rgb_frame(31))
    np.testing.assert_array_equal(wrapper.episodic_frames["camera.right"][0], make_rgb_frame(41))
    np.testing.assert_array_equal(wrapper.episodic_frames["overview"][0], make_rgb_frame(51))

    wrapper.step(None)
    assert isinstance(wrapper.episodic_frames, dict)
    np.testing.assert_array_equal(wrapper.episodic_frames["camera.left"][1], make_rgb_frame(41))
    np.testing.assert_array_equal(wrapper.episodic_frames["camera.right"][1], make_rgb_frame(51))
    np.testing.assert_array_equal(wrapper.episodic_frames["overview"][1], make_rgb_frame(61))


def test_render_stack_done_index_controls_done_selection():
    done = np.array([False, True])

    env_select_done = DummyBatchedDoneEnv(
        render_frames=[make_rgb_frame(1), make_rgb_frame(2)],
        terminations=[done],
    )
    wrapper_select_done = EpisodeRenderStackWrapper(env_select_done, done_index=1)
    wrapper_select_done.reset(mask=np.array([True, True]))
    wrapper_select_done.step(None)
    assert wrapper_select_done._has_post_episode is True

    env_select_not_done = DummyBatchedDoneEnv(
        render_frames=[make_rgb_frame(10), make_rgb_frame(20)],
        terminations=[done],
    )
    wrapper_select_not_done = EpisodeRenderStackWrapper(env_select_not_done, done_index=0)
    wrapper_select_not_done.reset(mask=np.array([True, True]))
    wrapper_select_not_done.step(None)
    assert wrapper_select_not_done._has_post_episode is False


def test_render_stack_fails_on_mapping_key_drift():
    env = DummyRenderEnv(
        [
            {"camera": {"left": make_rgb_frame(1), "right": make_rgb_frame(2)}},
            {"camera": {"left": make_rgb_frame(3)}},
        ]
    )
    wrapper = EpisodeRenderStackWrapper(env)

    wrapper.reset()
    with pytest.raises(ValueError, match="Flattened render keys changed within an episode"):
        wrapper.step(None)


def test_render_stack_fails_on_render_type_drift():
    env = DummyRenderEnv([make_nested_render(1), make_rgb_frame(2)])
    wrapper = EpisodeRenderStackWrapper(env)

    wrapper.reset()
    with pytest.raises(ValueError, match="Render frame type changed within an episode"):
        wrapper.step(None)


def test_episode_video_wrapper_writes_video_per_mapping_key(tmp_path):
    env = DummyRenderEnv([make_nested_render(1), make_nested_render(2), make_nested_render(3)])
    wrapper = EpisodeVideoWrapper(env, store_dir=tmp_path, format="mp4")

    wrapper.reset()
    wrapper.step(None)
    wrapper.reset()  # flush episode 0

    written_paths = sorted(tmp_path.glob("episode_0_step_0_key_*.mp4"))
    assert len(written_paths) == 3
    basenames = [path.name for path in written_paths]
    for path, name in zip(written_paths, basenames):
        assert name.startswith("episode_0_step_0_key_")
        assert name.endswith(".mp4")
        assert path.stat().st_size > 0


def test_episode_video_wrapper_keeps_single_frame_behavior(tmp_path):
    env = DummyRenderEnv([make_rgb_frame(1), make_rgb_frame(2), make_rgb_frame(3)])
    wrapper = EpisodeVideoWrapper(env, store_dir=tmp_path, format="mp4")

    wrapper.reset()
    wrapper.step(None)
    wrapper.reset()  # flush episode 0

    written_path = tmp_path / "episode_0_step_0.mp4"
    assert written_path.exists()
    assert written_path.stat().st_size > 0
