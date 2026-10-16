"""Helpers for deterministic, CPU-only VideoStorage tests."""

from __future__ import annotations

from typing import Any, Dict

import pytest

# Lossless RGB settings: ffv1 in a Matroska container round-trips uint8 RGB exactly.
LOSSLESS_RGB_KWARGS: Dict[str, Any] = dict(
    codec="ffv1",
    file_ext="mkv",
    file_pixel_format="bgr0",
    buffer_pixel_format="rgb24",
    hardware_acceleration=None,
    decode_backend="pyav",
)

# Lossless 16-bit depth settings.
LOSSLESS_DEPTH_KWARGS: Dict[str, Any] = dict(
    codec="ffv1",
    file_ext="mkv",
    file_pixel_format="gray16le",
    buffer_pixel_format="gray16le",
    hardware_acceleration=None,
    decode_backend="pyav",
)


def reset_video_codec_cache() -> None:
    """Forget any codec cached by ``VideoStorage.get_auto_codec`` (class-level state)."""
    from unienv_data.storages.video_storage import VideoStorage

    cache = getattr(VideoStorage, "_auto_codec_cache", None)
    if isinstance(cache, dict):
        cache.clear()
    if "_auto_codec" in VideoStorage.__dict__:
        delattr(VideoStorage, "_auto_codec")


@pytest.fixture
def clean_video_codec_cache():
    """Reset the auto-codec cache before and after a test so tests cannot leak state."""
    reset_video_codec_cache()
    yield
    reset_video_codec_cache()
