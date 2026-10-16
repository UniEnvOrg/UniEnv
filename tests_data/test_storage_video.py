"""VideoStorage specifics: codec selection, decoding paths and segment handling."""

import numpy as np
import pytest

pytest.importorskip("av")

from unienv_data.storages import video_storage as video_storage_module
from unienv_data.storages.video_storage import VideoStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace

from tests_common.storage_video import (
    LOSSLESS_DEPTH_KWARGS,
    LOSSLESS_RGB_KWARGS,
    clean_video_codec_cache,  # noqa: F401  (fixture)
    reset_video_codec_cache,
)

B = NumpyComputeBackend

pytestmark = pytest.mark.usefixtures("clean_video_codec_cache")


def _rgb_space(h: int = 8, w: int = 8) -> BoxSpace:
    return BoxSpace(B, 0, 255, np.uint8, shape=(h, w, 3))


def _frames(n: int, seed: int = 0, h: int = 8, w: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


@pytest.fixture
def fake_encoders(monkeypatch):
    available = {"av1", "hevc", "h264", "mpeg4", "vp8", "h264_nvenc", "hevc_qsv", "vp8_qsv", "mpeg4_amf"}
    monkeypatch.setattr(VideoStorage, "_encoder_available", staticmethod(lambda name: name in available))
    return available


def test_auto_codec_prefers_codec_order_before_hardware_suffix(fake_encoders):
    # hevc ranks above h264 in the codec preference list, so hevc_qsv wins over h264_nvenc,
    # and the lowest-priority candidate (mpeg4_amf / vp8_qsv) must not win.
    assert VideoStorage.get_auto_codec() == "hevc_qsv"


def test_auto_codec_respects_base_and_caches_per_base(fake_encoders):
    assert VideoStorage.get_auto_codec(base="h264") == "h264_nvenc"
    assert VideoStorage.get_auto_codec(base="mpeg4") == "mpeg4_amf"
    assert VideoStorage.get_auto_codec(base="vp8") == "vp8_qsv"
    assert VideoStorage.get_auto_codec() == "hevc_qsv"
    # cached results stay per base
    assert VideoStorage.get_auto_codec(base="h264") == "h264_nvenc"


def test_auto_codec_falls_back_to_software_codec(monkeypatch):
    monkeypatch.setattr(VideoStorage, "_encoder_available", staticmethod(lambda name: name in {"h264", "vp9"}))
    assert VideoStorage.get_auto_codec() == "h264"
    assert VideoStorage.get_auto_codec(base="vp9") == "vp9"


def test_auto_codec_raises_without_any_encoder(monkeypatch):
    monkeypatch.setattr(VideoStorage, "_encoder_available", staticmethod(lambda name: False))
    with pytest.raises(RuntimeError, match="No suitable video codec"):
        VideoStorage.get_auto_codec()


def test_encoder_available_rejects_decoder_only_and_unknown_names():
    assert not VideoStorage._encoder_available("definitely-not-a-codec")
    if "ffv1" in video_storage_module.codecs_available:
        assert VideoStorage._encoder_available("ffv1")


def test_reset_helper_clears_cache(fake_encoders):
    VideoStorage.get_auto_codec()
    reset_video_codec_cache()
    fake_encoders.discard("hevc_qsv")
    assert VideoStorage.get_auto_codec() == "h264_nvenc"
