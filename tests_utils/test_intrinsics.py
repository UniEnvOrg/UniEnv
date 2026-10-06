"""Synthetic intrinsic-calibration tests.

Detections are constructed directly from the board's object points projected
with a known camera matrix, which keeps the test deterministic: no rendering and
no detector run is involved.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from unienv_utils.calibration.board import CharucoBoardConfig, make_board  # noqa: E402
from unienv_utils.calibration.detect import CharucoDetection  # noqa: E402
from unienv_utils.calibration.intrinsics import (  # noqa: E402
    calibrate_intrinsics_charuco,
)

IMAGE_SIZE = (1280, 960)
GROUND_TRUTH_K = np.array(
    [[800.0, 0.0, 640.0], [0.0, 810.0, 480.0], [0.0, 0.0, 1.0]]
)
RNG = np.random.default_rng(2026)


def _synthetic_detections(
    cfg: CharucoBoardConfig,
    n_views: int = 12,
    noise_px: float = 0.2,
) -> list[CharucoDetection]:
    """Build ``n_views`` detections of the board seen from random viewpoints."""
    object_points = make_board(cfg).getChessboardCorners().reshape(-1, 3)
    ids = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1)
    detections = []
    for _ in range(n_views):
        rvec = RNG.uniform(-0.4, 0.4, 3)
        tvec = np.array([
            RNG.uniform(-0.2, 0.2),
            RNG.uniform(-0.15, 0.15),
            RNG.uniform(0.4, 0.7),
        ])
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, GROUND_TRUTH_K, np.zeros(5)
        )
        corners = projected + RNG.normal(0.0, noise_px, projected.shape)
        detections.append(
            CharucoDetection(
                corners=corners.astype(np.float32), ids=ids, n_markers=56
            )
        )
    return detections


def test_calibration_recovers_known_intrinsics() -> None:
    cfg = CharucoBoardConfig()
    result = calibrate_intrinsics_charuco(
        _synthetic_detections(cfg), cfg, IMAGE_SIZE
    )

    assert result.n_views == 12
    assert result.K.shape == (3, 3)
    assert result.image_size == IMAGE_SIZE
    assert result.dist.shape[0] >= 4
    assert len(result.per_view_errors) == 12
    assert result.mean_reprojection_error_px < 1.0
    assert max(result.per_view_errors) < 2.0

    fx, fy = result.K[0, 0], result.K[1, 1]
    assert abs(fx - GROUND_TRUTH_K[0, 0]) / GROUND_TRUTH_K[0, 0] < 0.02
    assert abs(fy - GROUND_TRUTH_K[1, 1]) / GROUND_TRUTH_K[1, 1] < 0.02
    assert abs(result.K[0, 2] - GROUND_TRUTH_K[0, 2]) < 5.0
    assert abs(result.K[1, 2] - GROUND_TRUTH_K[1, 2]) < 5.0
    assert abs(result.K[0, 1]) < 1.0


def test_calibration_requires_eight_views() -> None:
    cfg = CharucoBoardConfig()
    with pytest.raises(ValueError, match="at least 8 views"):
        calibrate_intrinsics_charuco(_synthetic_detections(cfg, n_views=7), cfg, IMAGE_SIZE)


def test_calibration_validates_image_size() -> None:
    cfg = CharucoBoardConfig()
    detections = _synthetic_detections(cfg)
    with pytest.raises(ValueError, match="image_size must be"):
        calibrate_intrinsics_charuco(detections, cfg, (1280,))
    with pytest.raises(ValueError, match="image_size must be"):
        calibrate_intrinsics_charuco(detections, cfg, (0, 960))


def test_calibration_returns_true_distortion_scale_for_zero_distortion() -> None:
    cfg = CharucoBoardConfig()
    result = calibrate_intrinsics_charuco(
        _synthetic_detections(cfg, noise_px=0.0), cfg, IMAGE_SIZE
    )
    assert np.allclose(result.K, GROUND_TRUTH_K, rtol=1e-3, atol=0.5)
    assert result.mean_reprojection_error_px < 0.05
