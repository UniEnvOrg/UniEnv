"""Tests for ChArUco detection and board-pose estimation on synthetic images."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from unienv_utils.calibration import detect as detect_module  # noqa: E402
from unienv_utils.calibration.board import (  # noqa: E402
    CharucoBoardConfig,
    make_board,
    render_board,
)
from unienv_utils.calibration.detect import (  # noqa: E402
    CharucoDetection,
    detect_charuco,
    estimate_board_pose,
)

PIXELS_PER_METER = 5000.0


@pytest.fixture(scope="module")
def cfg() -> CharucoBoardConfig:
    return CharucoBoardConfig()


@pytest.fixture(scope="module")
def board_image(cfg: CharucoBoardConfig) -> np.ndarray:
    return render_board(cfg, PIXELS_PER_METER)


def _warp_into_view(
    cfg: CharucoBoardConfig,
    board_image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Warp the flat rendered board into a synthetic camera view.

    The rendering and the target view are both pictures of the same planar
    board, so the plane homography built from the detected corners and their
    projections is exact.
    """
    detection = detect_charuco(board_image, cfg)
    assert detection is not None
    object_points, image_points = make_board(cfg).matchImagePoints(
        detection.corners, detection.ids
    )
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    homography, mask = cv2.findHomography(
        image_points.reshape(-1, 2), projected.reshape(-1, 2)
    )
    assert mask is not None and bool(mask.all())
    return cv2.warpPerspective(board_image, homography, image_size, borderValue=255)


def test_detect_charuco_finds_rendered_board(
    cfg: CharucoBoardConfig, board_image: np.ndarray
) -> None:
    detection = detect_charuco(board_image, cfg)
    assert detection is not None
    assert detection.n_corners == (cfg.col_count - 1) * (cfg.row_count - 1)
    assert detection.n_markers == cfg.col_count * cfg.row_count // 2
    assert detection.corners.shape == (detection.n_corners, 1, 2)
    assert detection.ids.shape == (detection.n_corners, 1)
    assert detection.ids.dtype == np.int32
    assert detection.corners.dtype == np.float32


def test_detect_charuco_accepts_bgr(
    cfg: CharucoBoardConfig, board_image: np.ndarray
) -> None:
    gray_detection = detect_charuco(board_image, cfg)
    bgr_detection = detect_charuco(cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR), cfg)
    assert bgr_detection is not None and gray_detection is not None
    assert bgr_detection.n_corners == gray_detection.n_corners
    assert np.allclose(bgr_detection.corners, gray_detection.corners, atol=1.0)


def test_detect_charuco_returns_none_without_board(cfg: CharucoBoardConfig) -> None:
    blank = np.full((480, 640), 255, dtype=np.uint8)
    assert detect_charuco(blank, cfg) is None


def test_detect_charuco_reuses_cached_builders(
    cfg: CharucoBoardConfig, board_image: np.ndarray
) -> None:
    first = detect_charuco(board_image, cfg)
    second = detect_charuco(board_image, CharucoBoardConfig())
    assert first is not None and second is not None
    assert np.array_equal(first.corners, second.corners)
    assert np.array_equal(first.ids, second.ids)
    # Equal configs share the cv2 objects instead of rebuilding them per call.
    assert detect_module._get_board(cfg) is detect_module._get_board(
        CharucoBoardConfig()
    )
    assert detect_module._get_detector_parameters() is (
        detect_module._get_detector_parameters()
    )
    assert detect_module._get_dictionary(cfg.aruco_dict_name) is (
        detect_module._get_dictionary(CharucoBoardConfig().aruco_dict_name)
    )


def test_detect_charuco_rejects_bad_input(cfg: CharucoBoardConfig) -> None:
    with pytest.raises(ValueError, match="must be uint8"):
        detect_charuco(np.zeros((64, 64), dtype=np.float32), cfg)
    with pytest.raises(ValueError, match="image must be"):
        detect_charuco(np.zeros((2, 3, 4, 5), dtype=np.uint8), cfg)


def test_estimate_board_pose_recovers_known_placement(
    cfg: CharucoBoardConfig, board_image: np.ndarray
) -> None:
    K = np.array([[600.0, 0.0, 640.0], [0.0, 600.0, 480.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    rvec = np.array([0.25, -0.35, 0.15])
    tvec = np.array([-0.02, 0.03, 0.55])
    warped = _warp_into_view(cfg, board_image, K, dist, rvec, tvec, (1280, 960))

    detection = detect_charuco(warped, cfg)
    assert detection is not None
    pose = estimate_board_pose(detection, cfg, K, dist)
    assert pose is not None
    assert pose.shape == (4, 4)

    expected = np.eye(4)
    expected[:3, :3] = cv2.Rodrigues(rvec)[0]
    expected[:3, 3] = tvec
    # Loose tolerances: 10% of the board distance and a couple of degrees.
    assert np.linalg.norm(pose[:3, 3] - expected[:3, 3]) < 0.1 * tvec[2]
    cos_angle = (np.trace(pose[:3, :3].T @ expected[:3, :3]) - 1.0) / 2.0
    assert np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))) < 2.0


def test_estimate_board_pose_returns_none_for_too_few_corners(
    cfg: CharucoBoardConfig,
) -> None:
    detection = CharucoDetection(
        corners=np.array([[[10.0, 10.0]], [[20.0, 10.0]]], dtype=np.float32),
        ids=np.array([[0], [1]], dtype=np.int32),
        n_markers=1,
    )
    assert estimate_board_pose(detection, cfg, np.eye(3), np.zeros(5)) is None


def test_charuco_detection_validates_shapes() -> None:
    with pytest.raises(ValueError, match="same length"):
        CharucoDetection(
            corners=np.zeros((2, 1, 2), dtype=np.float32),
            ids=np.zeros((3, 1), dtype=np.int32),
            n_markers=1,
        )
    with pytest.raises(ValueError, match="at least one corner"):
        CharucoDetection(
            corners=np.zeros((0, 1, 2), dtype=np.float32),
            ids=np.zeros((0, 1), dtype=np.int32),
            n_markers=0,
        )
