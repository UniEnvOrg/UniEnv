"""ChArUco detection and board-pose estimation.

The detector parameters are lifted from the battle-tested tianji teleoperation
stack: they widen the adaptive-threshold search window and lower the minimum
marker perimeter rate so that small, slightly blurred boards (e.g. a distant
board in a wide-FOV camera) are still detected, while ArUco dictionary decoding
keeps the false-positive rate under control.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from unienv_utils.calibration.board import (
    CharucoBoardConfig,
    make_board,
    predefined_dictionary,
)
from unienv_utils.calibration.transforms import T_from_rvec_tvec

__all__ = [
    "CharucoDetection",
    "detect_charuco",
    "estimate_board_pose",
    "make_detector_parameters",
]

# Minimum number of ChArUco corners needed for a pose (SOLVEPNP_ITERATIVE).
_MIN_POSE_CORNERS = 6


@dataclass(frozen=True)
class CharucoDetection:
    """Result of a ChArUco detection.

    Attributes
    ----------
    corners:
        Interpolated chessboard corners, shape ``(N, 1, 2)`` float32, in pixels
        of the (grayscale) input image.
    ids:
        Board corner ids, shape ``(N, 1)`` int32; ``ids[i]`` belongs to
        ``corners[i]``.
    n_markers:
        Number of ArUco markers that were detected and decoded.
    """

    corners: np.ndarray
    ids: np.ndarray
    n_markers: int

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners, dtype=np.float32).reshape(-1, 1, 2)
        ids = np.asarray(self.ids, dtype=np.int32).reshape(-1, 1)
        if len(corners) != len(ids):
            raise ValueError(
                f"corners and ids must have the same length, got {len(corners)} "
                f"and {len(ids)}."
            )
        if len(corners) == 0:
            raise ValueError("A CharucoDetection needs at least one corner.")
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "ids", ids)

    @property
    def n_corners(self) -> int:
        """Number of interpolated ChArUco corners."""
        return int(self.corners.shape[0])


def make_detector_parameters() -> "cv2.aruco.DetectorParameters":
    """Return the tuned ArUco ``DetectorParameters`` used across UniEnv."""
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.003
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 61
    params.adaptiveThreshWinSizeStep = 4
    params.errorCorrectionRate = 0.9
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Return a ``uint8`` grayscale copy of ``image`` (BGR/BGRA/gray accepted)."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    elif arr.ndim != 2:
        raise ValueError(
            f"image must be 2-D gray or 3-D BGR/BGRA, got shape {arr.shape}."
        )
    if arr.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got dtype {arr.dtype}.")
    return np.ascontiguousarray(arr)


def detect_charuco(
    image: np.ndarray,
    cfg: CharucoBoardConfig,
    *,
    min_markers: int = 4,
    min_charuco_corners: int = 6,
) -> CharucoDetection | None:
    """Detect the ChArUco board in ``image``.

    Parameters
    ----------
    image:
        BGR/BGRA or grayscale ``uint8`` image.
    cfg:
        Board specification; the returned ids index into this board.
    min_markers:
        Reject the frame when fewer ArUco markers were decoded.
    min_charuco_corners:
        Reject the frame when fewer ChArUco corners were interpolated.

    Returns
    -------
    CharucoDetection | None
        ``None`` when the board is not visible well enough for calibration.
    """
    gray = _to_gray(image)
    board = make_board(cfg)
    dictionary = predefined_dictionary(cfg.aruco_dict_name)
    params = make_detector_parameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    else:
        # OpenCV < 4.7 free function.
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=params
        )
    if marker_ids is None or len(marker_ids) < min_markers:
        return None

    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
        _retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
    else:
        # OpenCV >= 4.13 removed the free interpolation function.  Supplying
        # markerCorners/markerIds prevents CharucoDetector from rerunning its
        # own, less sensitive, marker detector.
        charuco_detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
            gray, markerCorners=marker_corners, markerIds=marker_ids
        )
    if charuco_ids is None or len(charuco_ids) < min_charuco_corners:
        return None
    return CharucoDetection(
        corners=charuco_corners,
        ids=charuco_ids,
        n_markers=int(len(marker_ids)),
    )


def estimate_board_pose(
    detection: CharucoDetection,
    cfg: CharucoBoardConfig,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray | None:
    """Estimate ``T_cam_board`` from a ChArUco detection via PnP.

    Parameters
    ----------
    detection:
        Detection of the board described by ``cfg``.
    cfg:
        Board specification.
    K:
        3x3 camera matrix (``fx, fy`` in pixels, ``cx, cy`` in pixels).
    dist:
        Distortion coefficients, as accepted by OpenCV (any length).

    Returns
    -------
    np.ndarray | None
        The 4x4 ``T_cam_board`` transform mapping board points into camera
        coordinates (metres), or ``None`` when PnP failed.
    """
    camera_matrix = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(dist, dtype=np.float64).reshape(-1)
    if detection.n_corners < _MIN_POSE_CORNERS:
        return None
    board = make_board(cfg)
    if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            detection.corners, detection.ids, board, camera_matrix, dist_coeffs,
            rvec, tvec,
        )
        if not ok:
            return None
    else:
        # OpenCV >= 4.13 removed the free pose function.
        object_points, image_points = board.matchImagePoints(
            detection.corners, detection.ids
        )
        if object_points is None or len(object_points) < _MIN_POSE_CORNERS:
            return None
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
    return T_from_rvec_tvec(rvec, tvec)
