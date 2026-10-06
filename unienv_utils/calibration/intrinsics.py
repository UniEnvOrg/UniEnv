"""ChArUco intrinsic calibration (``K`` and distortion coefficients).

Wraps ``cv2.aruco.calibrateCameraCharuco``.  OpenCV >= 4.13 removed that free
function, so the fallback builds object/image point arrays from the detections
via ``CharucoBoard.matchImagePoints`` and calls ``cv2.calibrateCamera`` --- the
same maths, only the board bookkeeping differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from unienv_utils.calibration.board import CharucoBoardConfig, make_board
from unienv_utils.calibration.detect import CharucoDetection

__all__ = [
    "IntrinsicsResult",
    "calibrate_intrinsics_charuco",
]

# Practical minimum for a well-conditioned ChArUco intrinsic calibration.
_MIN_VIEWS = 8


@dataclass(frozen=True)
class IntrinsicsResult:
    """Result of a ChArUco intrinsic calibration.

    Attributes
    ----------
    K:
        3x3 camera matrix (``fx, fy`` in pixels, ``cx, cy`` in pixels).
    dist:
        Distortion coefficients as returned by OpenCV.
    image_size:
        ``(width, height)`` in pixels of the calibration images.
    mean_reprojection_error_px:
        Mean of ``per_view_errors``, in pixels.  Deliberately the per-view mean
        rather than the solver's pooled RMS, so this field and
        ``per_view_errors`` are always mutually consistent.
    per_view_errors:
        RMS reprojection error of each view, in the order of the detections.
    n_views:
        Number of views used.
    """

    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]
    mean_reprojection_error_px: float
    per_view_errors: tuple[float, ...]
    n_views: int


def _normalise_image_size(image_size: Sequence[int]) -> tuple[int, int]:
    """Return ``(width, height)`` as positive ints."""
    values = tuple(int(value) for value in image_size)
    if len(values) != 2 or values[0] <= 0 or values[1] <= 0:
        raise ValueError(
            f"image_size must be (width, height) in pixels, got {tuple(image_size)!r}."
        )
    return values


def calibrate_intrinsics_charuco(
    detections: Sequence[CharucoDetection],
    cfg: CharucoBoardConfig,
    image_size: Sequence[int],
) -> IntrinsicsResult:
    """Calibrate intrinsics from ChArUco detections of the same board.

    Parameters
    ----------
    detections:
        Detections of the board described by ``cfg`` from different viewpoints.
    cfg:
        Board specification.
    image_size:
        ``(width, height)`` of the calibration images, in pixels.

    Returns
    -------
    IntrinsicsResult
        Camera matrix, distortion coefficients and reprojection errors.

    Raises
    ------
    ValueError
        If fewer than 8 views are supplied.
    """
    views = list(detections)
    if len(views) < _MIN_VIEWS:
        raise ValueError(
            f"Intrinsic calibration needs at least {_MIN_VIEWS} views, "
            f"got {len(views)}."
        )
    size = _normalise_image_size(image_size)
    board = make_board(cfg)
    object_points = []
    image_points = []
    for view in views:
        obj, img = board.matchImagePoints(
            view.corners.astype(np.float32), view.ids.astype(np.int32)
        )
        object_points.append(obj)
        image_points.append(img)

    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        (
            _rms,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
        ) = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=[view.corners for view in views],
            charucoIds=[view.ids for view in views],
            board=board,
            imageSize=size,
            cameraMatrix=None,
            distCoeffs=None,
        )
        rvecs = list(rvecs)
        tvecs = list(tvecs)
    else:
        # OpenCV >= 4.13 removed the free function: same maths via calibrateCamera.
        (
            _rms,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
        ) = cv2.calibrateCamera(
            object_points,
            image_points,
            size,
            None,
            None,
        )

    per_view_errors = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        error = np.linalg.norm(
            projected.reshape(-1, 2) - np.asarray(img).reshape(-1, 2), axis=1
        )
        per_view_errors.append(float(np.sqrt(np.mean(error**2))))

    return IntrinsicsResult(
        K=np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        dist=np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
        image_size=size,
        mean_reprojection_error_px=float(np.mean(per_view_errors)),
        per_view_errors=tuple(per_view_errors),
        n_views=len(views),
    )
