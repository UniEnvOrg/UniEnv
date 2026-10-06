"""Eye-to-hand calibration: solve ``T_base_cam`` for a fixed camera.

The board is rigidly attached to the end-effector; the camera is bolted in
place.  For each captured pose we have ``T_base_gripper`` (forward kinematics)
and ``T_cam_board`` (PnP on the detected board).  The unknown ``T_base_cam``
satisfies the loop identity::

    T_base_gripper @ T_gripper_board == T_base_cam @ T_cam_board

``cv2.calibrateHandEye`` solves the eye-in-hand problem (it returns
``T_gripper_cam`` from ``gripper2base`` / ``target2cam`` inputs).  Feeding it
the *inverted* gripper poses as ``gripper2base`` swaps the roles of the frames,
so the returned transform is ``T_base_cam`` directly --- exactly the input swap
used by tianji's ``calibrate_extrinsics.solve_session``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from unienv_utils.calibration.transforms import (
    average_transforms,
    invert_T,
    rotation_angle_deg,
    validate_T,
)

__all__ = [
    "HandEyeObservation",
    "HandEyeResult",
    "solve_hand_to_base",
]

logger = logging.getLogger(__name__)

# Below these thresholds the pose set is too degenerate for a stable solve.
_MIN_RELATIVE_ROTATION_DEG = 5.0
_AXIS_DEGENERATE_TOL_DEG = 1.0
_MAX_AXIS_SPREAD_DEG = 15.0


@dataclass(frozen=True)
class HandEyeObservation:
    """One capture of a hand-eye session.

    Attributes
    ----------
    T_base_gripper:
        4x4 robot pose from forward kinematics (metres).
    T_cam_board:
        4x4 board pose from PnP on that capture's image (metres).
    """

    T_base_gripper: np.ndarray
    T_cam_board: np.ndarray


@dataclass(frozen=True)
class HandEyeResult:
    """Result of an eye-to-hand solve.

    Attributes
    ----------
    T_base_cam:
        4x4 transform mapping camera coordinates into robot-base coordinates.
    T_gripper_board:
        Averaged 4x4 transform mapping board coordinates into gripper
        coordinates, i.e. the physical board mounting.
    translation_mm_mean, translation_mm_max:
        Per-frame spread of ``T_gripper_board`` translations, in millimetres.
        This is the standard hand-eye consistency metric.
    rotation_deg_mean, rotation_deg_max:
        Per-frame spread of ``T_gripper_board`` rotations, in degrees.
    n_poses:
        Number of observations used.
    """

    T_base_cam: np.ndarray
    T_gripper_board: np.ndarray
    translation_mm_mean: float
    translation_mm_max: float
    rotation_deg_mean: float
    rotation_deg_max: float
    n_poses: int


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    """Flip ``axis`` into a canonical hemisphere (first non-zero component > 0).

    Rodrigues vectors for rotations about ``a`` and ``-a`` differ only by their
    sign, so without this normalisation opposite-direction rotations about the
    same axis cancel in the mean and hide the low-diversity warning.
    """
    for component in axis:
        if component > 0.0:
            return axis
        if component < 0.0:
            return -axis
    return axis


def _diversity_warning(observations: Sequence[HandEyeObservation]) -> str | None:
    """Return a warning message when the gripper poses are degenerate."""
    reference = np.asarray(observations[0].T_base_gripper, dtype=np.float64)[:3, :3]
    axes: list[np.ndarray] = []
    max_angle_deg = 0.0
    for observation in observations[1:]:
        rotation = np.asarray(observation.T_base_gripper, dtype=np.float64)[:3, :3]
        relative = reference.T @ rotation
        angle_deg = rotation_angle_deg(relative)
        max_angle_deg = max(max_angle_deg, angle_deg)
        if angle_deg > _AXIS_DEGENERATE_TOL_DEG:
            axis = cv2.Rodrigues(relative)[0].reshape(3)
            axes.append(_canonical_axis(axis / np.linalg.norm(axis)))
    if max_angle_deg < _MIN_RELATIVE_ROTATION_DEG:
        return (
            "orientation diversity is low: the gripper rotates by at most "
            f"{max_angle_deg:.2f} deg across all {len(observations)} poses; "
            "hand-eye calibration needs several distinct wrist orientations."
        )
    if len(axes) >= 2:
        mean_axis = np.mean(np.stack(axes), axis=0)
        norm = float(np.linalg.norm(mean_axis))
        if norm > 0.0:
            mean_axis = mean_axis / norm
            worst_dot = min(
                abs(float(np.dot(axis, mean_axis))) for axis in axes
            )
            spread_deg = float(
                np.degrees(np.arccos(np.clip(worst_dot, -1.0, 1.0)))
            )
            if spread_deg < _MAX_AXIS_SPREAD_DEG:
                return (
                    "orientation diversity is low: all relative rotation axes are "
                    f"within {spread_deg:.2f} deg of each other, which risks a "
                    "rank-deficient hand-eye solve."
                )
    return None


def solve_hand_to_base(
    observations: Sequence[HandEyeObservation],
    *,
    method: int = cv2.CALIB_HAND_EYE_PARK,
    min_poses: int = 4,
) -> HandEyeResult:
    """Solve ``T_base_cam`` (eye-to-hand) from board observations.

    Parameters
    ----------
    observations:
        Captures pairing a robot pose with a board detection.
    method:
        A ``cv2.CALIB_HAND_EYE_*`` method.  ``CALIB_HAND_EYE_PARK`` is the
        default used by tianji and is numerically exact on clean data.
    min_poses:
        Minimum number of observations required.

    Returns
    -------
    HandEyeResult
        Solution plus per-frame consistency residuals, by default in the
        ``T_gripper_board`` frame: mean < 3 mm / max < 8 mm indicates a good
        capture set.

    Raises
    ------
    ValueError
        If fewer than ``min_poses`` observations were given or if any pose is
        not a valid rigid transform.
    """
    poses = list(observations)
    if len(poses) < min_poses:
        raise ValueError(
            f"Hand-eye calibration needs at least {min_poses} observations, "
            f"got {len(poses)}."
        )
    for index, observation in enumerate(poses):
        if not validate_T(observation.T_base_gripper):
            raise ValueError(f"observations[{index}].T_base_gripper is not a valid T.")
        if not validate_T(observation.T_cam_board):
            raise ValueError(f"observations[{index}].T_cam_board is not a valid T.")

    warning = _diversity_warning(poses)
    if warning is not None:
        logger.warning("%s", warning)

    # Eye-to-hand input swap: the inverted gripper pose plays the role of
    # "gripper2base", so calibrateHandEye returns T_base_cam.
    R_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    R_target2cam: list[np.ndarray] = []
    t_target2cam: list[np.ndarray] = []
    for observation in poses:
        T_ee_base = invert_T(observation.T_base_gripper)
        T_cam_board = np.asarray(observation.T_cam_board, dtype=np.float64)
        R_gripper2base.append(T_ee_base[:3, :3])
        t_gripper2base.append(T_ee_base[:3, 3])
        R_target2cam.append(T_cam_board[:3, :3])
        t_target2cam.append(T_cam_board[:3, 3])

    R_base_cam, t_base_cam = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=method,
    )
    T_base_cam = np.eye(4, dtype=np.float64)
    T_base_cam[:3, :3] = R_base_cam
    T_base_cam[:3, 3] = np.asarray(t_base_cam, dtype=np.float64).reshape(3)
    if not np.isfinite(T_base_cam).all():
        raise ValueError(
            "Hand-eye solve failed: the observation set is degenerate (e.g. no "
            "rotation about at least two distinct axes); add captures with more "
            "varied wrist orientations."
        )

    # Per-frame board mounting from the loop identity, then its spread.
    per_frame = [
        invert_T(observation.T_base_gripper)
        @ T_base_cam
        @ np.asarray(observation.T_cam_board, dtype=np.float64)
        for observation in poses
    ]
    T_gripper_board = average_transforms(per_frame)
    translation_mm = np.array([
        float(np.linalg.norm(T[:3, 3] - T_gripper_board[:3, 3]))
        for T in per_frame
    ]) * 1000.0
    rotation_deg = np.array([
        rotation_angle_deg(T_gripper_board[:3, :3].T @ T[:3, :3])
        for T in per_frame
    ])
    return HandEyeResult(
        T_base_cam=T_base_cam,
        T_gripper_board=T_gripper_board,
        translation_mm_mean=float(translation_mm.mean()),
        translation_mm_max=float(translation_mm.max()),
        rotation_deg_mean=float(rotation_deg.mean()),
        rotation_deg_max=float(rotation_deg.max()),
        n_poses=len(poses),
    )
