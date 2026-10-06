"""Relative pose between two rigidly mounted cameras.

Both cameras observe the same board in an otherwise arbitrary pose.  For each
paired observation the relative transform is::

    T_camA_camB = T_camA_board @ inverse(T_camB_board)

By default the repeated estimates are gated with tianji's MAD-based outlier
rejection (translation and rotation errors from a component-wise median pose),
then the inliers are averaged (mean translation, quaternion-mean rotation).  The
result can be transferred into the robot base frame with a single composition:
``T_base_camB = T_base_camA @ T_camA_camB``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from unienv_utils.calibration.transforms import (
    average_transforms,
    invert_T,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_angle_deg,
    validate_T,
)

__all__ = [
    "PairObservation",
    "PairResult",
    "solve_camera_relative",
    "transfer_to_base",
]

# Outlier-gating constants ported from tianji's camera-pair solve.
_OUTLIER_TRANSLATION_FLOOR_M = 0.010
_OUTLIER_ROTATION_FLOOR_DEG = 1.0
_OUTLIER_MAD_SCALE = 3.5


@dataclass(frozen=True)
class PairObservation:
    """One synchronised observation of the same board by two cameras.

    Attributes
    ----------
    T_camA_board:
        4x4 board pose in camera A's frame (metres).
    T_camB_board:
        4x4 board pose in camera B's frame (metres).
    """

    T_camA_board: np.ndarray
    T_camB_board: np.ndarray


@dataclass(frozen=True)
class PairResult:
    """Result of a camera-pair relative-pose solve.

    Attributes
    ----------
    T_camA_camB:
        4x4 transform mapping camera B coordinates into camera A coordinates,
        averaged over the inlier observations.
    translation_mm_mean, translation_mm_max:
        Per-inlier deviation of the candidates from the average, in millimetres.
    rotation_deg_mean, rotation_deg_max:
        Per-inlier deviation of the candidates from the average, in degrees.
    n_pairs:
        Number of paired observations supplied to the solve.
    n_inliers:
        Number of observations that survived outlier rejection (all of them
        when ``reject_outliers=False``).
    inlier_mask:
        Per-observation booleans in input order; ``True`` marks the
        observations used for the average and all residual statistics above.
    """

    T_camA_camB: np.ndarray
    translation_mm_mean: float
    translation_mm_max: float
    rotation_deg_mean: float
    rotation_deg_max: float
    n_pairs: int
    n_inliers: int
    inlier_mask: tuple[bool, ...]


def _robust_center_transform(candidates: Sequence[np.ndarray]) -> np.ndarray:
    """Component-wise median pose used only for outlier classification."""
    quats = np.stack([matrix_to_quaternion(T[:3, :3]) for T in candidates])
    anchor = quats[0]
    quats = np.stack([(-q if float(np.dot(q, anchor)) < 0.0 else q) for q in quats])
    q_median = np.median(quats, axis=0)
    q_median /= np.linalg.norm(q_median)
    center = np.eye(4, dtype=np.float64)
    center[:3, :3] = quaternion_to_matrix(q_median)
    center[:3, 3] = np.median(np.stack([T[:3, 3] for T in candidates]), axis=0)
    return center


def _candidate_errors(
    candidates: Sequence[np.ndarray], center: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate translation (m) and rotation (deg) errors from ``center``."""
    translation_m = np.array([
        float(np.linalg.norm(T[:3, 3] - center[:3, 3])) for T in candidates
    ])
    rotation_deg = np.array([
        rotation_angle_deg(center[:3, :3].T @ T[:3, :3]) for T in candidates
    ])
    return translation_m, rotation_deg


def _robust_limit(values: np.ndarray, floor: float) -> float:
    """MAD-based rejection threshold, never tighter than ``floor``."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(floor, median + _OUTLIER_MAD_SCALE * 1.4826 * mad)


def solve_camera_relative(
    observations: Sequence[PairObservation],
    *,
    reject_outliers: bool = True,
    min_inliers: int = 8,
) -> PairResult:
    """Solve ``T_camA_camB`` from paired board observations.

    Parameters
    ----------
    observations:
        Paired board observations.
    reject_outliers:
        When ``True`` (default), candidates whose translation or rotation error
        from a robust median centre exceeds a MAD-based threshold (floors:
        10 mm / 1 deg) are dropped before averaging.
    min_inliers:
        Minimum number of candidates that must survive outlier rejection;
        defaults to tianji's floor of 8 paired views.

    Returns
    -------
    PairResult
        Averaged transform, inlier bookkeeping and residuals over the inliers.

    Raises
    ------
    ValueError
        If no observations are given, ``min_inliers`` is not positive, a pose
        is not a valid rigid transform, or fewer than ``min_inliers``
        candidates survive outlier rejection.
    """
    pairs = list(observations)
    if not pairs:
        raise ValueError("Camera-pair calibration needs at least one observation.")
    if min_inliers < 1:
        raise ValueError(f"min_inliers must be at least 1, got {min_inliers}.")
    candidates = []
    for index, pair in enumerate(pairs):
        if not validate_T(pair.T_camA_board):
            raise ValueError(f"observations[{index}].T_camA_board is not a valid T.")
        if not validate_T(pair.T_camB_board):
            raise ValueError(f"observations[{index}].T_camB_board is not a valid T.")
        candidates.append(
            np.asarray(pair.T_camA_board, dtype=np.float64)
            @ invert_T(np.asarray(pair.T_camB_board, dtype=np.float64))
        )

    if reject_outliers:
        center = _robust_center_transform(candidates)
        translation_m, rotation_deg_all = _candidate_errors(candidates, center)
        translation_limit = _robust_limit(
            translation_m, _OUTLIER_TRANSLATION_FLOOR_M
        )
        rotation_limit = _robust_limit(rotation_deg_all, _OUTLIER_ROTATION_FLOOR_DEG)
        inliers = (translation_m <= translation_limit) & (
            rotation_deg_all <= rotation_limit
        )
        n_inliers = int(inliers.sum())
        if n_inliers < min_inliers:
            raise ValueError(
                f"Only {n_inliers}/{len(candidates)} paired observations survived "
                f"outlier rejection; need at least {min_inliers}. Check the "
                "detections or collect more views."
            )
        selected = [T for T, keep in zip(candidates, inliers) if keep]
        inlier_mask = tuple(bool(keep) for keep in inliers)
    else:
        selected = candidates
        inlier_mask = (True,) * len(candidates)

    T_camA_camB = average_transforms(selected)
    translation_mm = np.array([
        float(np.linalg.norm(T[:3, 3] - T_camA_camB[:3, 3])) for T in selected
    ]) * 1000.0
    rotation_deg = np.array([
        rotation_angle_deg(T_camA_camB[:3, :3].T @ T[:3, :3]) for T in selected
    ])
    return PairResult(
        T_camA_camB=T_camA_camB,
        translation_mm_mean=float(translation_mm.mean()),
        translation_mm_max=float(translation_mm.max()),
        rotation_deg_mean=float(rotation_deg.mean()),
        rotation_deg_max=float(rotation_deg.max()),
        n_pairs=len(candidates),
        n_inliers=len(selected),
        inlier_mask=inlier_mask,
    )


def transfer_to_base(T_base_camA: np.ndarray, T_camA_camB: np.ndarray) -> np.ndarray:
    """Return ``T_base_camB = T_base_camA @ T_camA_camB``.

    Raises
    ------
    ValueError
        If either input is not a valid rigid transform.
    """
    if not validate_T(T_base_camA):
        raise ValueError("T_base_camA is not a valid T.")
    if not validate_T(T_camA_camB):
        raise ValueError("T_camA_camB is not a valid T.")
    return np.asarray(T_base_camA, dtype=np.float64) @ np.asarray(
        T_camA_camB, dtype=np.float64
    )
