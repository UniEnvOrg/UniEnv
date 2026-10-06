"""Relative pose between two rigidly mounted cameras.

Both cameras observe the same board in an otherwise arbitrary pose.  For each
paired observation the relative transform is::

    T_camA_camB = T_camA_board @ inverse(T_camB_board)

The repeated estimates are averaged (mean translation, quaternion-mean
rotation) and can then be transferred into the robot base frame with a single
composition: ``T_base_camB = T_base_camA @ T_camA_camB``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from unienv_utils.calibration.transforms import (
    average_transforms,
    invert_T,
    rotation_angle_deg,
    validate_T,
)

__all__ = [
    "PairObservation",
    "PairResult",
    "solve_camera_relative",
    "transfer_to_base",
]


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
        4x4 transform mapping camera B coordinates into camera A coordinates.
    translation_mm_mean, translation_mm_max:
        Per-pair deviation of the candidates from the average, in millimetres.
    rotation_deg_mean, rotation_deg_max:
        Per-pair deviation of the candidates from the average, in degrees.
    n_pairs:
        Number of paired observations used.
    """

    T_camA_camB: np.ndarray
    translation_mm_mean: float
    translation_mm_max: float
    rotation_deg_mean: float
    rotation_deg_max: float
    n_pairs: int


def solve_camera_relative(observations: Sequence[PairObservation]) -> PairResult:
    """Solve ``T_camA_camB`` from paired board observations.

    Raises
    ------
    ValueError
        If no observations are given or a pose is not a valid rigid transform.
    """
    pairs = list(observations)
    if not pairs:
        raise ValueError("Camera-pair calibration needs at least one observation.")
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

    T_camA_camB = average_transforms(candidates)
    translation_mm = np.array([
        float(np.linalg.norm(T[:3, 3] - T_camA_camB[:3, 3])) for T in candidates
    ]) * 1000.0
    rotation_deg = np.array([
        rotation_angle_deg(T_camA_camB[:3, :3].T @ T[:3, :3]) for T in candidates
    ])
    return PairResult(
        T_camA_camB=T_camA_camB,
        translation_mm_mean=float(translation_mm.mean()),
        translation_mm_max=float(translation_mm.max()),
        rotation_deg_mean=float(rotation_deg.mean()),
        rotation_deg_max=float(rotation_deg.max()),
        n_pairs=len(candidates),
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
