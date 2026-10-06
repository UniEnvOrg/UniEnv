"""Synthetic camera-pair (two rigidly mounted cameras) tests."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from unienv_utils.calibration.camera_pair import (  # noqa: E402
    PairObservation,
    solve_camera_relative,
    transfer_to_base,
)
from unienv_utils.calibration.transforms import (  # noqa: E402
    invert_T,
    rotation_angle_deg,
    validate_T,
)

RNG = np.random.default_rng(11)


def _homogeneous(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    T[:3, 3] = tvec
    return T


GROUND_TRUTH_T_CAMA_CAMB = _homogeneous([0.2, 0.3, -0.1], [0.2, -0.05, 0.02])


def _make_pairs(
    n: int = 15,
    *,
    rotation_noise_deg: float = 0.0,
    translation_noise_mm: float = 0.0,
) -> list[PairObservation]:
    pairs = []
    for _ in range(n):
        T_camB_board = _homogeneous(
            RNG.uniform(-np.pi, np.pi, 3),
            RNG.uniform(-0.2, 0.2, 3) + np.array([0.0, 0.0, 0.6]),
        )
        T_camA_board = GROUND_TRUTH_T_CAMA_CAMB @ T_camB_board
        if rotation_noise_deg or translation_noise_mm:
            T_camA_board = T_camA_board @ _homogeneous(
                np.radians(RNG.normal(0.0, rotation_noise_deg, 3)),
                RNG.normal(0.0, translation_noise_mm / 1000.0, 3),
            )
        pairs.append(PairObservation(T_camA_board, T_camB_board))
    return pairs


def _corrupt_translation(
    pairs: list[PairObservation], indices: Iterable[int], delta_m: float = 0.05
) -> list[PairObservation]:
    """Return a copy of ``pairs`` with ``indices`` shifted ~5 cm along camera A x."""
    corrupted = list(pairs)
    for index in indices:
        T_camA_board = corrupted[index].T_camA_board.copy()
        T_camA_board[:3, 3] += np.array([delta_m, 0.0, 0.0])
        corrupted[index] = PairObservation(
            T_camA_board, corrupted[index].T_camB_board
        )
    return corrupted


def test_solve_recovers_relative_pose_without_noise() -> None:
    result = solve_camera_relative(_make_pairs())
    assert result.n_pairs == 15
    assert validate_T(result.T_camA_camB)
    assert np.allclose(result.T_camA_camB, GROUND_TRUTH_T_CAMA_CAMB, atol=1e-6)
    assert result.translation_mm_max < 1e-6
    assert result.rotation_deg_max < 1e-3


def test_solve_with_noise_stays_within_a_few_mm() -> None:
    result = solve_camera_relative(
        _make_pairs(rotation_noise_deg=0.1, translation_noise_mm=0.5)
    )
    assert np.linalg.norm(
        result.T_camA_camB[:3, 3] - GROUND_TRUTH_T_CAMA_CAMB[:3, 3]
    ) < 0.005
    angle_deg = rotation_angle_deg(
        result.T_camA_camB[:3, :3].T @ GROUND_TRUTH_T_CAMA_CAMB[:3, :3]
    )
    assert angle_deg < 0.5
    assert 0.0 < result.translation_mm_mean <= result.translation_mm_max < 5.0
    assert 0.0 < result.rotation_deg_mean <= result.rotation_deg_max < 1.0


def test_per_pair_relative_pose_is_exact_by_construction() -> None:
    """Every synthetic pair must reproduce T_camA_camB exactly (no averaging slack)."""
    for pair in _make_pairs(5):
        candidate = pair.T_camA_board @ invert_T(pair.T_camB_board)
        assert np.allclose(candidate, GROUND_TRUTH_T_CAMA_CAMB, atol=1e-12)


def test_transfer_to_base_composes_correctly() -> None:
    T_base_camA = _homogeneous([0.1, -0.2, 0.3], [1.0, 0.5, -0.2])
    expected = T_base_camA @ GROUND_TRUTH_T_CAMA_CAMB
    transferred = transfer_to_base(T_base_camA, GROUND_TRUTH_T_CAMA_CAMB)
    assert np.allclose(transferred, expected, atol=1e-12)
    assert validate_T(transferred)
    assert np.allclose(
        transfer_to_base(T_base_camA, GROUND_TRUTH_T_CAMA_CAMB)[:3, 3],
        (T_base_camA @ GROUND_TRUTH_T_CAMA_CAMB)[:3, 3],
    )


def test_single_corrupted_pair_is_rejected() -> None:
    """One pair off by ~5 cm must not drag the solve away from ground truth."""
    pairs = _corrupt_translation(_make_pairs(16), [7])
    result = solve_camera_relative(pairs)

    assert result.n_pairs == 16
    assert result.n_inliers == 15
    assert result.inlier_mask[7] is False
    assert all(keep for index, keep in enumerate(result.inlier_mask) if index != 7)
    assert np.linalg.norm(
        result.T_camA_camB[:3, 3] - GROUND_TRUTH_T_CAMA_CAMB[:3, 3]
    ) < 0.002
    # Residuals are computed over the inliers: clean candidates are exact.
    assert result.translation_mm_max < 1e-6


def test_outlier_rejection_can_be_disabled() -> None:
    pairs = _corrupt_translation(_make_pairs(16), [7])
    robust = solve_camera_relative(pairs)
    naive = solve_camera_relative(pairs, reject_outliers=False)

    assert naive.n_pairs == naive.n_inliers == 16
    assert naive.inlier_mask == (True,) * 16
    naive_error = float(np.linalg.norm(
        naive.T_camA_camB[:3, 3] - GROUND_TRUTH_T_CAMA_CAMB[:3, 3]
    ))
    robust_error = float(np.linalg.norm(
        robust.T_camA_camB[:3, 3] - GROUND_TRUTH_T_CAMA_CAMB[:3, 3]
    ))
    # The plain average is dragged ~3 mm by the single 5 cm corruption.
    assert 0.001 < naive_error < 0.05
    assert robust_error < 0.1 * naive_error


def test_min_inliers_violation_raises() -> None:
    pairs = _corrupt_translation(_make_pairs(10), [1, 4, 7])
    with pytest.raises(ValueError, match="survived outlier rejection"):
        solve_camera_relative(pairs)
    # The same set solves when the caller accepts the 7 surviving observations.
    result = solve_camera_relative(pairs, min_inliers=7)
    assert result.n_inliers == 7
    assert result.n_pairs == 10
    with pytest.raises(ValueError, match="min_inliers"):
        solve_camera_relative(pairs, min_inliers=0)


def test_solve_camera_relative_validates_inputs() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        solve_camera_relative([])
    with pytest.raises(ValueError, match="T_camB_board is not a valid T"):
        solve_camera_relative([PairObservation(np.eye(4), np.eye(3))])
    with pytest.raises(ValueError, match="T_camA_camB is not a valid T"):
        transfer_to_base(np.eye(4), np.eye(3))
