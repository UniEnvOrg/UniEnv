"""Synthetic hand-eye calibration tests.

Ground truth rig: a fixed camera (``T_base_cam``) and a board bolted to the
gripper (``T_gripper_board``).  For each synthetic robot pose we compute the
board detection exactly from the loop identity::

    T_cam_board = inverse(T_base_cam) @ T_base_gripper @ T_gripper_board

No cameras, no images and no GPU are involved.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from unienv_utils.calibration.hand_eye import (  # noqa: E402
    HandEyeObservation,
    solve_hand_to_base,
)
from unienv_utils.calibration.transforms import (  # noqa: E402
    invert_T,
    rotation_angle_deg,
    validate_T,
)

RNG = np.random.default_rng(7)


def _homogeneous(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    T[:3, 3] = tvec
    return T


GROUND_TRUTH_T_BASE_CAM = _homogeneous(
    [0.3, -0.4, 0.5], [0.5, 0.1, 0.9]
)
GROUND_TRUTH_T_GRIPPER_BOARD = _homogeneous(
    [1.1, 0.2, -0.7], [0.02, -0.03, 0.15]
)


def _make_observations(
    n: int = 20,
    *,
    rotation_noise_deg: float = 0.0,
    translation_noise_mm: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[HandEyeObservation]:
    """Build ``n`` random, orientation-diverse observations of the rig."""
    generator = RNG if rng is None else rng
    observations = []
    for _ in range(n):
        T_base_gripper = _homogeneous(
            generator.uniform(-np.pi, np.pi, 3), generator.uniform(-0.3, 0.3, 3)
        )
        T_cam_board = (
            invert_T(GROUND_TRUTH_T_BASE_CAM)
            @ T_base_gripper
            @ GROUND_TRUTH_T_GRIPPER_BOARD
        )
        if rotation_noise_deg or translation_noise_mm:
            T_cam_board = T_cam_board @ _homogeneous(
                np.radians(generator.normal(0.0, rotation_noise_deg, 3)),
                generator.normal(0.0, translation_noise_mm / 1000.0, 3),
            )
        observations.append(HandEyeObservation(T_base_gripper, T_cam_board))
    return observations


def test_solve_recovers_ground_truth_without_noise() -> None:
    result = solve_hand_to_base(_make_observations())

    assert result.n_poses == 20
    assert validate_T(result.T_base_cam)
    assert validate_T(result.T_gripper_board)
    assert np.allclose(result.T_base_cam, GROUND_TRUTH_T_BASE_CAM, atol=1e-6)
    assert np.allclose(
        result.T_gripper_board, GROUND_TRUTH_T_GRIPPER_BOARD, atol=1e-6
    )
    # Clean data: the board mounting is perfectly consistent across frames.
    assert result.translation_mm_max < 1e-6
    assert result.rotation_deg_max < 1e-3


def test_solve_with_pose_noise_stays_within_a_few_mm() -> None:
    observations = _make_observations(
        rotation_noise_deg=0.1, translation_noise_mm=0.5
    )
    result = solve_hand_to_base(observations)

    assert np.linalg.norm(
        result.T_base_cam[:3, 3] - GROUND_TRUTH_T_BASE_CAM[:3, 3]
    ) < 0.005
    angle_deg = rotation_angle_deg(
        result.T_base_cam[:3, :3].T @ GROUND_TRUTH_T_BASE_CAM[:3, :3]
    )
    assert angle_deg < 0.5


def test_residual_stats_are_present_and_sane() -> None:
    result = solve_hand_to_base(
        _make_observations(rotation_noise_deg=0.1, translation_noise_mm=0.5)
    )
    assert 0.0 < result.translation_mm_mean <= result.translation_mm_max < 5.0
    assert 0.0 < result.rotation_deg_mean <= result.rotation_deg_max < 1.0


def test_min_poses_is_enforced() -> None:
    with pytest.raises(ValueError, match="at least 4 observations"):
        solve_hand_to_base(_make_observations(3))
    # A lower min_poses makes the same 3-pose set solvable.
    result = solve_hand_to_base(_make_observations(3), min_poses=3)
    assert result.n_poses == 3


def test_invalid_pose_is_rejected() -> None:
    observations = _make_observations(5)
    broken = observations[:]
    broken[2] = HandEyeObservation(
        observations[2].T_base_gripper, np.eye(3)
    )
    with pytest.raises(ValueError, match="T_cam_board is not a valid T"):
        solve_hand_to_base(broken)


def test_low_orientation_diversity_warns_and_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Translations only: warn loudly, then refuse to return a bogus solve."""
    observations = []
    for _ in range(6):
        T_base_gripper = _homogeneous([0.0, 0.0, 0.0], RNG.uniform(-0.3, 0.3, 3))
        observations.append(
            HandEyeObservation(
                T_base_gripper,
                invert_T(GROUND_TRUTH_T_BASE_CAM)
                @ T_base_gripper
                @ GROUND_TRUTH_T_GRIPPER_BOARD,
            )
        )
    with caplog.at_level(logging.WARNING, logger="unienv_utils.calibration.hand_eye"):
        with pytest.raises(ValueError, match="degenerate"):
            solve_hand_to_base(observations)
    assert any("orientation diversity is low" in record.message for record in caplog.records)


def test_parallel_rotation_axes_warn_and_fail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rotations about a single axis are still rank deficient for hand-eye."""
    observations = []
    for angle in np.linspace(0.0, np.radians(60.0), 6):
        T_base_gripper = _homogeneous([0.0, 0.0, angle], [0.0, 0.0, 0.1])
        observations.append(
            HandEyeObservation(
                T_base_gripper,
                invert_T(GROUND_TRUTH_T_BASE_CAM)
                @ T_base_gripper
                @ GROUND_TRUTH_T_GRIPPER_BOARD,
            )
        )
    with caplog.at_level(logging.WARNING, logger="unienv_utils.calibration.hand_eye"):
        with pytest.raises(ValueError, match="degenerate"):
            solve_hand_to_base(observations)
    assert any(
        "orientation diversity is low" in record.message for record in caplog.records
    )


def test_opposite_single_axis_rotations_warn_and_fail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """+/- rotations about one axis cancel only in a signed mean, not in reality."""
    observations = []
    for angle_deg in (30.0, -30.0, 20.0, -20.0, 10.0, 0.0):
        T_base_gripper = _homogeneous(
            [0.0, 0.0, np.radians(angle_deg)], [0.0, 0.0, 0.1]
        )
        observations.append(
            HandEyeObservation(
                T_base_gripper,
                invert_T(GROUND_TRUTH_T_BASE_CAM)
                @ T_base_gripper
                @ GROUND_TRUTH_T_GRIPPER_BOARD,
            )
        )
    with caplog.at_level(logging.WARNING, logger="unienv_utils.calibration.hand_eye"):
        with pytest.raises(ValueError, match="degenerate"):
            solve_hand_to_base(observations)
    assert any(
        "orientation diversity is low" in record.message for record in caplog.records
    )


@pytest.mark.parametrize(
    "method",
    [cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_DANIILIDIS],
    ids=("TSAI", "DANIILIDIS"),
)
def test_alternative_solvers_recover_clean_rig(method: int) -> None:
    """Argument-marshalling smoke test for the other cv2 hand-eye solvers."""
    result = solve_hand_to_base(
        _make_observations(rng=np.random.default_rng(123)), method=method
    )
    assert validate_T(result.T_base_cam)
    assert np.linalg.norm(
        result.T_base_cam[:3, 3] - GROUND_TRUTH_T_BASE_CAM[:3, 3]
    ) < 1e-3
    assert rotation_angle_deg(
        result.T_base_cam[:3, :3].T @ GROUND_TRUTH_T_BASE_CAM[:3, :3]
    ) < 0.1


def test_diverse_poses_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="unienv_utils.calibration.hand_eye"):
        solve_hand_to_base(_make_observations())
    assert not caplog.records
