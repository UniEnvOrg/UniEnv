"""Tests for the rigid-transform helpers and the 6D pose convention."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from unienv_utils.calibration.transforms import (  # noqa: E402
    average_transforms,
    compose,
    invert_T,
    matrix_to_quaternion,
    pose6d_to_T,
    quaternion_average,
    rotation_angle_deg,
    T_from_rvec_tvec,
    T_to_pose6d,
    T_to_rvec_tvec,
    validate_T,
)

RNG = np.random.default_rng(20260706)


def _rodrigues(rvec) -> np.ndarray:
    """cv2.Rodrigues refuses plain Python lists on newer OpenCV versions."""
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]


def _random_T(rng: np.random.Generator) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rodrigues(rng.uniform(-np.pi, np.pi, 3))
    T[:3, 3] = rng.uniform(-0.5, 0.5, 3)
    return T


def test_invert_and_compose_roundtrip() -> None:
    T = _random_T(RNG)
    assert np.allclose(invert_T(T) @ T, np.eye(4), atol=1e-12)
    assert np.allclose(compose(T, invert_T(T)), np.eye(4), atol=1e-12)
    assert np.allclose(compose(np.eye(4), T, np.eye(4)), T, atol=1e-12)
    assert np.allclose(compose(), np.eye(4))


def test_rvec_tvec_roundtrip() -> None:
    T = _random_T(RNG)
    rvec, tvec = T_to_rvec_tvec(T)
    assert rvec.shape == (3, 1) and tvec.shape == (3, 1)
    assert np.allclose(T_from_rvec_tvec(rvec, tvec), T, atol=1e-12)


def test_pose6d_sxyz_roundtrip() -> None:
    """``T_to_pose6d`` must invert ``pose6d_to_T`` exactly (sxyz convention)."""
    for _ in range(200):
        T = _random_T(RNG)
        assert np.allclose(pose6d_to_T(T_to_pose6d(T)), T, atol=1e-9)


def test_pose6d_uses_extrinsic_xyz_convention() -> None:
    """A pure in-plane rotation must land in ``rz`` (DROID/original sxyz order)."""
    yaw = 0.7
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, yaw])
    assert np.allclose(pose6d_to_T(pose)[:3, :3], _rodrigues([0, 0, yaw]))
    assert T_to_pose6d(pose6d_to_T(pose)) == pytest.approx(pose)

    pitch = np.array([0.0, 0.0, 0.0, 0.0, 0.4, 0.0])
    assert np.allclose(pose6d_to_T(pitch)[:3, :3], _rodrigues([0, 0.4, 0]))
    # R = Rz(rz) @ Ry(ry) @ Rx(rx): the first listed rotation is innermost.
    mixed = pose6d_to_T([0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
    assert np.allclose(
        mixed[:3, :3],
        _rodrigues([0, 0, 0.3]) @ _rodrigues([0, 0.2, 0]) @ _rodrigues([0.1, 0, 0]),
        atol=1e-12,
    )


def test_pose6d_gimbal_lock_roundtrip() -> None:
    for rx, ry, rz in ((0.3, np.pi / 2, 0.7), (0.2, -np.pi / 2, -0.4)):
        T = pose6d_to_T([0.1, -0.2, 0.3, rx, ry, rz])
        assert np.allclose(pose6d_to_T(T_to_pose6d(T)), T, atol=1e-9)


def test_validate_T_accepts_and_rejects() -> None:
    assert validate_T(np.eye(4))
    assert validate_T(_random_T(RNG))
    assert not validate_T(np.eye(3))
    bad_last_row = np.eye(4)
    bad_last_row[3, 3] = 2.0
    assert not validate_T(bad_last_row)
    scaled = np.eye(4)
    scaled[:3, :3] *= 2.0
    assert not validate_T(scaled)
    mirrored = np.eye(4)
    mirrored[0, 0] = -1.0
    assert not validate_T(mirrored)


def test_rotation_angle_deg() -> None:
    assert rotation_angle_deg(np.eye(3)) == pytest.approx(0.0)
    assert rotation_angle_deg(_rodrigues([0, 0, np.pi / 2])) == pytest.approx(90.0)


def test_invert_T_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        invert_T(np.zeros((3, 4)))


def test_average_transforms_and_quaternion_sign_flip() -> None:
    T = _random_T(RNG)
    assert np.allclose(average_transforms([T, T, T]), T, atol=1e-9)

    # Opposite quaternion signs represent the same rotation: the hemisphere fix
    # must keep the average equal to that rotation.
    q = matrix_to_quaternion(T[:3, :3])
    mean = quaternion_average([q, -q, q])
    assert abs(float(np.dot(mean, q))) == pytest.approx(1.0)

    midpoint = np.eye(4)
    midpoint[:3, 3] = np.array([1.0, 1.0, 1.0])
    other = np.eye(4)
    other[:3, 3] = np.array([3.0, 3.0, 3.0])
    assert average_transforms([midpoint, other])[:3, 3] == pytest.approx(
        [2.0, 2.0, 2.0]
    )

    with pytest.raises(ValueError, match="zero transforms"):
        average_transforms([])
