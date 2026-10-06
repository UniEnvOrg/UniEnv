"""Rigid-transform helpers shared by the calibration subpackage.

Conventions
-----------
* ``T_a_b`` is a 4x4 homogeneous matrix mapping points expressed in frame ``b``
  into frame ``a``.  Translations are in metres, rotations are in radians
  unless a name ends in ``_deg``.
* 6D poses are ``(x, y, z, rx, ry, rz)`` with the ``sxyz`` Euler convention:
  ``R = Rz(rz) @ Ry(ry) @ Rx(rx)`` (extrinsic XYZ, intrinsic ZYX), which is the
  same convention as ``scipy.spatial.transform.Rotation.from_euler("xyz")``.
  This matches the DROID-format pose convention.  Gimbal lock (``ry = +-pi/2``)
  is resolved by setting ``rz = 0``.
* Rotations are averaged with a sign-agnostic quaternion mean, so residuals are
  meaningful even when individual poses have nearly opposite quaternion signs.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

__all__ = [
    "average_transforms",
    "compose",
    "invert_T",
    "matrix_to_quaternion",
    "pose6d_to_T",
    "quaternion_average",
    "quaternion_to_matrix",
    "rotation_angle_deg",
    "T_from_rvec_tvec",
    "T_to_pose6d",
    "T_to_rvec_tvec",
    "validate_T",
]

# Below this pitch cosine the Euler extraction is treated as singular.
_EULER_SINGULAR_TOL = 1e-9


def _as_T(T: np.ndarray, name: str = "T") -> np.ndarray:
    """Return ``T`` as a float64 4x4 array (raises ValueError on bad shape)."""
    arr = np.asarray(T, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def validate_T(T: np.ndarray, atol: float = 1e-6) -> bool:
    """Check that ``T`` is a valid rigid transform.

    A valid transform has shape (4, 4), a final row of ``[0, 0, 0, 1]`` and a
    rotation block that is orthonormal with determinant ``+1``.
    """
    arr = np.asarray(T, dtype=np.float64)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        return False
    if not np.allclose(arr[3], (0.0, 0.0, 0.0, 1.0), atol=atol):
        return False
    rotation = arr[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        return False
    return bool(np.isclose(np.linalg.det(rotation), 1.0, atol=atol))


def invert_T(T: np.ndarray) -> np.ndarray:
    """Return the inverse of a rigid transform (cheap transpose form)."""
    arr = _as_T(T)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = arr[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ arr[:3, 3]
    return inverse


def compose(*Ts: np.ndarray) -> np.ndarray:
    """Compose transforms left-to-right: ``compose(T_a_b, T_b_c) == T_a_c``."""
    result = np.eye(4, dtype=np.float64)
    for index, T in enumerate(Ts):
        result = result @ _as_T(T, f"Ts[{index}]")
    return result


def T_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from a Rodrigues rotation vector and translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def T_to_rvec_tvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 transform into a ``(rvec, tvec)`` pair for OpenCV."""
    arr = _as_T(T)
    rvec = cv2.Rodrigues(arr[:3, :3])[0].reshape(3, 1)
    return rvec, arr[:3, 3].reshape(3, 1)


def rotation_angle_deg(R: np.ndarray) -> float:
    """Return the intrinsic rotation angle of a rotation matrix, in degrees."""
    rotation = np.asarray(R, dtype=np.float64)[:3, :3]
    cos_angle = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _euler_sxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """Extract ``(rx, ry, rz)`` such that ``R = Rz(rz) @ Ry(ry) @ Rx(rx)``."""
    sin_pitch = -float(R[2, 0])
    cos_pitch = float(np.hypot(R[0, 0], R[1, 0]))
    ry = float(np.arctan2(sin_pitch, cos_pitch))
    if cos_pitch > _EULER_SINGULAR_TOL:
        rx = float(np.arctan2(R[2, 1], R[2, 2]))
        rz = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        # Gimbal lock: only the sum/difference of rx and rz is observable.
        sign = 1.0 if sin_pitch >= 0.0 else -1.0
        rx = float(np.arctan2(sign * R[0, 1], sign * R[0, 2]))
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def pose6d_to_T(pose6d: np.ndarray) -> np.ndarray:
    """Convert a ``(x, y, z, rx, ry, rz)`` pose (``sxyz`` Euler) to 4x4.

    ``rx, ry, rz`` are rotations about the fixed X, Y, Z axes applied in that
    order (``R = Rz(rz) @ Ry(ry) @ Rx(rx)``), in radians.
    """
    values = np.asarray(pose6d, dtype=np.float64).reshape(-1)
    if values.shape != (6,):
        raise ValueError(f"pose6d must have 6 elements, got {values.shape}.")
    rx, ry, rz = values[3], values[4], values[5]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = values[:3]
    return T


def T_to_pose6d(T: np.ndarray) -> np.ndarray:
    """Convert a 4x4 transform to a ``(x, y, z, rx, ry, rz)`` pose (``sxyz``)."""
    arr = _as_T(T)
    return np.concatenate([arr[:3, 3], _euler_sxyz_from_matrix(arr[:3, :3])])


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to an ``xyzw`` unit quaternion."""
    rotation = np.asarray(R, dtype=np.float64)[:3, :3]
    # Eigenvector form is stable near 180 degrees and has no branch singularity.
    K = np.array([
        [rotation[0, 0] - rotation[1, 1] - rotation[2, 2],
         rotation[1, 0] + rotation[0, 1],
         rotation[2, 0] + rotation[0, 2],
         rotation[1, 2] - rotation[2, 1]],
        [rotation[1, 0] + rotation[0, 1],
         rotation[1, 1] - rotation[0, 0] - rotation[2, 2],
         rotation[2, 1] + rotation[1, 2],
         rotation[2, 0] - rotation[0, 2]],
        [rotation[2, 0] + rotation[0, 2],
         rotation[2, 1] + rotation[1, 2],
         rotation[2, 2] - rotation[0, 0] - rotation[1, 1],
         rotation[0, 1] - rotation[1, 0]],
        [rotation[1, 2] - rotation[2, 1],
         rotation[2, 0] - rotation[0, 2],
         rotation[0, 1] - rotation[1, 0],
         rotation.trace()],
    ]) / 3.0
    values, vectors = np.linalg.eigh(K)
    quaternion = vectors[:, int(np.argmax(values))]
    # The symmetric K convention above yields the quaternion for R.T.
    quaternion[:3] *= -1.0
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert an ``xyzw`` quaternion to a rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.hypot(np.hypot(x, y), np.hypot(z, w)))
    if norm == 0.0:
        raise ValueError("Cannot convert a zero quaternion.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quaternion_average(quaternions: Sequence[np.ndarray]) -> np.ndarray:
    """Return the mean ``xyzw`` quaternion with an explicit hemisphere fix.

    Each quaternion is first aligned to the first one (``q -> -q`` when the dot
    product is negative) so that the sign ambiguity of the representation does
    not cancel the average.
    """
    if len(quaternions) == 0:
        raise ValueError("Cannot average zero quaternions.")
    quats = [np.asarray(q, dtype=np.float64).reshape(4) for q in quaternions]
    anchor = quats[0]
    aligned = [(-q if float(np.dot(q, anchor)) < 0.0 else q) for q in quats]
    scatter = np.zeros((4, 4), dtype=np.float64)
    for q in aligned:
        scatter += np.outer(q, q)
    values, vectors = np.linalg.eigh(scatter)
    mean = vectors[:, int(np.argmax(values))]
    if float(np.dot(mean, anchor)) < 0.0:
        mean = -mean
    return mean / np.linalg.norm(mean)


def average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average rigid transforms (mean translation, quaternion-mean rotation)."""
    if len(transforms) == 0:
        raise ValueError("Cannot average zero transforms.")
    mats = [_as_T(T, f"transforms[{i}]") for i, T in enumerate(transforms)]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_to_matrix(
        quaternion_average([matrix_to_quaternion(T[:3, :3]) for T in mats]))
    result[:3, 3] = np.mean(np.stack([T[:3, 3] for T in mats]), axis=0)
    return result
