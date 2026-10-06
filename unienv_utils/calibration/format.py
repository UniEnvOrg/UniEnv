"""Canonical UniEnv calibration JSON format (``unienv_calibration/v1``).

Schema::

    {
      "format": "unienv_calibration/v1",
      "camera_serial": "31726771",
      "frame": "robot_base",
      "T_base_cam": [[...], [...], [...], [...]],
      "intrinsics": {
        "K": [[...], [...], [...]],
        "distortion": [...],
        "resolution": [width, height],
        "source": "zed_sdk_factory" | "charuco_calibrated" | "unknown"
      },
      "solver": "cv2.calibrateHandEye/PARK + eye-to-hand input swap",
      "residuals": {
        "translation_mm_mean": 0.0,
        "translation_mm_max": 0.0,
        "rotation_deg_max": 0.0
      },
      "board": {
        "row_count": 9, "col_count": 14, "checker_size_m": 0.02,
        "marker_size_m": 0.016, "aruco_dict_name": "DICT_5X5_100"
      },
      "created": "2026-05-17T12:34:56+02:00",
      "n_observations": 27
    }

All transforms map into the frame named by ``"frame"`` (translation in metres).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from unienv_utils.calibration.board import CharucoBoardConfig
from unienv_utils.calibration.intrinsics import IntrinsicsResult
from unienv_utils.calibration.transforms import validate_T

__all__ = [
    "CALIBRATION_FORMAT",
    "INTRINSICS_SOURCES",
    "load_calibration",
    "REQUIRED_RESIDUAL_KEYS",
    "save_calibration",
    "validate_calibration_dict",
]

CALIBRATION_FORMAT = "unienv_calibration/v1"
INTRINSICS_SOURCES = ("zed_sdk_factory", "charuco_calibrated", "unknown")
REQUIRED_RESIDUAL_KEYS = ("translation_mm_mean", "translation_mm_max", "rotation_deg_max")

_DEFAULT_FRAME = "robot_base"


def _normalise_intrinsics(intrinsics: Any) -> dict[str, Any] | None:
    """Convert an ``IntrinsicsResult`` or mapping to the JSON intrinsics block."""
    if intrinsics is None:
        return None
    if isinstance(intrinsics, IntrinsicsResult):
        intrinsics = {
            "K": intrinsics.K,
            "distortion": intrinsics.dist,
            "resolution": list(intrinsics.image_size),
            "source": "charuco_calibrated",
        }
    if not isinstance(intrinsics, Mapping):
        raise ValueError(
            "intrinsics must be an IntrinsicsResult, a mapping or None, got "
            f"{type(intrinsics)!r}."
        )
    block = dict(intrinsics)
    block.setdefault("source", "unknown")
    K = np.asarray(block.get("K"), dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics['K'] must be 3x3, got {K.shape}.")
    block["K"] = K.tolist()
    if "distortion" not in block:
        raise ValueError("intrinsics must contain a 'distortion' list.")
    block["distortion"] = np.asarray(
        block["distortion"], dtype=np.float64
    ).reshape(-1).tolist()
    resolution = block.get("resolution")
    if resolution is None or len(tuple(resolution)) != 2:
        raise ValueError(
            "intrinsics['resolution'] must be [width, height], got "
            f"{resolution!r}."
        )
    block["resolution"] = [int(value) for value in resolution]
    return block


def _normalise_residuals(residuals: Any) -> dict[str, float] | None:
    """Validate and normalise a residuals mapping."""
    if residuals is None:
        return None
    if not isinstance(residuals, Mapping):
        raise ValueError(f"residuals must be a mapping or None, got {type(residuals)!r}.")
    block: dict[str, float] = {}
    for key, value in residuals.items():
        if value is None:
            block[str(key)] = float("nan")
        else:
            block[str(key)] = float(value)
    missing = [key for key in REQUIRED_RESIDUAL_KEYS if key not in block]
    if missing:
        raise ValueError(f"residuals is missing required keys: {missing}.")
    return block


def save_calibration(
    path: str | Path,
    *,
    T_base_cam: np.ndarray,
    camera_serial: str,
    solver: str,
    intrinsics: Any = None,
    residuals: Mapping[str, float] | None = None,
    board: CharucoBoardConfig | None = None,
    frame: str = _DEFAULT_FRAME,
    n_observations: int | None = None,
) -> Path:
    """Write a calibration file in the canonical ``unienv_calibration/v1`` format.

    Parameters
    ----------
    path:
        Output ``.json`` path; parent directories are created.
    T_base_cam:
        4x4 rigid transform mapping camera coordinates into ``frame``.
    camera_serial:
        Camera identifier (ZED serial or similar).
    solver:
        Human-readable description of the solver that produced the transform.
    intrinsics:
        ``IntrinsicsResult`` or mapping with ``K``, ``distortion``,
        ``resolution`` and optionally ``source``.
    residuals:
        Mapping with at least ``translation_mm_mean``, ``translation_mm_max``
        and ``rotation_deg_max``.
    board:
        Board configuration used for the calibration.
    frame:
        Name of the target frame of ``T_base_cam``.
    n_observations:
        Number of captures that fed the solve.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    transform = np.asarray(T_base_cam, dtype=np.float64)
    if not validate_T(transform):
        raise ValueError("T_base_cam is not a valid rigid transform.")
    if not str(camera_serial):
        raise ValueError("camera_serial must be a non-empty identifier.")
    document: dict[str, Any] = {
        "format": CALIBRATION_FORMAT,
        "camera_serial": str(camera_serial),
        "frame": str(frame),
        "T_base_cam": transform.tolist(),
        "intrinsics": _normalise_intrinsics(intrinsics),
        "solver": str(solver),
        "residuals": _normalise_residuals(residuals),
        "board": board.to_dict() if board is not None else None,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_observations": None if n_observations is None else int(n_observations),
    }
    validate_calibration_dict(document)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    return out_path


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Load and validate a calibration file.

    Raises
    ------
    ValueError
        If the file is malformed or has an unknown ``format`` version.
    """
    with open(Path(path)) as handle:
        document = json.load(handle)
    validate_calibration_dict(document)
    return document


def _parse_created(value: Any) -> None:
    """Validate an ISO-8601 creation timestamp."""
    if not isinstance(value, str):
        raise ValueError(f"created must be an ISO-8601 string, got {value!r}.")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"created is not a valid ISO-8601 timestamp: {value!r}.") from error


def validate_calibration_dict(d: Mapping[str, Any]) -> None:
    """Validate a calibration document in place; raise ``ValueError`` if bad.

    Only the structural contract is checked: format version, transform shape
    and orthonormality, intrinsics shapes, residuals keys, board validity and
    timestamp parseability.
    """
    if not isinstance(d, Mapping):
        raise ValueError(f"calibration document must be a mapping, got {type(d)!r}.")
    version = d.get("format")
    if version != CALIBRATION_FORMAT:
        raise ValueError(
            f"Unsupported calibration format {version!r}; this build reads "
            f"{CALIBRATION_FORMAT!r} (or migrate the file)."
        )
    for key in ("camera_serial", "frame", "solver"):
        if not isinstance(d.get(key), str) or not d[key]:
            raise ValueError(f"{key} must be a non-empty string, got {d.get(key)!r}.")
    transform = np.asarray(d.get("T_base_cam"), dtype=np.float64)
    if not validate_T(transform):
        raise ValueError("T_base_cam must be a valid 4x4 rigid transform.")

    intrinsics = d.get("intrinsics")
    if intrinsics is not None:
        if not isinstance(intrinsics, Mapping):
            raise ValueError("intrinsics must be a mapping or null.")
        K = np.asarray(intrinsics.get("K"), dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"intrinsics['K'] must be 3x3, got {K.shape}.")
        if not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            raise ValueError("intrinsics['K'] must be finite with positive focals.")
        distortion = np.asarray(intrinsics.get("distortion"), dtype=np.float64)
        if distortion.ndim != 1 or not np.isfinite(distortion).all():
            raise ValueError("intrinsics['distortion'] must be a finite 1-D list.")
        resolution = tuple(intrinsics.get("resolution") or ())
        if len(resolution) != 2 or min(resolution) <= 0:
            raise ValueError(
                f"intrinsics['resolution'] must be [width, height], got {resolution!r}."
            )
        source = intrinsics.get("source")
        if source not in INTRINSICS_SOURCES:
            raise ValueError(
                f"intrinsics['source'] must be one of {list(INTRINSICS_SOURCES)}, "
                f"got {source!r}."
            )

    residuals = d.get("residuals")
    if residuals is not None:
        if not isinstance(residuals, Mapping):
            raise ValueError("residuals must be a mapping or null.")
        missing = [key for key in REQUIRED_RESIDUAL_KEYS if key not in residuals]
        if missing:
            raise ValueError(f"residuals is missing required keys: {missing}.")
        for key, value in residuals.items():
            if value is None:
                continue
            try:
                float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"residuals[{key!r}] must be numeric, got {value!r}."
                ) from error

    board = d.get("board")
    if board is not None:
        if not isinstance(board, Mapping):
            raise ValueError("board must be a mapping or null.")
        try:
            CharucoBoardConfig(**board)
        except (TypeError, ValueError) as error:
            raise ValueError(f"board is not a valid board config: {error}") from error

    n_observations = d.get("n_observations")
    if n_observations is not None:
        if not isinstance(n_observations, int) or isinstance(n_observations, bool):
            raise ValueError(
                f"n_observations must be an integer or null, got {n_observations!r}."
            )
        if n_observations < 0:
            raise ValueError(f"n_observations must be >= 0, got {n_observations}.")

    _parse_created(d.get("created"))
