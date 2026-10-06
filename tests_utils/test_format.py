"""Tests for the canonical ``unienv_calibration/v1`` JSON format."""

from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from unienv_utils.calibration.board import CharucoBoardConfig  # noqa: E402
from unienv_utils.calibration.format import (  # noqa: E402
    CALIBRATION_FORMAT,
    load_calibration,
    save_calibration,
    validate_calibration_dict,
)
from unienv_utils.calibration.intrinsics import IntrinsicsResult  # noqa: E402

T_BASE_CAM = np.array([
    [0.0, -1.0, 0.0, 0.5],
    [1.0, 0.0, 0.0, 0.1],
    [0.0, 0.0, 1.0, 0.9],
    [0.0, 0.0, 0.0, 1.0],
])
RESIDUALS = {
    "translation_mm_mean": 1.2,
    "translation_mm_max": 3.4,
    "rotation_deg_max": 0.35,
}


def _intrinsics_result() -> IntrinsicsResult:
    return IntrinsicsResult(
        K=np.array([[800.0, 0.0, 640.0], [0.0, 810.0, 480.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5),
        image_size=(1280, 960),
        mean_reprojection_error_px=0.42,
        per_view_errors=(0.4, 0.44),
        n_views=2,
    )


def _saved_document(tmp_path, **overrides):
    kwargs = {
        "T_base_cam": T_BASE_CAM,
        "camera_serial": "31726771",
        "solver": "cv2.calibrateHandEye/PARK + eye-to-hand input swap",
        "intrinsics": _intrinsics_result(),
        "residuals": RESIDUALS,
        "board": CharucoBoardConfig.tianji_default(),
        "n_observations": 27,
    }
    kwargs.update(overrides)
    path = tmp_path / "calib.json"
    save_calibration(path, **kwargs)
    return path


def test_save_load_round_trip(tmp_path) -> None:
    path = _saved_document(tmp_path)
    document = load_calibration(path)

    assert document["format"] == CALIBRATION_FORMAT
    assert document["camera_serial"] == "31726771"
    assert document["frame"] == "robot_base"
    assert np.allclose(document["T_base_cam"], T_BASE_CAM)
    assert document["solver"].startswith("cv2.calibrateHandEye")
    assert document["residuals"] == RESIDUALS
    assert document["n_observations"] == 27
    assert document["board"] == CharucoBoardConfig.tianji_default().to_dict()
    assert document["intrinsics"]["K"] == _intrinsics_result().K.tolist()
    assert document["intrinsics"]["distortion"] == [0.0] * 5
    assert document["intrinsics"]["resolution"] == [1280, 960]
    assert document["intrinsics"]["source"] == "charuco_calibrated"
    # ISO-8601 with a timezone offset.
    assert "T" in document["created"]
    assert document["created"][-6] in "+-"


def test_optional_blocks_may_be_absent(tmp_path) -> None:
    document = load_calibration(
        _saved_document(tmp_path, intrinsics=None, residuals=None, board=None)
    )
    assert document["intrinsics"] is None
    assert document["residuals"] is None
    assert document["board"] is None


def test_missing_residual_values_round_trip_as_null(tmp_path) -> None:
    residuals = {
        "translation_mm_mean": None,
        "translation_mm_max": 3.4,
        "rotation_deg_max": 0.35,
    }
    path = _saved_document(tmp_path, residuals=residuals)
    raw = path.read_text()

    assert "NaN" not in raw
    assert '"translation_mm_mean": null' in raw
    # The file is strict JSON: no NaN/Infinity literals anywhere.
    json.dumps(json.loads(raw), allow_nan=False)
    document = load_calibration(path)
    assert document["residuals"] == residuals
    assert document["residuals"]["translation_mm_mean"] is None


def test_save_fails_loudly_for_non_finite_residuals(tmp_path) -> None:
    with pytest.raises(ValueError):
        save_calibration(
            tmp_path / "nan.json",
            T_base_cam=T_BASE_CAM,
            camera_serial="1",
            solver="hand-eye",
            residuals={
                "translation_mm_mean": float("nan"),
                "translation_mm_max": 3.4,
                "rotation_deg_max": 0.35,
            },
        )


def test_load_rejects_unknown_format_version(tmp_path) -> None:
    path = _saved_document(tmp_path)
    raw = json.loads(path.read_text())
    raw["format"] = "unienv_calibration/v2"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Unsupported calibration format"):
        load_calibration(path)


def test_validate_rejects_malformed_transform() -> None:
    path_document = {
        "format": CALIBRATION_FORMAT,
        "camera_serial": "31726771",
        "frame": "robot_base",
        "T_base_cam": np.eye(3).tolist(),
        "solver": "hand-eye",
        "intrinsics": None,
        "residuals": None,
        "board": None,
        "created": "2026-05-17T12:34:56+02:00",
        "n_observations": None,
    }
    with pytest.raises(ValueError, match="valid 4x4 rigid transform"):
        validate_calibration_dict(path_document)

    bad_last_row = dict(path_document)
    bad_last_row["T_base_cam"] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]]
    with pytest.raises(ValueError, match="valid 4x4 rigid transform"):
        validate_calibration_dict(bad_last_row)

    scaled = dict(path_document)
    scaled["T_base_cam"] = (np.eye(4) * 2.0).tolist()
    with pytest.raises(ValueError, match="valid 4x4 rigid transform"):
        validate_calibration_dict(scaled)


def test_save_rejects_invalid_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="valid rigid transform"):
        save_calibration(
            tmp_path / "bad.json",
            T_base_cam=np.eye(3),
            camera_serial="1",
            solver="hand-eye",
        )
    with pytest.raises(ValueError, match="camera_serial"):
        save_calibration(
            tmp_path / "bad.json", T_base_cam=T_BASE_CAM, camera_serial="", solver="x"
        )
    with pytest.raises(ValueError, match="missing required keys"):
        save_calibration(
            tmp_path / "bad.json",
            T_base_cam=T_BASE_CAM,
            camera_serial="1",
            solver="hand-eye",
            residuals={"translation_mm_mean": 1.0},
        )
    with pytest.raises(ValueError, match="must be 3x3"):
        save_calibration(
            tmp_path / "bad.json",
            T_base_cam=T_BASE_CAM,
            camera_serial="1",
            solver="hand-eye",
            intrinsics={"K": [[1.0]], "distortion": [], "resolution": [2, 2]},
        )
    with pytest.raises(ValueError, match="intrinsics must contain"):
        save_calibration(
            tmp_path / "bad.json",
            T_base_cam=T_BASE_CAM,
            camera_serial="1",
            solver="hand-eye",
            intrinsics={"K": np.eye(3).tolist(), "resolution": [2, 2]},
        )


def test_validate_rejects_bad_intrinsics_and_board(tmp_path) -> None:
    document = load_calibration(_saved_document(tmp_path))

    bad_source = json.loads(json.dumps(document))
    bad_source["intrinsics"]["source"] = "vibes"
    with pytest.raises(ValueError, match="intrinsics\\['source'\\]"):
        validate_calibration_dict(bad_source)

    bad_resolution = json.loads(json.dumps(document))
    bad_resolution["intrinsics"]["resolution"] = [1280]
    with pytest.raises(ValueError, match="resolution"):
        validate_calibration_dict(bad_resolution)

    bad_board = json.loads(json.dumps(document))
    bad_board["board"]["marker_size_m"] = 0.05
    with pytest.raises(ValueError, match="board is not a valid board config"):
        validate_calibration_dict(bad_board)

    bad_board_type = json.loads(json.dumps(document))
    bad_board_type["board"] = {"row_count": 9, "unknown_field": 1}
    with pytest.raises(ValueError, match="board is not a valid board config"):
        validate_calibration_dict(bad_board_type)

    bad_created = json.loads(json.dumps(document))
    bad_created["created"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="ISO-8601"):
        validate_calibration_dict(bad_created)

    bad_n = json.loads(json.dumps(document))
    bad_n["n_observations"] = -1
    with pytest.raises(ValueError, match="n_observations"):
        validate_calibration_dict(bad_n)


def test_frame_can_be_overridden(tmp_path) -> None:
    document = load_calibration(
        _saved_document(tmp_path, frame="base_link", camera_serial=12345)
    )
    assert document["frame"] == "base_link"
    assert document["camera_serial"] == "12345"
