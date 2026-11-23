"""Tests for board configuration, dictionary resolution and rendering."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from unienv_utils.calibration.board import (  # noqa: E402
    CharucoBoardConfig,
    make_board,
    predefined_dictionary,
    render_board,
    resolve_aruco_dict,
)
from unienv_utils.calibration.detect import detect_charuco  # noqa: E402


def test_defaults_match_tianji_board() -> None:
    cfg = CharucoBoardConfig()
    assert cfg.row_count == 9
    assert cfg.col_count == 14
    assert cfg.checker_size_m == 0.020
    assert cfg.marker_size_m == 0.016
    assert cfg.aruco_dict_name == "DICT_5X5_100"
    assert cfg == CharucoBoardConfig.tianji_default()
    assert cfg.board_size_m == pytest.approx((0.28, 0.18))


def test_marker_size_must_be_smaller_than_checker() -> None:
    with pytest.raises(ValueError, match="marker_size_m must be smaller"):
        CharucoBoardConfig(checker_size_m=0.02, marker_size_m=0.02)
    with pytest.raises(ValueError, match="marker_size_m must be smaller"):
        CharucoBoardConfig(checker_size_m=0.02, marker_size_m=0.025)


def test_grid_and_size_validation() -> None:
    with pytest.raises(ValueError, match="row_count and col_count"):
        CharucoBoardConfig(row_count=1, col_count=14)
    with pytest.raises(ValueError, match="row_count and col_count"):
        CharucoBoardConfig(row_count=9, col_count=1)
    with pytest.raises(ValueError, match="checker_size_m must be positive"):
        CharucoBoardConfig(checker_size_m=0.0)
    with pytest.raises(ValueError, match="marker_size_m must be positive"):
        CharucoBoardConfig(marker_size_m=-0.01)


def test_unknown_dictionary_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown ArUco dictionary"):
        CharucoBoardConfig(aruco_dict_name="DICT_NOT_A_DICT")
    with pytest.raises(ValueError, match="aruco_dict_name must be a string"):
        CharucoBoardConfig(aruco_dict_name=4)  # type: ignore[arg-type]


def test_resolve_aruco_dict() -> None:
    assert resolve_aruco_dict("DICT_5X5_100") == int(cv2.aruco.DICT_5X5_100)
    assert resolve_aruco_dict("dict_6x6_250") == int(cv2.aruco.DICT_6X6_250)
    assert resolve_aruco_dict("cv2.aruco.DICT_4X4_50") == int(cv2.aruco.DICT_4X4_50)
    with pytest.raises(ValueError, match="Unknown ArUco dictionary"):
        resolve_aruco_dict("DICT_99X99_1")
    with pytest.raises(ValueError, match="Unknown ArUco dictionary"):
        resolve_aruco_dict("not_a_dict")


def test_make_board_matches_config() -> None:
    cfg = CharucoBoardConfig()
    board = make_board(cfg)
    assert board.getLegacyPattern() is False
    assert board.getSquareLength() == pytest.approx(cfg.checker_size_m)
    assert board.getMarkerLength() == pytest.approx(cfg.marker_size_m)
    assert tuple(board.getChessboardSize()) == (cfg.col_count, cfg.row_count)
    assert predefined_dictionary(cfg.aruco_dict_name) is not None


def test_render_board_has_expected_geometry() -> None:
    cfg = CharucoBoardConfig()
    pixels_per_meter = 5000.0
    image = render_board(cfg, pixels_per_meter)
    width_m, height_m = cfg.board_size_m
    margin = 20  # default margin_px, added on every side
    assert image.dtype == np.uint8
    assert image.ndim == 2
    assert image.shape == (
        int(round(height_m * pixels_per_meter)) + 2 * margin,
        int(round(width_m * pixels_per_meter)) + 2 * margin,
    )
    # The board itself must be dark-on-white ink, not a blank canvas.
    assert image.min() == 0
    assert image.max() == 255
    # The margins are white
    assert image[:margin].min() == 255 and image[-margin:].min() == 255
    assert image[:, :margin].min() == 255 and image[:, -margin:].min() == 255

    half = render_board(cfg, pixels_per_meter / 2.0, margin_px=0)
    assert half.shape == (int(round(height_m * pixels_per_meter / 2)), int(round(width_m * pixels_per_meter / 2)))


@pytest.mark.parametrize("margin_px", [0, 20, 57])
def test_rendered_checkers_have_the_physical_size(margin_px) -> None:
    # Regression: the margins were carved out of the board-sized image, so with the default
    # 20 px margin a 20 mm checker rendered at 5000 px/m was 95.5 px (19.1 mm printed), not 100 px.
    cfg = CharucoBoardConfig()
    pixels_per_meter = 5000.0
    image = render_board(cfg, pixels_per_meter, margin_px=margin_px)
    detection = detect_charuco(image, cfg)
    assert detection is not None
    corners = {int(i): c for i, c in zip(detection.ids.reshape(-1), detection.corners.reshape(-1, 2))}
    inner_cols = cfg.col_count - 1
    pitches = [np.linalg.norm(corners[i + 1] - corners[i]) for i in corners
               if i + 1 in corners and (i + 1) % inner_cols != 0]
    pitches += [np.linalg.norm(corners[i + inner_cols] - corners[i]) for i in corners if i + inner_cols in corners]
    expected_px = cfg.checker_size_m * pixels_per_meter
    np.testing.assert_allclose(pitches, expected_px, atol=0.5)
    # The first inner corner sits one checker inside the margin
    np.testing.assert_allclose(corners[0], [margin_px + expected_px] * 2, atol=1.0)


def test_render_board_validates_inputs() -> None:
    cfg = CharucoBoardConfig()
    with pytest.raises(ValueError, match="pixels_per_meter must be positive"):
        render_board(cfg, 0.0)
    with pytest.raises(ValueError, match="margin_px must be non-negative"):
        render_board(cfg, 5000.0, margin_px=-1)
