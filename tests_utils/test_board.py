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
    assert image.dtype == np.uint8
    assert image.ndim == 2
    assert image.shape == (
        int(round(height_m * pixels_per_meter)),
        int(round(width_m * pixels_per_meter)),
    )
    # The board itself must be dark-on-white ink, not a blank canvas.
    assert image.min() == 0
    assert image.max() == 255

    half = render_board(cfg, pixels_per_meter / 2.0)
    assert half.shape[0] == image.shape[0] // 2
    assert half.shape[1] == image.shape[1] // 2


def test_render_board_validates_inputs() -> None:
    cfg = CharucoBoardConfig()
    with pytest.raises(ValueError, match="pixels_per_meter must be positive"):
        render_board(cfg, 0.0)
    with pytest.raises(ValueError, match="margin_px must be non-negative"):
        render_board(cfg, 5000.0, margin_px=-1)
