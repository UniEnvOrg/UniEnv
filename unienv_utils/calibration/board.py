"""ChArUco board configuration, construction and rendering.

The defaults match the board used by the tianji teleoperation stack
(9 x 14 squares, 20 mm checkers, 16 mm markers, ``DICT_5X5_100``), which is the
canonical UniEnv board for hand-eye and multi-camera calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

__all__ = [
    "CharucoBoardConfig",
    "make_board",
    "predefined_dictionary",
    "render_board",
    "resolve_aruco_dict",
]


def resolve_aruco_dict(name: str) -> int:
    """Resolve an ArUco dictionary name (e.g. ``"DICT_5X5_100"``) to cv2's id.

    Raises
    ------
    ValueError
        If ``name`` is not a predefined ``cv2.aruco.DICT_*`` constant.
    """
    if not isinstance(name, str):
        raise ValueError(f"aruco_dict_name must be a string, got {type(name)!r}.")
    key = name.strip().upper()
    if key.startswith("CV2.ARUCO."):
        key = key.rsplit(".", 1)[-1]
    if not key.startswith("DICT_") or not hasattr(cv2.aruco, key):
        raise ValueError(
            f"Unknown ArUco dictionary {name!r}; use a predefined "
            f"cv2.aruco.DICT_* constant such as 'DICT_5X5_100'."
        )
    return int(getattr(cv2.aruco, key))


def predefined_dictionary(name: str) -> "cv2.aruco.Dictionary":
    """Return the ``cv2.aruco`` dictionary object for a dictionary name."""
    dictionary_id = resolve_aruco_dict(name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


@dataclass(frozen=True)
class CharucoBoardConfig:
    """Physical specification of a ChArUco board.

    Attributes
    ----------
    row_count:
        Number of chessboard squares along the short (Y) axis.
    col_count:
        Number of chessboard squares along the long (X) axis.
    checker_size_m:
        Side length of one chessboard square, in metres.
    marker_size_m:
        Side length of one ArUco marker, in metres.  Must be smaller than
        ``checker_size_m``.
    aruco_dict_name:
        Name of a predefined ``cv2.aruco.DICT_*`` constant.
    """

    row_count: int = 9
    col_count: int = 14
    checker_size_m: float = 0.020
    marker_size_m: float = 0.016
    aruco_dict_name: str = "DICT_5X5_100"

    def __post_init__(self) -> None:
        if self.row_count < 2 or self.col_count < 2:
            raise ValueError(
                "row_count and col_count must both be >= 2, got "
                f"{self.row_count} and {self.col_count}."
            )
        if not self.checker_size_m > 0.0:
            raise ValueError(
                f"checker_size_m must be positive, got {self.checker_size_m}."
            )
        if not self.marker_size_m > 0.0:
            raise ValueError(
                f"marker_size_m must be positive, got {self.marker_size_m}."
            )
        if self.marker_size_m >= self.checker_size_m:
            raise ValueError(
                "marker_size_m must be smaller than checker_size_m, got "
                f"{self.marker_size_m} >= {self.checker_size_m}."
            )
        resolve_aruco_dict(self.aruco_dict_name)

    @classmethod
    def tianji_default(cls) -> "CharucoBoardConfig":
        """Return the canonical tianji/UniEnv board configuration.

        Identical to the dataclass defaults; provided explicitly so callers can
        state which physical board they assume.
        """
        return cls()

    @property
    def board_size_m(self) -> tuple[float, float]:
        """Physical ``(width, height)`` of the printed chessboard, in metres."""
        return (
            self.col_count * self.checker_size_m,
            self.row_count * self.checker_size_m,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-serialisable dictionary."""
        return asdict(self)


def make_board(cfg: CharucoBoardConfig) -> "cv2.aruco.CharucoBoard":
    """Build the ``cv2.aruco.CharucoBoard`` described by ``cfg``."""
    dictionary = predefined_dictionary(cfg.aruco_dict_name)
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        # OpenCV < 4.7 naming.
        return cv2.aruco.CharucoBoard_create(
            cfg.col_count,
            cfg.row_count,
            cfg.checker_size_m,
            cfg.marker_size_m,
            dictionary,
        )
    return cv2.aruco.CharucoBoard(
        (cfg.col_count, cfg.row_count),
        cfg.checker_size_m,
        cfg.marker_size_m,
        dictionary,
    )


def render_board(
    cfg: CharucoBoardConfig,
    pixels_per_meter: float,
    margin_px: int = 20,
) -> np.ndarray:
    """Render a printable grayscale image of the ChArUco board.

    Parameters
    ----------
    cfg:
        Board specification.
    pixels_per_meter:
        Print resolution.  Use 5000 for a 20 mm checker to become 100 px, i.e.
        roughly 127 DPI, so the board prints at the right physical size.
    margin_px:
        White border added around the board (ignored by OpenCV < 4.7).

    Returns
    -------
    np.ndarray
        ``uint8`` grayscale image of shape ``(height_px, width_px)``.
    """
    if not pixels_per_meter > 0.0:
        raise ValueError(f"pixels_per_meter must be positive, got {pixels_per_meter}.")
    if margin_px < 0:
        raise ValueError(f"margin_px must be non-negative, got {margin_px}.")
    width_m, height_m = cfg.board_size_m
    width_px = int(round(width_m * pixels_per_meter))
    height_px = int(round(height_m * pixels_per_meter))
    board = make_board(cfg)
    if hasattr(board, "generateImage"):
        image = board.generateImage(
            (width_px, height_px), marginSize=int(margin_px), borderBits=1
        )
    else:
        # OpenCV < 4.7: no margin support, this is what tianji falls back to.
        image = board.draw((width_px, height_px))
    return np.asarray(image, dtype=np.uint8)
