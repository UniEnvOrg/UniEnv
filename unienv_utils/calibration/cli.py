"""Command line interface for the calibration subpackage.

Currently exposes a single utility command::

    python -m unienv_utils.calibration print-board --out /tmp/board.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import cv2

from unienv_utils.calibration.board import CharucoBoardConfig, render_board

__all__ = ["build_parser", "cmd_print_board", "main"]

logger = logging.getLogger("unienv_utils.calibration")

_DEFAULT_PIXELS_PER_METER = 5000.0


def cmd_print_board(args: argparse.Namespace) -> int:
    """Render the printable ChArUco board PNG."""
    cfg = CharucoBoardConfig(
        row_count=args.rows,
        col_count=args.cols,
        checker_size_m=args.checker_mm / 1000.0,
        marker_size_m=args.marker_mm / 1000.0,
        aruco_dict_name=args.aruco_dict,
    )
    image = render_board(cfg, args.pixels_per_meter, margin_px=args.margin_px)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        logger.error("Failed to write %s", out_path)
        return 1
    width_m, height_m = cfg.board_size_m
    logger.info(
        "Wrote %s (%d x %d px, %.1f x %.1f mm at %.0f px/m, %s, %s)",
        out_path,
        image.shape[1],
        image.shape[0],
        width_m * 1000.0,
        height_m * 1000.0,
        args.pixels_per_meter,
        f"{cfg.col_count}x{cfg.row_count} squares",
        cfg.aruco_dict_name,
    )
    logger.info(
        "Print at 100%% scale (no fit-to-page). Verify a checker = %.1f mm "
        "with calipers before calibrating.",
        cfg.checker_size_m * 1000.0,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m unienv_utils.calibration`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m unienv_utils.calibration",
        description="UniEnv ChArUco calibration utilities.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only log warnings and errors."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    print_board = subparsers.add_parser(
        "print-board", help="Render the printable ChArUco board as a PNG."
    )
    print_board.add_argument("--out", required=True, help="Output PNG path.")
    print_board.add_argument(
        "--pixels-per-meter",
        type=float,
        default=_DEFAULT_PIXELS_PER_METER,
        help=f"Print resolution (default {_DEFAULT_PIXELS_PER_METER:.0f}).",
    )
    print_board.add_argument(
        "--rows", type=int, default=9, help="Chessboard squares along Y (default 9)."
    )
    print_board.add_argument(
        "--cols", type=int, default=14, help="Chessboard squares along X (default 14)."
    )
    print_board.add_argument(
        "--checker-mm", type=float, default=20.0, help="Checker size in mm (default 20)."
    )
    print_board.add_argument(
        "--marker-mm", type=float, default=16.0, help="Marker size in mm (default 16)."
    )
    print_board.add_argument(
        "--dict",
        dest="aruco_dict",
        default="DICT_5X5_100",
        help="ArUco dictionary name (default DICT_5X5_100).",
    )
    print_board.add_argument(
        "--margin-px", type=int, default=20, help="White border in pixels (default 20)."
    )
    print_board.set_defaults(handler=cmd_print_board)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``python -m unienv_utils.calibration``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.WARNING if args.quiet else logging.INFO,
        datefmt="%H:%M:%S",
    )
    try:
        return int(args.handler(args))
    except (ValueError, OSError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
