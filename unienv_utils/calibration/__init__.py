"""Unified ChArUco camera calibration utilities.

Importing this package requires OpenCV with the ``aruco`` module, which is not a
hard dependency of UniEnv::

    pip install unienv[calibration]   # pulls opencv-contrib-python

Public API
----------
board
    :class:`CharucoBoardConfig`, :func:`make_board`, :func:`render_board`,
    :func:`resolve_aruco_dict`.
detect
    :class:`CharucoDetection`, :func:`detect_charuco`,
    :func:`estimate_board_pose`.
hand_eye
    :class:`HandEyeObservation`, :class:`HandEyeResult`,
    :func:`solve_hand_to_base`.
camera_pair
    :class:`PairObservation`, :class:`PairResult`,
    :func:`solve_camera_relative`, :func:`transfer_to_base`.
intrinsics
    :class:`IntrinsicsResult`, :func:`calibrate_intrinsics_charuco`.
format
    :func:`save_calibration`, :func:`load_calibration`,
    :func:`validate_calibration_dict`.
transforms
    Transform maths (:func:`invert_T`, :func:`compose`,
    :func:`pose6d_to_T`, ...).
"""

from __future__ import annotations

try:
    import cv2  # noqa: F401
    from cv2 import aruco  # noqa: F401
except (ImportError, AttributeError) as _error:  # pragma: no cover - env dependent
    raise ImportError(
        "unienv_utils.calibration requires OpenCV with the aruco module: "
        "pip install unienv[calibration] (opencv-contrib-python)"
    ) from _error

from unienv_utils.calibration.board import (  # noqa: E402
    CharucoBoardConfig,
    make_board,
    predefined_dictionary,
    render_board,
    resolve_aruco_dict,
)
from unienv_utils.calibration.camera_pair import (  # noqa: E402
    PairObservation,
    PairResult,
    solve_camera_relative,
    transfer_to_base,
)
from unienv_utils.calibration.detect import (  # noqa: E402
    CharucoDetection,
    detect_charuco,
    estimate_board_pose,
    make_detector_parameters,
)
from unienv_utils.calibration.format import (  # noqa: E402
    CALIBRATION_FORMAT,
    INTRINSICS_SOURCES,
    load_calibration,
    save_calibration,
    validate_calibration_dict,
)
from unienv_utils.calibration.hand_eye import (  # noqa: E402
    HandEyeObservation,
    HandEyeResult,
    solve_hand_to_base,
)
from unienv_utils.calibration.intrinsics import (  # noqa: E402
    IntrinsicsResult,
    calibrate_intrinsics_charuco,
)
from unienv_utils.calibration.transforms import (  # noqa: E402
    average_transforms,
    compose,
    invert_T,
    pose6d_to_T,
    rotation_angle_deg,
    T_from_rvec_tvec,
    T_to_pose6d,
    T_to_rvec_tvec,
    validate_T,
)

__all__ = [
    "CALIBRATION_FORMAT",
    "INTRINSICS_SOURCES",
    "CharucoBoardConfig",
    "CharucoDetection",
    "HandEyeObservation",
    "HandEyeResult",
    "IntrinsicsResult",
    "PairObservation",
    "PairResult",
    "T_from_rvec_tvec",
    "T_to_pose6d",
    "T_to_rvec_tvec",
    "average_transforms",
    "calibrate_intrinsics_charuco",
    "compose",
    "detect_charuco",
    "estimate_board_pose",
    "invert_T",
    "load_calibration",
    "make_board",
    "make_detector_parameters",
    "pose6d_to_T",
    "predefined_dictionary",
    "render_board",
    "resolve_aruco_dict",
    "rotation_angle_deg",
    "save_calibration",
    "solve_camera_relative",
    "solve_hand_to_base",
    "transfer_to_base",
    "validate_T",
    "validate_calibration_dict",
]
