"""Import-behaviour tests for ``unienv_utils`` and its calibration subpackage."""

from __future__ import annotations

import subprocess
import sys

import pytest

cv2 = pytest.importorskip("cv2")

import unienv_utils  # noqa: E402

# Simulating a missing OpenCV: ``import cv2`` raises ImportError when the name is
# mapped to ``None`` in ``sys.modules``, which keeps the check free of import hooks.
_MISSING_CV2_SCRIPT = """
import sys
sys.modules["cv2"] = None
try:
    import unienv_utils.calibration
except ImportError as error:
    message = str(error)
    assert "pip install unienv[calibration]" in message, message
    assert "opencv-contrib-python" in message, message
else:
    raise AssertionError("unienv_utils.calibration must not import without cv2")
"""


def _run_python(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_version_constant_is_exposed() -> None:
    assert unienv_utils.__version__ == "0.0.1b14"


def test_top_level_package_does_not_import_cv2() -> None:
    _run_python(
        "import sys; import unienv_utils; "
        "assert 'cv2' not in sys.modules, sorted(sys.modules)"
    )


def test_calibration_import_requires_cv2() -> None:
    _run_python(_MISSING_CV2_SCRIPT)


def test_calibration_package_reexports_public_api() -> None:
    from unienv_utils import calibration

    for name in calibration.__all__:
        assert hasattr(calibration, name), name
    # The two solvers and the format helpers are the entry points users need.
    assert callable(calibration.solve_hand_to_base)
    assert callable(calibration.solve_camera_relative)
    assert callable(calibration.save_calibration)
    assert calibration.CALIBRATION_FORMAT == "unienv_calibration/v1"
