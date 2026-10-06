"""General-purpose utilities for UniEnv.

Subpackages
-----------
calibration
    ChArUco board helpers, hand-eye / camera-pair calibration solvers and a
    canonical calibration JSON format.  Requires the optional ``calibration``
    extra (``opencv-contrib-python``); importing this top-level package never
    imports OpenCV.
"""

__version__ = "0.0.1b14"

__all__ = [
    "__version__",
]
