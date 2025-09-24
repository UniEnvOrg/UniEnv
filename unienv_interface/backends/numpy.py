try:
    from xbarray.backends.numpy import NumpyComputeBackend as XBNumpyBackend
except ImportError:
    from xbarray.numpy import NumpyComputeBackend as XBNumpyBackend
from xbarray import ComputeBackend

import numpy as np
from typing import Any

NumpyComputeBackend : ComputeBackend = XBNumpyBackend
NumpyArrayType = np.ndarray
NumpyDeviceType = Any
NumpyDtypeType = np.dtype
NumpyRNGType = np.random.Generator

__all__ = [
    'NumpyComputeBackend',
    'NumpyArrayType',
    'NumpyDeviceType',
    'NumpyDtypeType',
    'NumpyRNGType'
]
