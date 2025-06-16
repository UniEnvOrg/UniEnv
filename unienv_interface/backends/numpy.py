from xbarray.numpy import NumpyComputeBackend as XBNumpyBackend
from xbarray import ComputeBackend

NumpyComputeBackend : ComputeBackend = XBNumpyBackend
__all__ = [
    'NumpyComputeBackend',
]
