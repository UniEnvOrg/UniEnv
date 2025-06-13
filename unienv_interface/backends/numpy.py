from xbarray.jax import JaxComputeBackend as XBNumpyBackend
from xbarray import ComputeBackend

NumpyComputeBackend : ComputeBackend = XBNumpyBackend
__all__ = [
    'NumpyComputeBackend',
]
