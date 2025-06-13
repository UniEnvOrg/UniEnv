from xbarray.jax import JaxComputeBackend as XBJaxBackend
from xbarray import ComputeBackend

JaxComputeBackend : ComputeBackend = XBJaxBackend

__all__ = [
    'JaxComputeBackend',
]