try:
    from xbarray.backends.jax import JaxComputeBackend as XBJaxBackend
except ImportError:
    from xbarray.jax import JaxComputeBackend as XBJaxBackend
from xbarray import ComputeBackend
from typing import Union
import jax

JaxComputeBackend : ComputeBackend = XBJaxBackend
JaxArrayType = jax.Array
JaxDeviceType = Union[jax.Device, jax.sharding.Sharding]
JaxDtypeType = jax.numpy.dtype
JaxRNGType = jax.Array

__all__ = [
    'JaxComputeBackend',
    'JaxArrayType',
    'JaxDeviceType',
    'JaxDtypeType',
    'JaxRNGType'
]