from xbarray.pytorch import PytorchComputeBackend as XBPytorchBackend
from xbarray import ComputeBackend

PyTorchComputeBackend: ComputeBackend = XBPytorchBackend

__all__ = [
    'PyTorchComputeBackend',
]