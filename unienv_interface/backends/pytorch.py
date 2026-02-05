try:
    from xbarray.backends.pytorch import PyTorchComputeBackend as XBPytorchBackend
except ImportError:
    from xbarray.pytorch import PyTorchComputeBackend as XBPytorchBackend
from xbarray import ComputeBackend

from typing import Union
import torch

PyTorchComputeBackend: ComputeBackend = XBPytorchBackend
PyTorchArrayType = torch.Tensor
PyTorchDeviceType = Union[torch.device, str]
PyTorchDtypeType = torch.dtype
PyTorchRNGType = torch.Generator

__all__ = [
    'PyTorchComputeBackend',
    'PyTorchArrayType',
    'PyTorchDeviceType',
    'PyTorchDtypeType',
    'PyTorchRNGType'
]