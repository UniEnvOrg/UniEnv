try:
    from xbarray.backends.pytorch import PytorchComputeBackend as XBPytorchBackend
except ImportError:
    from xbarray.pytorch import PytorchComputeBackend as XBPytorchBackend
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