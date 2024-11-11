from typing import Optional, Generic, TypeVar, Dict, Union, Any, Sequence, SupportsFloat, Tuple, Type
from .base import ComputeBackend
import array_api_compat.torch
import torch
import numpy as np
import dlpack

PYTORCH_DTYPE_CAST_MAP = {
    torch.uint16: torch.int16,
    torch.uint32: torch.int32,
    torch.uint64: torch.int64,
    torch.float8_e4m3fn: torch.float16,
    torch.float8_e5m2: torch.float16,
}
TorchDevice = Union[torch.device, str]

class PyTorchComputeBackend(ComputeBackend[torch.Tensor, TorchDevice, torch.dtype, torch.Generator]):
    array_api_namespace = array_api_compat.torch
    default_integer_dtype = torch.int
    default_floating_dtype = torch.float
    default_boolean_dtype = torch.bool

    @classmethod
    def is_backendarray(cls, data : Any) -> bool:
        return array_api_compat.is_torch_array(data)
    
    @classmethod
    def is_device_tpu(cls, device: TorchDevice) -> bool:
        return torch.device(device).type == "xpu"

    @classmethod
    def is_device_gpu(cls, device: TorchDevice) -> bool:
        return torch.device(device).type == "cuda" or torch.device(device).type == "xpu"
    
    @classmethod
    def is_device_cpu(cls, device: TorchDevice) -> bool:
        return torch.device(device).type == "cpu"

    @classmethod
    def from_numpy(cls, data : np.ndarray, dtype : Optional[torch.dtype] = None, device : Optional[TorchDevice] = None) -> torch.Tensor:
        t = torch.from_numpy(data)
        if dtype is not None or device is not None:
            t = t.to(device=device, dtype=dtype)
        if t.dtype in PYTORCH_DTYPE_CAST_MAP.keys():
            t = t.to(PYTORCH_DTYPE_CAST_MAP[t.dtype])
        return t

    @classmethod
    def to_numpy(cls, data : torch.Tensor) -> np.ndarray:
        # Torch bfloat16 is not supported by numpy
        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32)

        return data.cpu().numpy()

    @classmethod
    def from_other_backend(cls, data : dlpack.DLPackObject, backend : Optional[ComputeBackend] = None) -> torch.Tensor:
        t = torch.from_dlpack(data)
        if t.dtype in PYTORCH_DTYPE_CAST_MAP.keys():
            t = t.to(PYTORCH_DTYPE_CAST_MAP[t.dtype])
        return t

    @classmethod
    def replace_inplace(cls, data: torch.Tensor, index: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        data[index] = value
        return data

    @classmethod
    def random_number_generator(cls, seed : Optional[int] = None, device : Optional[TorchDevice] = None) -> torch.Generator:
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)
        return rng
    
    @classmethod
    def random_discrete_uniform(cls, rng : torch.Generator, shape : Sequence[int], from_num : int, to_num : Optional[int], dtype : Optional[torch.dtype] = None, device : Optional[TorchDevice] = None) -> Tuple[torch.Generator, torch.Tensor]:
        """
        Sample from a discrete uniform distribution [from_num, to_num) with shape `shape`.
        """
        t = torch.zeros(shape, dtype=dtype, device=device)
        t.random_(from_num, to_num, generator=rng)
        return rng, t

    @classmethod
    def random_uniform(cls, rng : torch.Generator, shape : Sequence[int], lower_bound : float = 0.0, upper_bound : float = 1.0, dtype : Optional[torch.dtype] = None, device : Optional[TorchDevice] = None) -> Tuple[torch.Generator, torch.Tensor]:
        t = torch.zeros(shape, dtype=dtype, device=device)
        t.uniform_(lower_bound, upper_bound, generator=rng)
        return rng, t

    @classmethod
    def random_exponential(cls, rng : torch.Generator, shape : Sequence[int], lambd : float = 1.0, dtype : Optional[torch.dtype] = None, device : Optional[TorchDevice] = None) -> Tuple[torch.Generator, torch.Tensor]:
        t = torch.zeros(shape, dtype=dtype, device=device)
        t.exponential_(lambd, generator=rng)
        return rng, t

    @classmethod
    def random_normal(cls, rng : torch.Generator, shape : Sequence[int], mean : float = 0.0, std : float = 1.0, dtype : Optional[torch.dtype] = None, device : Optional[TorchDevice] = None) -> Tuple[torch.Generator, torch.Tensor]:
        t = torch.zeros(shape, dtype=dtype, device=device)
        t.normal_(mean, std, generator=rng)
        return rng, t

    @classmethod
    def random_geometric(cls, rng: torch.Generator, shape: Sequence[int], p: float, dtype: Optional[torch.dtype] = None, device: Optional[TorchDevice] = None) -> Tuple[torch.Generator | torch.Tensor]:
        t = torch.zeros(shape, dtype=dtype, device=device)
        t.geometric_(p, generator=rng)
        return rng, t

    @classmethod
    def dtype_is_real_integer(cls, dtype : torch.dtype) -> bool:
        return dtype in cls.list_real_integer_dtypes()
    
    @classmethod
    def list_real_integer_dtypes(cls) -> Sequence[torch.dtype]:
        # https://pytorch.org/docs/stable/tensors.html#id12
        return (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int, torch.long, int)

    @classmethod
    def dtype_is_real_floating(cls, dtype : torch.dtype) -> bool:
        return dtype in cls.list_real_floating_dtypes()
    
    @classmethod
    def list_real_floating_dtypes(cls) -> Sequence[torch.dtype]:
        return (torch.float16, torch.float32, torch.float64, torch.float, torch.double, torch.bfloat16)
    
    @classmethod
    def dtype_is_boolean(cls, dtype : torch.dtype) -> bool:
        return dtype in cls.list_boolean_dtypes()
    
    @classmethod
    def list_boolean_dtypes(cls) -> Sequence[torch.dtype]:
        return (torch.bool,)