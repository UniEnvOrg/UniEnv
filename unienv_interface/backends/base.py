from typing import Optional, Generic, TypeVar, Dict, Union, Any, Sequence, SupportsFloat, Tuple
import abc
import numpy as np
import dlpack
import gymnasium as gym
import array_api_compat

BArrayType = TypeVar("BArrayType", covariant=True)
# _BDictType = TypeVar("_BDictType", Dict, covariant=True)
BDeviceType = TypeVar("BDeviceType", covariant=True)
BDtypeType = TypeVar("BDtypeType", covariant=True)
BRNGType = TypeVar("BRNGType", covariant=True)
class ComputeBackend(abc.ABC, Generic[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    array_api_namespace : Any
    default_integer_dtype : BDtypeType
    default_floating_dtype : BDtypeType

    @classmethod
    @abc.abstractmethod
    def is_backendarray(cls, data : Any) -> bool:
        raise NotImplementedError
    
    # @classmethod
    # @abc.abstractmethod
    # def is_backenddict(cls, data : Any) -> bool:
    #     raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_numpy(cls, data : np.ndarray, dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> BArrayType:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def to_numpy(cls, data : BArrayType) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_dlpack(cls, data : dlpack.DLPackObject) -> BArrayType:
        raise NotImplementedError

    # @classmethod
    # def from_dict(cls, data : Dict[str,Any]) -> _BDictType:
    #     new_data = {}
    #     for k,v in data.items():
    #         if isinstance(v, np.ndarray):
    #             new_data[k] = cls.from_numpy(v)
    #         elif isinstance(v, dict):
    #             new_data[k] = cls.from_dict(v)
    #     return new_data
    
    # @classmethod
    # def to_dict(cls, data: _BDictType) -> Dict[str, Any]:
    #     new_data = {}
    #     for k,v in data.items():
    #         if cls.is_backendarray(v):
    #             new_data[k] = cls.to_numpy(v)
    #         elif cls.is_backenddict(v):
    #             new_data[k] = cls.to_dict(v)
    #         else:
    #             new_data[k] = v
    #     return new_data

    @classmethod
    @abc.abstractmethod
    def replace_inplace(cls, data : BArrayType, index : BArrayType, value : BArrayType) -> BArrayType:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def random_number_generator(cls, seed : Optional[int] = None, device : Optional[BDeviceType] = None) -> BRNGType:
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def random_discrete_uniform(cls, rng : BRNGType, shape : Sequence[int], from_num : int, to_num : Optional[int], dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> Tuple[BRNGType, BArrayType]:
        """
        Sample from a discrete uniform distribution [from_num, to_num) with shape `shape`.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def random_uniform(cls, rng : BRNGType, shape : Sequence[int], lower_bound : float = 0.0, upper_bound : float = 1.0, dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def random_exponential(cls, rng : BRNGType, shape : Sequence[int], lambd : float = 1.0, dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def random_normal(cls, rng : BRNGType, shape : Sequence[int], mean : float = 0.0, std : float = 1.0, dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def random_geometric(cls, rng : BRNGType, shape : Sequence[int], p : float, dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def dtype_is_real_integer(cls, dtype : BDtypeType) -> bool:
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def dtype_is_real_floating(cls, dtype : BDtypeType) -> bool:
        raise NotImplementedError