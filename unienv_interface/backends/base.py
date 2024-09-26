from typing import Optional, Generic, TypeVar, Dict, Union, Any, Sequence, SupportsFloat, Tuple
import abc
import numpy as np
import dlpack
import gymnasium as gym
import array_api_compat

_BArrayType = TypeVar("_BArrayType", covariant=True)
_BDictType = TypeVar("_BDictType", Dict, covariant=True)
_BDeviceType = TypeVar("_BDeviceType", covariant=True)
_BDtypeType = TypeVar("_BDtypeType", covariant=True)
_BRNGType = TypeVar("_BRNGType", covariant=True)
class ComputeBackend(abc.ABC, Generic[_BArrayType, _BDictType, _BDeviceType, _BDtypeType, _BRNGType]):
    array_api_namespace : Any

    @classmethod
    @abc.abstractmethod
    def is_backendarray(cls, data : Any) -> bool:
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def is_backenddict(cls, data : Any) -> bool:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_numpy(cls, data : np.ndarray, dtype : Optional[_BDtypeType] = None, device : Optional[_BDeviceType] = None) -> _BArrayType:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def to_numpy(cls, data : _BArrayType) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_dlpack(cls, data : dlpack.DLPackObject) -> _BArrayType:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data : Dict[str,Any]) -> _BDictType:
        new_data = {}
        for k,v in data.items():
            if isinstance(v, np.ndarray):
                new_data[k] = cls.from_numpy(v)
            elif isinstance(v, dict):
                new_data[k] = cls.from_dict(v)
        return new_data
    
    @classmethod
    def to_dict(cls, data: _BDictType) -> Dict[str, Any]:
        new_data = {}
        for k,v in data.items():
            if cls.is_backendarray(v):
                new_data[k] = cls.to_numpy(v)
            elif cls.is_backenddict(v):
                new_data[k] = cls.to_dict(v)
            else:
                new_data[k] = v
        return new_data

    @classmethod
    @abc.abstractmethod
    def random_number_generator(cls, seed : Optional[int] = None, device : Optional[_BDeviceType] = None) -> _BRNGType:
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def random_discrete_uniform(cls, rng : _BRNGType, shape : Sequence[int], from_num : int, to_num : Optional[int], dtype : Optional[_BDtypeType] = None, device : Optional[_BDeviceType] = None) -> Tuple[_BRNGType, _BArrayType]:
        """
        Sample from a discrete uniform distribution [from_num, to_num) with shape `shape`.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def random_uniform(cls, rng : _BRNGType, shape : Sequence[int], lower_bound : float = 0.0, upper_bound : float = 1.0, dtype : Optional[_BDtypeType] = None, device : Optional[_BDeviceType] = None) -> Tuple[_BRNGType, _BArrayType]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def random_exponential(cls, rng : _BRNGType, shape : Sequence[int], lambd : float = 1.0, dtype : Optional[_BDtypeType] = None, device : Optional[_BDeviceType] = None) -> Tuple[_BRNGType, _BArrayType]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def random_normal(cls, rng : _BRNGType, shape : Sequence[int], mean : float = 0.0, std : float = 1.0, dtype : Optional[_BDtypeType] = None, device : Optional[_BDeviceType] = None) -> Tuple[_BRNGType, _BArrayType]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def dtype_is_real_integer(cls, dtype : _BDtypeType) -> bool:
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def dtype_is_real_floating(cls, dtype : _BDtypeType) -> bool:
        raise NotImplementedError