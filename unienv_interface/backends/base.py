from typing import Optional, Generic, TypeVar, Dict, Union, Any, Sequence, SupportsFloat, Tuple, Type
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
class ComputeBackend(type, Generic[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    ARRAY_TYPE : Type[BArrayType]
    DEVICE_TYPE : Type[BDeviceType]
    DTYPE_TYPE : Type[BDtypeType]
    RNG_TYPE : Type[BRNGType]
    
    array_api_namespace : Any
    default_integer_dtype : BDtypeType
    default_floating_dtype : BDtypeType
    default_boolean_dtype : BDtypeType

    @abc.abstractmethod
    def is_backendarray(cls, data : Any) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_device_tpu(cls, device : BDeviceType) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def is_device_gpu(cls, device : BDeviceType) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_device_cpu(cls, device : BDeviceType) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def from_numpy(cls, data : np.ndarray, dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> BArrayType:
        raise NotImplementedError

    @abc.abstractmethod
    def to_numpy(cls, data : BArrayType) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def from_other_backend(cls, data : dlpack.DLPackObject, backend : Optional["ComputeBackend"] = None) -> BArrayType:
        raise NotImplementedError

    @abc.abstractmethod
    def to_device(cls, data : BArrayType, device : BDeviceType) -> BArrayType:
        raise NotImplementedError
    
    @classmethod
    def get_device(cls, data : BArrayType) -> BDeviceType:
        return array_api_compat.device(data)

    @abc.abstractmethod
    def replace_inplace(cls, data : BArrayType, index : BArrayType, value : BArrayType) -> BArrayType:
        raise NotImplementedError

    @abc.abstractmethod
    def random_number_generator(cls, seed : Optional[int] = None, device : Optional[BDeviceType] = None) -> BRNGType:
        raise NotImplementedError
    
    @abc.abstractmethod
    def random_discrete_uniform(cls, rng : BRNGType, shape : Sequence[int], from_num : int, to_num : Optional[int], dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> Tuple[BRNGType, BArrayType]:
        """
        Sample from a discrete uniform distribution [from_num, to_num) with shape `shape`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def random_uniform(cls, rng : BRNGType, shape : Sequence[int], lower_bound : float = 0.0, upper_bound : float = 1.0, dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    def __repr__(cls):
        return cls.__name__

    @abc.abstractmethod
    def random_exponential(cls, rng : BRNGType, shape : Sequence[int], lambd : float = 1.0, dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @abc.abstractmethod
    def random_normal(cls, rng : BRNGType, shape : Sequence[int], mean : float = 0.0, std : float = 1.0, dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @abc.abstractmethod
    def random_geometric(cls, rng : BRNGType, shape : Sequence[int], p : float, dtype : Optional[BDtypeType] = None, device : Optional[BDeviceType] = None) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @abc.abstractmethod
    def dtype_is_real_integer(cls, dtype : BDtypeType) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def list_real_integer_dtypes(cls) -> Sequence[BDtypeType]:
        raise NotImplementedError

    @abc.abstractmethod
    def dtype_is_real_floating(cls, dtype : BDtypeType) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def list_real_floating_dtypes(cls) -> Sequence[BDtypeType]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def dtype_is_boolean(cls, dtype : BDtypeType) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def list_boolean_dtypes(cls) -> Sequence[BDtypeType]:
        raise NotImplementedError
    
    @classmethod
    def abbreviate_array(cls, array : BArrayType, try_cast_scalar : bool = True) -> Union[float, int, BArrayType]:
        """
        Abbreivates an array to a single element if possible.
        Or, if some dimensions are the same, abbreviates to a smaller array (but with the same number of dimensions).
        """
        abbr_array = array
        idx = cls.array_api_namespace.zeros(1, dtype=cls.default_integer_dtype, device=cls.get_device(abbr_array))
        for dim_i in range(len(array.shape)):
            first_elem = cls.array_api_namespace.take(abbr_array, idx, axis=dim_i)
            if cls.array_api_namespace.all(abbr_array == first_elem):
                abbr_array = first_elem
            else:
                continue
        if try_cast_scalar:
            if all(i == 1 for i in abbr_array.shape):
                elem = abbr_array[tuple([0] * len(abbr_array.shape))]
                if cls.dtype_is_real_floating(elem.dtype):
                    return float(elem)
                elif cls.dtype_is_real_integer(elem.dtype):
                    return int(elem)
                elif cls.dtype_is_boolean(elem.dtype):
                    return bool(elem)
                else:
                    raise ValueError(f"Abbreviated array element dtype must be a real floating or integer or boolean type, actual dtype: {elem.dtype}")
        else:
            return array