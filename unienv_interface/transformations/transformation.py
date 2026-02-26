import abc
from typing import Generic, TypeVar, Tuple, Dict, Any, Optional, SupportsFloat, Type, Sequence, Union
from unienv_interface.space import Space
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu

SourceDataT = TypeVar("SourceDataT")
SourceBArrT = TypeVar("SourceBArrTypeT")
SourceBDeviceT = TypeVar("TargetBDeviceT")
SourceBDTypeT = TypeVar("TargetBDTypeT")
SourceBDRNGT = TypeVar("TargetBDRNGT")
TargetDataT = TypeVar("TargetDataT")


class DataTransformation(abc.ABC):
    has_inverse: bool = False
    
    @abc.abstractmethod
    def get_target_space_from_source(
        self, source_space: Space[SourceDataT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT]
    ) -> Optional[Space[TargetDataT, BDeviceType, BDtypeType, BRNGType]]:
        """
        Returns the target space based on the source space (if supported, otherwise throws ValueError).
        This is useful for transformations that depend on the source space.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def transform(
        self, 
        source_space: Space[SourceDataT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT],
        data : SourceDataT
    ) -> TargetDataT:
        raise NotImplementedError
    
    def direction_inverse(
        self,
        source_space: Optional[Space[SourceDataT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT]] = None,
    ) -> "Optional[DataTransformation]":
        return None
    
    def close(self):
        pass

    def __del__(self):
        self.close()
    
    @abc.abstractmethod
    def serialize(
        self,
        source_space: Optional[Space[SourceDataT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT]] = None,
    ) -> Dict[str, Any]:
        """
        Serialize the transformation to a JSON-compatible dictionary.

        Args:
            source_space: Optional source space context for transformations whose
                serialization depends on backend/device information.

        Returns:
            dict: A dictionary representation of the transformation containing:
                - Parameters specific to the transformation type.
                  The helper `transformation_to_json` is responsible for adding
                  the serialized transformation type field.
        """
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        source_space: Optional[Space[Any, BDeviceType, BDtypeType, BRNGType]] = None,
    ) -> "DataTransformation":
        """
        Deserialize a transformation from a JSON-compatible dictionary.

        Args:
            json_data: The dictionary containing the transformation data
            source_space: Optional source space context for transformations whose
                deserialization depends on backend/device information.

        Returns:
            DataTransformation: A new instance of the transformation
        """
        raise NotImplementedError
