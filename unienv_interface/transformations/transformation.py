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
    """Abstract transformation between two UniEnv data spaces.

    Transformations are used by wrappers, storages, and batch views to adapt
    data while keeping enough metadata to derive the target space and, when
    supported, reconstruct the inverse transform.

    Reassigning an attribute of a transformation increments the global
    ``DataTransformation._config_epoch`` (attributes whose name ends with
    ``_cache`` are exempt), so caches derived from a transformation's
    configuration (e.g. ``ChainedTransformation``'s intermediate spaces) are
    rebuilt. Containers and arrays held by a transformation should be replaced,
    not mutated in place, once it has been used.
    """
    has_inverse: bool = False
    _config_epoch: int = 0

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__ and not name.endswith("_cache"):
            DataTransformation._config_epoch += 1
        super().__setattr__(name, value)
    
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
        """Apply the transformation to data drawn from ``source_space``."""
        raise NotImplementedError
    
    def direction_inverse(
        self,
        source_space: Optional[Space[SourceDataT, SourceBDeviceT, SourceBDTypeT, SourceBDRNGT]] = None,
    ) -> "Optional[DataTransformation]":
        """Return the inverse transformation when one exists."""
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
