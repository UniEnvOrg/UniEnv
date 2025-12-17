from importlib import metadata
from typing import Generic, TypeVar, Generic, Optional, Any, Dict, Tuple, Sequence, Union, List, Iterable, Type

from unienv_interface.space import Space, BoxSpace, BinarySpace
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils.symbol_util import *

from unienv_data.base import SpaceStorage
from ._list_storage import ListStorageBase

import numpy as np
import os
import json
import shutil

from PIL import Image

class NPZStorage(ListStorageBase[
    BArrayType,
    BArrayType,
    BDeviceType,
    BDtypeType,
    BRNGType,
]):
    # ========== Class Attributes ==========
    @classmethod
    def create(
        cls,
        single_instance_space: BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType],
        *args,
        compressed : bool = True,
        capacity : Optional[int] = None,
        cache_path : Optional[str] = None,
        multiprocessing : bool = False,
        **kwargs
    ) -> "NPZStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        assert not os.path.exists(cache_path), f"Cache path {cache_path} already exists"
        os.makedirs(cache_path, exist_ok=True)
        return NPZStorage(
            single_instance_space,
            compressed=compressed,
            cache_filename=cache_path,
            capacity=capacity,
        )

    @classmethod
    def load_from(
        cls,
        path : Union[str, os.PathLike],
        single_instance_space : BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType],
        *,
        capacity : Optional[int] = None,
        read_only : bool = True,
        multiprocessing : bool = False,
        **kwargs
    ) -> "NPZStorage[BArrayType, BDeviceType, BDtypeType, BRNGType]":
        metadata_path = os.path.join(path, "npz_metadata.json")
        assert os.path.exists(metadata_path), f"Metadata file {metadata_path} does not exist"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        assert metadata["storage_type"] == cls.__name__, \
            f"Expected storage type {cls.__name__}, but found {metadata['storage_type']}"
        
        compressed = metadata.get("compressed", True)
        if "capacity" in metadata:
            capacity = None if metadata['capacity'] is None else int(metadata["capacity"])
        length = None if capacity is None else metadata["length"]

        return NPZStorage(
            single_instance_space,
            compressed=compressed,
            cache_filename=path,
            mutable=not read_only,
            capacity=capacity,
            length=length,
        )

    # ========== Instance Implementations ==========
    def __init__(
        self,
        single_instance_space: BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType],
        compressed : bool,
        cache_filename : Union[str, os.PathLike],
        mutable : bool = True,
        capacity : Optional[int] = None,
        length : int = 0,
    ):
        assert isinstance(single_instance_space, BoxSpace) or isinstance(single_instance_space, BinarySpace), "single_instance_space must be a BoxSpace or BinarySpace"
        super().__init__(
            single_instance_space,
            file_ext="npz",
            cache_filename=cache_filename,
            mutable=mutable,
            capacity=capacity,
            length=length,
        )
        self.compressed = compressed

    def get_from_file(self, filename : str) -> BArrayType:
        if not os.path.exists(filename):
            return self.backend.zeros(
                self.single_instance_space.shape,
                dtype=self.single_instance_space.dtype,
                device=self.single_instance_space.device,
            )
        
        dat = np.load(filename, allow_pickle=False)
        return self.backend.from_numpy(
            dat['data'], 
            dtype=self.single_instance_space.dtype, 
            device=self.single_instance_space.device
        )
    
    def set_to_file(self, filename : str, value : BArrayType):
        np_value = self.backend.to_numpy(value)
        if self.compressed:
            np.savez_compressed(filename, data=np_value)
        else:
            np.savez(filename, data=np_value)

    def dumps(self, path):
        assert os.path.samefile(path, self.cache_filename), \
            f"Dump path {path} does not match cache filename {self.cache_filename}"
        metadata = {
            "storage_type": __class__.__name__,
            "compressed": self.compressed,
            "capacity": self.capacity,
            "length": self.length,
        }
        with open(os.path.join(path, "npz_metadata.json"), "w") as f:
            json.dump(metadata, f)

    def close(self):
        pass