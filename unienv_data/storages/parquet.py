from typing import Optional, Any, Dict, Tuple, Sequence, Union, List
from math import ceil, prod

from unienv_interface.space import Space, BoxSpace, DictSpace, TextSpace, BinarySpace
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.backends import BArrayType
from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType

from unienv_data.base import SpaceStorage, BatchT
from unienv_data.replay_buffer import ReplayBuffer

import numpy as np
import os
import json
import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ========== Type Aliases ==========

ParquetBatchType = Union[Dict[str, Any], NumpyArrayType, str]
ParquetSpaceType = Union[DictSpace, BoxSpace, TextSpace, BinarySpace]

# ========== Helper Functions ==========

def _space_to_column_specs(
    space: ParquetSpaceType,
    default_key: str = "data",
) -> List[Tuple[str, np.dtype, Tuple[int, ...]]]:
    """
    Convert a space to a flat list of column specifications.
    Each spec is (column_name, numpy_dtype, element_shape).

    DictSpace must be flat (single level, no nested DictSpace).
    Non-DictSpace spaces produce a single column named `default_key`.
    """
    if isinstance(space, DictSpace):
        specs = []
        for key, sub_space in space.spaces.items():
            assert not isinstance(sub_space, DictSpace), \
                f"Nested DictSpace is not supported in ParquetStorage. Key '{key}' contains a DictSpace."
            if isinstance(sub_space, (BoxSpace, BinarySpace)):
                dtype = sub_space.dtype or NumpyComputeBackend.default_boolean_dtype
                specs.append((key, np.dtype(dtype), sub_space.shape))
            elif isinstance(sub_space, TextSpace):
                specs.append((key, np.dtype(object), ()))
            else:
                raise ValueError(f"Unsupported space type: {type(sub_space)}")
        return specs
    elif isinstance(space, (BoxSpace, BinarySpace)):
        dtype = space.dtype or NumpyComputeBackend.default_boolean_dtype
        return [(default_key, np.dtype(dtype), space.shape)]
    elif isinstance(space, TextSpace):
        return [(default_key, np.dtype(object), ())]
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


_NUMPY_TO_ARROW = {
    np.dtype(np.float16): pa.float16(),
    np.dtype(np.float32): pa.float32(),
    np.dtype(np.float64): pa.float64(),
    np.dtype(np.int8): pa.int8(),
    np.dtype(np.int16): pa.int16(),
    np.dtype(np.int32): pa.int32(),
    np.dtype(np.int64): pa.int64(),
    np.dtype(np.uint8): pa.uint8(),
    np.dtype(np.uint16): pa.uint16(),
    np.dtype(np.uint32): pa.uint32(),
    np.dtype(np.uint64): pa.uint64(),
    np.dtype(np.bool_): pa.bool_(),
}

_ARROW_TO_NUMPY = {v: k for k, v in _NUMPY_TO_ARROW.items()}


def _numpy_dtype_to_arrow_type(dtype: np.dtype) -> pa.DataType:
    dtype = np.dtype(dtype)
    if dtype in _NUMPY_TO_ARROW:
        return _NUMPY_TO_ARROW[dtype]
    raise ValueError(f"Unsupported numpy dtype for Arrow conversion: {dtype}")


def _arrow_type_to_numpy_dtype(arrow_type: pa.DataType) -> np.dtype:
    if arrow_type in _ARROW_TO_NUMPY:
        return _ARROW_TO_NUMPY[arrow_type]
    raise ValueError(f"Unsupported Arrow type for numpy conversion: {arrow_type}")


def _serialize_column_dtypes(
    column_specs: List[Tuple[str, np.dtype, Tuple[int, ...]]],
) -> Dict[str, str]:
    """Serialize column dtypes for metadata persistence."""
    return {name: np.dtype(dtype).name for name, dtype, _ in column_specs}


def _deserialize_column_dtypes(
    raw_column_dtypes: Dict[str, str],
) -> Dict[str, np.dtype]:
    """Deserialize metadata dtype strings to numpy dtypes."""
    return {name: np.dtype(dtype_name) for name, dtype_name in raw_column_dtypes.items()}


def _build_arrow_schema(
    column_specs: List[Tuple[str, np.dtype, Tuple[int, ...]]],
) -> pa.Schema:
    """Build a pyarrow schema from column specifications."""
    fields = []
    for col_name, dtype, shape in column_specs:
        if dtype == np.dtype(object):
            # TextSpace
            arrow_type = pa.string()
        else:
            base_type = _numpy_dtype_to_arrow_type(dtype)
            n_elements = prod(shape) if shape else 0
            if n_elements > 1:
                arrow_type = pa.list_(base_type, n_elements)
            elif n_elements == 1:
                # Shape like (1,) or (1,1) — still a list for consistency
                arrow_type = pa.list_(base_type, 1)
            else:
                # Scalar (shape == ())
                arrow_type = base_type
        fields.append(pa.field(col_name, arrow_type))
    return pa.schema(fields)


def _numpy_arrays_to_arrow_table(
    data_dict: Dict[str, np.ndarray],
    column_specs: List[Tuple[str, np.dtype, Tuple[int, ...]]],
    num_rows: int,
    column_shapes_metadata: Dict[str, List[int]],
) -> pa.Table:
    """Convert a dict of numpy arrays to a pyarrow Table."""
    schema = _build_arrow_schema(column_specs)
    arrays = []
    for col_name, dtype, shape in column_specs:
        arr = data_dict[col_name][:num_rows]
        if dtype == np.dtype(object):
            # TextSpace — convert to list of strings
            arrays.append(pa.array(arr.tolist(), type=pa.string()))
        else:
            n_elements = prod(shape) if shape else 0
            if n_elements >= 1:
                # Flatten to 2D (num_rows, n_elements), then convert to list of lists
                flat = arr.reshape(num_rows, n_elements)
                arrays.append(pa.FixedSizeListArray.from_arrays(
                    pa.array(flat.ravel(), type=_numpy_dtype_to_arrow_type(dtype)),
                    list_size=n_elements,
                ))
            else:
                # Scalar
                arrays.append(pa.array(arr, type=_numpy_dtype_to_arrow_type(dtype)))

    # Store column shapes in metadata
    metadata = {
        b"column_shapes": json.dumps(column_shapes_metadata).encode("utf-8"),
    }
    schema = schema.with_metadata(metadata)
    return pa.table(arrays, schema=schema)


def _arrow_table_to_numpy_arrays(
    table: pa.Table,
    column_shapes: Dict[str, Tuple[int, ...]],
) -> Dict[str, np.ndarray]:
    """Convert a pyarrow Table to a dict of numpy arrays."""
    result = {}
    num_rows = table.num_rows
    for col_name in table.column_names:
        column = table.column(col_name)
        shape = column_shapes.get(col_name, ())

        if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
            # TextSpace
            result[col_name] = np.array(column.to_pylist(), dtype=object)
        elif pa.types.is_fixed_size_list(column.type):
            # Multi-element: extract flat values, reshape
            n_elements = column.type.list_size
            # ChunkedArray needs combine_chunks() to get a plain Array with .values
            arr = column.combine_chunks()
            flat_values = arr.values.to_numpy(zero_copy_only=False)
            result[col_name] = flat_values.reshape(num_rows, *shape)
        else:
            # Scalar
            result[col_name] = column.to_numpy(zero_copy_only=False)
    return result


def _piece_filename(piece_idx: int) -> str:
    return f"part-{piece_idx:06d}.parquet"


# ========== ParquetStorage ==========

class ParquetStorage(SpaceStorage[
    ParquetBatchType,
    NumpyArrayType,
    NumpyDeviceType,
    NumpyDtypeType,
    NumpyRNGType,
]):
    # ========== Class Attributes ==========
    single_file_ext: Optional[str] = None  # Directory-based storage
    DEFAULT_KEY: str = "data"

    @classmethod
    def create(
        cls,
        single_instance_space: ParquetSpaceType,
        *args,
        capacity: Optional[int] = None,
        cache_path: Optional[Union[str, os.PathLike]] = None,
        multiprocessing: bool = False,
        piece_size: int = 1024,
        compression: Union[str, Dict[str, str], None] = "snappy",
        **kwargs,
    ) -> "ParquetStorage":
        assert cache_path is not None, \
            "cache_path must be provided for ParquetStorage"
        assert piece_size > 0, "piece_size must be greater than 0"

        storage_dir = str(cache_path)
        os.makedirs(storage_dir, exist_ok=True)

        column_specs = _space_to_column_specs(single_instance_space, cls.DEFAULT_KEY)
        column_shapes = {name: shape for name, _, shape in column_specs}
        column_dtypes = _serialize_column_dtypes(column_specs)

        if capacity is not None:
            # Fixed capacity: pre-create all piece files with zero-initialized data
            num_pieces = ceil(capacity / piece_size)
            for piece_idx in range(num_pieces):
                start = piece_idx * piece_size
                end = min(start + piece_size, capacity)
                piece_rows = end - start
                piece_data = _allocate_piece_arrays(column_specs, piece_rows)
                _write_piece_to_disk(
                    storage_dir, piece_idx, piece_data, column_specs,
                    piece_rows, column_shapes, compression,
                )
            total_len = capacity
        else:
            # Unlimited capacity: start empty
            num_pieces = 0
            total_len = 0

        # Write metadata
        _write_metadata(
            storage_dir,
            piece_size,
            total_len,
            column_shapes,
            compression,
            column_dtypes=column_dtypes,
        )

        return cls(
            single_instance_space,
            storage_dir=storage_dir,
            capacity=capacity,
            piece_size=piece_size,
            num_pieces=num_pieces,
            total_len=total_len,
            column_specs=column_specs,
            column_shapes=column_shapes,
            read_only=False,
            compression=compression,
        )

    @classmethod
    def load_from(
        cls,
        path: Union[str, os.PathLike],
        single_instance_space: ParquetSpaceType,
        *,
        capacity: Optional[int] = None,
        read_only: bool = True,
        multiprocessing: bool = False,
        **kwargs,
    ) -> "ParquetStorage":
        storage_dir = str(path)
        assert os.path.isdir(storage_dir), \
            f"Path {storage_dir} is not a directory"

        metadata = _read_metadata(storage_dir)
        piece_size = metadata["piece_size"]
        total_len = metadata["num_rows"]
        column_shapes_raw = metadata["column_shapes"]
        column_shapes = {k: tuple(v) for k, v in column_shapes_raw.items()}
        compression = metadata.get("compression", "snappy")
        assert piece_size > 0, "Corrupted metadata: piece_size must be greater than 0"

        column_specs = _space_to_column_specs(single_instance_space, cls.DEFAULT_KEY)

        # Use metadata-backed piece count to avoid being affected by stale files.
        num_pieces = ceil(total_len / piece_size) if total_len > 0 else 0
        for piece_idx in range(num_pieces):
            assert os.path.exists(os.path.join(storage_dir, _piece_filename(piece_idx))), \
                f"Missing parquet piece file: {_piece_filename(piece_idx)}"

        return cls(
            single_instance_space,
            storage_dir=storage_dir,
            capacity=capacity,
            piece_size=piece_size,
            num_pieces=num_pieces,
            total_len=total_len,
            column_specs=column_specs,
            column_shapes=column_shapes,
            read_only=read_only,
            compression=compression,
        )

    @staticmethod
    def build_space_from_parquet_file(
        path: Union[str, os.PathLike],
    ) -> Tuple[int, Optional[int], ParquetSpaceType]:
        """
        Infer space, count, and capacity from a parquet storage directory.
        Returns (count, capacity, space).
        """
        storage_dir = str(path)
        metadata = _read_metadata(storage_dir)
        count = metadata["num_rows"]
        column_shapes_raw = metadata["column_shapes"]
        column_shapes = {k: tuple(v) for k, v in column_shapes_raw.items()}
        column_dtypes_raw = metadata.get("column_dtypes")

        piece_files = sorted([
            f for f in os.listdir(storage_dir)
            if f.startswith("part-") and f.endswith(".parquet")
        ])

        if len(piece_files) > 0:
            # Read the schema from the first piece.
            pf = pq.ParquetFile(os.path.join(storage_dir, piece_files[0]))
            schema = pf.schema_arrow
            space = _schema_to_space(schema, column_shapes, ParquetStorage.DEFAULT_KEY)
        else:
            assert column_dtypes_raw is not None, \
                "No piece files found in storage directory and metadata lacks column_dtypes for space inference"
            column_dtypes = _deserialize_column_dtypes(column_dtypes_raw)
            space = _column_metadata_to_space(
                column_dtypes,
                column_shapes,
                ParquetStorage.DEFAULT_KEY,
            )
        return count, count, space

    @staticmethod
    def load_replay_buffer_from_raw_parquet(
        path: Union[str, os.PathLike],
    ) -> ReplayBuffer[ParquetBatchType, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]:
        count, capacity, single_instance_space = ParquetStorage.build_space_from_parquet_file(path)
        storage = ParquetStorage.load_from(
            path,
            single_instance_space,
            capacity=None,
            read_only=True,
        )
        return ReplayBuffer(
            storage,
            storage_path_relative="storage",
            count=count,
            offset=0,
            cache_path=None,
        )

    # ========== Instance Methods ==========

    def __init__(
        self,
        single_instance_space: ParquetSpaceType,
        storage_dir: str,
        capacity: Optional[int],
        piece_size: int,
        num_pieces: int,
        total_len: int,
        column_specs: List[Tuple[str, np.dtype, Tuple[int, ...]]],
        column_shapes: Dict[str, Tuple[int, ...]],
        read_only: bool = False,
        compression: Union[str, Dict[str, str], None] = "snappy",
    ):
        super().__init__(single_instance_space)
        self._batched_instance_space = sbu.batch_space(
            self.single_instance_space, 1
        )
        self._storage_dir = storage_dir
        self.capacity = capacity
        self._piece_size = piece_size
        self._num_pieces = num_pieces
        self._len = total_len
        self._column_specs = column_specs
        self._column_shapes = column_shapes
        self._read_only = read_only
        self._compression = compression

        # Piece cache: piece_idx -> {col_name: np.ndarray}
        self._cached_pieces: Dict[int, Dict[str, np.ndarray]] = {}
        # Read-only Arrow cache: piece_idx -> pyarrow.Table
        self._cached_piece_tables: Dict[int, pa.Table] = {}
        self._dirty_pieces: set = set()

    @property
    def is_mutable(self) -> bool:
        return not self._read_only

    @property
    def is_multiprocessing_safe(self) -> bool:
        return not self.is_mutable

    @property
    def cache_filename(self) -> Optional[Union[str, os.PathLike]]:
        return self._storage_dir

    def __len__(self) -> int:
        return self._len

    # ========== Piece Management ==========

    def _piece_path(self, piece_idx: int) -> str:
        return os.path.join(self._storage_dir, _piece_filename(piece_idx))

    def _piece_row_range(self, piece_idx: int) -> Tuple[int, int]:
        """Return (start_row, end_row) for a given piece index."""
        start = piece_idx * self._piece_size
        end = min(start + self._piece_size, self._len)
        return start, end

    def _load_piece(self, piece_idx: int) -> Dict[str, np.ndarray]:
        """Load a piece from disk into cache if not already cached."""
        if piece_idx in self._cached_pieces:
            return self._cached_pieces[piece_idx]

        piece_path = self._piece_path(piece_idx)
        table = pq.read_table(piece_path, memory_map=self._read_only)
        arrays = _arrow_table_to_numpy_arrays(table, self._column_shapes)

        # For mutable mode, ensure arrays are writable copies
        if not self._read_only:
            arrays = {k: np.array(v) for k, v in arrays.items()}

        self._cached_pieces[piece_idx] = arrays
        return arrays

    def _load_piece_table(self, piece_idx: int) -> pa.Table:
        """Load a piece as an Arrow table for read-only access."""
        if piece_idx in self._cached_piece_tables:
            return self._cached_piece_tables[piece_idx]
        piece_path = self._piece_path(piece_idx)
        table = pq.read_table(piece_path, memory_map=True)
        self._cached_piece_tables[piece_idx] = table
        return table

    def _read_only_piece_column_values(
        self,
        piece_idx: int,
        col_name: str,
        local_idxs: np.ndarray,
        dtype: np.dtype,
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Fetch only selected rows from a read-only piece column."""
        table = self._load_piece_table(piece_idx)
        column = table.column(col_name)
        take_indices = pa.array(local_idxs, type=pa.int64())
        taken = pc.take(column, take_indices)

        if isinstance(taken, pa.ChunkedArray):
            taken = taken.combine_chunks()

        if dtype == np.dtype(object):
            return np.array(taken.to_pylist(), dtype=object)
        if pa.types.is_fixed_size_list(taken.type):
            flat_values = taken.values.to_numpy(zero_copy_only=False)
            return flat_values.reshape(local_idxs.shape[0], *shape)
        return taken.to_numpy(zero_copy_only=False)

    def _flush_piece(self, piece_idx: int) -> None:
        """Write a dirty piece back to disk."""
        if piece_idx not in self._dirty_pieces:
            return
        if piece_idx not in self._cached_pieces:
            return

        piece_data = self._cached_pieces[piece_idx]
        start, end = self._piece_row_range(piece_idx)
        piece_rows = end - start

        _write_piece_to_disk(
            self._storage_dir, piece_idx, piece_data,
            self._column_specs, piece_rows,
            self._column_shapes, self._compression,
        )
        self._dirty_pieces.discard(piece_idx)

    def _flush_all_dirty(self) -> None:
        """Flush all dirty pieces to disk."""
        for piece_idx in list(self._dirty_pieces):
            self._flush_piece(piece_idx)

    def _index_to_piece_mapping(
        self, index: Union[int, slice, np.ndarray, Sequence[int]],
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Map a storage-level index to per-piece local indices.

        Returns:
            Dict mapping piece_idx -> (local_indices, output_positions)
            where local_indices are row indices within the piece
            and output_positions are positions in the final output array
        """
        # Normalize index to a flat array of global row indices
        if index is Ellipsis:
            global_indices = np.arange(self._len)
        elif isinstance(index, (int, np.integer)):
            idx = int(index)
            if idx < 0:
                idx += self._len
            assert 0 <= idx < self._len, \
                f"Index {index} out of bounds for storage of length {self._len}"
            global_indices = np.array([idx], dtype=np.int64)
        elif isinstance(index, slice):
            global_indices = np.arange(*index.indices(self._len))
        elif isinstance(index, np.ndarray) or (
            isinstance(index, Sequence) and not isinstance(index, (str, bytes))
        ):
            arr = np.asarray(index)
            if arr.dtype.kind == "b":
                assert arr.ndim == 1 and arr.shape[0] == self._len, \
                    f"Boolean index must be 1D with length {self._len}, got shape {arr.shape}"
                global_indices = np.nonzero(arr)[0].astype(np.int64)
            else:
                global_indices = arr.ravel().astype(np.int64)
        else:
            raise ValueError(f"Unsupported index type: {type(index)}")

        if global_indices.size == 0:
            return {}

        # Normalize and validate array indices.
        negative_mask = global_indices < 0
        if np.any(negative_mask):
            global_indices = global_indices.copy()
            global_indices[negative_mask] += self._len
        assert np.all(global_indices >= 0) and np.all(global_indices < self._len), \
            f"Index values out of bounds for storage of length {self._len}: {global_indices}"

        # Map each global index to its piece
        piece_indices = global_indices // self._piece_size
        local_indices = global_indices % self._piece_size

        # Group by piece
        mapping: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        unique_pieces = np.unique(piece_indices)
        for p_idx in unique_pieces:
            mask = piece_indices == p_idx
            mapping[int(p_idx)] = (
                local_indices[mask],
                np.where(mask)[0],
            )
        return mapping

    # ========== get / set ==========

    def _get_from_columns(
        self,
        space: ParquetSpaceType,
        col_data_slices: Dict[str, np.ndarray],
        is_single: bool,
    ) -> ParquetBatchType:
        """
        Build the batch result from column data slices.
        """
        if isinstance(space, DictSpace):
            result = {}
            for key, sub_space in space.spaces.items():
                arr = col_data_slices[key]
                if isinstance(sub_space, TextSpace):
                    if is_single:
                        result[key] = str(arr) if not isinstance(arr, str) else arr
                    else:
                        result[key] = arr
                else:
                    result[key] = arr
            return result
        else:
            col_name = self.DEFAULT_KEY
            arr = col_data_slices[col_name]
            if isinstance(space, TextSpace):
                if is_single:
                    return str(arr) if not isinstance(arr, str) else arr
                return arr
            return arr

    def get(self, index) -> ParquetBatchType:
        is_single = isinstance(index, (int, np.integer))
        mapping = self._index_to_piece_mapping(index)

        if not mapping:
            # Empty result
            return self._empty_result(self.single_instance_space)

        # Determine total output size
        if is_single:
            total_size = 1
        elif index is Ellipsis:
            total_size = self._len
        elif isinstance(index, slice):
            total_size = len(range(*index.indices(self._len)))
        else:
            total_size = np.asarray(index).ravel().size

        # Collect results from each piece
        col_results: Dict[str, List] = {name: [None] * total_size for name, _, _ in self._column_specs}

        for piece_idx, (local_idxs, output_positions) in mapping.items():
            piece_data = None if self._read_only else self._load_piece(piece_idx)
            for col_name, dtype, shape in self._column_specs:
                if self._read_only:
                    values = self._read_only_piece_column_values(
                        piece_idx, col_name, local_idxs, dtype, shape,
                    )
                else:
                    values = piece_data[col_name][local_idxs]
                for i, pos in enumerate(output_positions):
                    col_results[col_name][pos] = values[i]

        # Assemble into proper numpy arrays
        assembled: Dict[str, np.ndarray] = {}
        for col_name, dtype, shape in self._column_specs:
            parts = col_results[col_name]
            if dtype == np.dtype(object):
                arr = np.array(parts, dtype=object)
            else:
                arr = np.stack(parts)
            if is_single:
                arr = arr[0]
            assembled[col_name] = arr

        return self._get_from_columns(
            self.single_instance_space, assembled, is_single
        )

    def set(self, index, value: ParquetBatchType) -> None:
        assert not self._read_only, "Cannot set values on a read-only ParquetStorage"

        is_single = isinstance(index, (int, np.integer))
        mapping = self._index_to_piece_mapping(index)

        # Decompose value into per-column arrays
        col_values: Dict[str, np.ndarray] = {}
        if isinstance(self.single_instance_space, DictSpace):
            for key in self.single_instance_space.spaces:
                v = value[key]
                if is_single:
                    v = np.asarray(v)[np.newaxis]
                col_values[key] = np.asarray(v)
        else:
            v = value
            if is_single:
                v = np.asarray(v)[np.newaxis]
            col_values[self.DEFAULT_KEY] = np.asarray(v)

        for piece_idx, (local_idxs, output_positions) in mapping.items():
            piece_data = self._load_piece(piece_idx)
            for col_name, _, _ in self._column_specs:
                piece_data[col_name][local_idxs] = col_values[col_name][output_positions]
            self._dirty_pieces.add(piece_idx)

    # ========== extend / shrink ==========

    def extend_length(self, length: int) -> None:
        assert not self._read_only, "Cannot extend length of a read-only ParquetStorage"
        assert self.capacity is None, \
            "Cannot extend length of a storage with fixed capacity"
        assert length > 0, "Length must be greater than 0"

        remaining = length
        # If last piece has room, extend it
        if self._num_pieces > 0:
            last_piece_idx = self._num_pieces - 1
            start, end = self._piece_row_range(last_piece_idx)
            current_piece_rows = end - start
            if current_piece_rows < self._piece_size:
                room = self._piece_size - current_piece_rows
                extend_by = min(room, remaining)
                # Load last piece, resize arrays
                piece_data = self._load_piece(last_piece_idx)
                new_piece_rows = current_piece_rows + extend_by
                for col_name, dtype, shape in self._column_specs:
                    old = piece_data[col_name]
                    if dtype == np.dtype(object):
                        new_arr = np.empty((new_piece_rows, *shape), dtype=object)
                    else:
                        new_arr = np.zeros((new_piece_rows, *shape), dtype=dtype)
                    new_arr[:current_piece_rows] = old
                    piece_data[col_name] = new_arr
                self._cached_pieces[last_piece_idx] = piece_data
                self._dirty_pieces.add(last_piece_idx)
                self._len += extend_by
                remaining -= extend_by

        # Create new pieces for remaining rows
        while remaining > 0:
            piece_rows = min(self._piece_size, remaining)
            piece_data = _allocate_piece_arrays(self._column_specs, piece_rows)
            piece_idx = self._num_pieces
            self._num_pieces += 1
            self._cached_pieces[piece_idx] = piece_data
            self._dirty_pieces.add(piece_idx)
            self._len += piece_rows
            remaining -= piece_rows

        # Flush dirty pieces and update metadata
        self._flush_all_dirty()
        _write_metadata(
            self._storage_dir, self._piece_size, self._len,
            self._column_shapes, self._compression,
            column_dtypes=_serialize_column_dtypes(self._column_specs),
        )

    def shrink_length(self, length: int) -> None:
        assert not self._read_only, "Cannot shrink length of a read-only ParquetStorage"
        assert self.capacity is None, \
            "Cannot shrink length of a storage with fixed capacity"
        assert length > 0, "Length must be greater than 0"
        new_len = self._len - length
        assert new_len >= 0, "New length must be non-negative"

        # Remove trailing pieces
        new_num_pieces = ceil(new_len / self._piece_size) if new_len > 0 else 0
        for piece_idx in range(new_num_pieces, self._num_pieces):
            # Remove from cache
            self._cached_pieces.pop(piece_idx, None)
            self._dirty_pieces.discard(piece_idx)
            # Remove file
            piece_path = self._piece_path(piece_idx)
            if os.path.exists(piece_path):
                os.remove(piece_path)

        # Truncate last piece if needed
        if new_num_pieces > 0 and new_len > 0:
            last_piece_idx = new_num_pieces - 1
            expected_rows = new_len - last_piece_idx * self._piece_size
            if last_piece_idx in self._cached_pieces:
                piece_data = self._cached_pieces[last_piece_idx]
                current_rows = next(iter(piece_data.values())).shape[0]
                if current_rows > expected_rows:
                    for col_name in piece_data:
                        piece_data[col_name] = piece_data[col_name][:expected_rows]
                    self._dirty_pieces.add(last_piece_idx)
            else:
                # Load piece and truncate
                piece_data = self._load_piece(last_piece_idx)
                current_rows = next(iter(piece_data.values())).shape[0]
                if current_rows > expected_rows:
                    for col_name in piece_data:
                        piece_data[col_name] = piece_data[col_name][:expected_rows]
                    self._dirty_pieces.add(last_piece_idx)

        self._num_pieces = new_num_pieces
        self._len = new_len

        self._flush_all_dirty()
        _write_metadata(
            self._storage_dir, self._piece_size, self._len,
            self._column_shapes, self._compression,
            column_dtypes=_serialize_column_dtypes(self._column_specs),
        )

    # ========== dumps / close ==========

    def dumps(self, path: Union[str, os.PathLike]) -> None:
        path = str(path)
        if os.path.abspath(path) == os.path.abspath(self._storage_dir):
            # Same directory: just flush dirty pieces
            self._flush_all_dirty()
            _write_metadata(
                self._storage_dir, self._piece_size, self._len,
                self._column_shapes, self._compression,
                column_dtypes=_serialize_column_dtypes(self._column_specs),
            )
        else:
            # Different directory: flush dirty, then copy all pieces
            self._flush_all_dirty()
            os.makedirs(path, exist_ok=True)
            for piece_idx in range(self._num_pieces):
                src = self._piece_path(piece_idx)
                dst = os.path.join(path, _piece_filename(piece_idx))
                shutil.copy2(src, dst)
            _write_metadata(
                path, self._piece_size, self._len,
                self._column_shapes, self._compression,
                column_dtypes=_serialize_column_dtypes(self._column_specs),
            )

    def close(self) -> None:
        if self.is_mutable and self._dirty_pieces:
            self._flush_all_dirty()
        self._cached_pieces.clear()
        self._cached_piece_tables.clear()
        self._dirty_pieces.clear()

    # ========== get_column ==========

    def get_column(self, nested_keys: Sequence[str]) -> "ParquetStorage":
        assert len(nested_keys) == 1, \
            "ParquetStorage only supports flat DictSpace; nested_keys must have exactly one element"
        key = nested_keys[0]
        assert isinstance(self.single_instance_space, DictSpace), \
            "get_column requires a DictSpace at the top level"
        assert key in self.single_instance_space.spaces, \
            f"Key '{key}' not found in DictSpace"

        from unienv_interface.utils.dict_util import nested_get
        sub_space = self.single_instance_space.spaces[key]
        sub_column_specs = _space_to_column_specs(sub_space, self.DEFAULT_KEY)

        # Remap column specs: the sub-storage uses DEFAULT_KEY for the single column
        # but in our piece cache, the data is under the original key
        return _ColumnViewParquetStorage(
            sub_space,
            parent=self,
            source_key=key,
            column_specs=sub_column_specs,
        )

    # ========== Pickling ==========

    def __getstate__(self):
        if self.is_mutable and self._dirty_pieces:
            self._flush_all_dirty()
        state = self.__dict__.copy()
        state.pop("_cached_pieces", None)
        state.pop("_cached_piece_tables", None)
        state.pop("_dirty_pieces", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cached_pieces = {}
        self._cached_piece_tables = {}
        self._dirty_pieces = set()

    # ========== Utilities ==========

    def _empty_result(self, space: ParquetSpaceType):
        """Return an empty batch for the given space."""
        if isinstance(space, DictSpace):
            return {key: self._empty_result(sub) for key, sub in space.spaces.items()}
        elif isinstance(space, TextSpace):
            return np.array([], dtype=object)
        elif isinstance(space, (BoxSpace, BinarySpace)):
            dtype = space.dtype or NumpyComputeBackend.default_boolean_dtype
            return np.array([], dtype=dtype).reshape(0, *space.shape)
        raise ValueError(f"Unsupported space type: {type(space)}")


class _ColumnViewParquetStorage(SpaceStorage[
    ParquetBatchType,
    NumpyArrayType,
    NumpyDeviceType,
    NumpyDtypeType,
    NumpyRNGType,
]):
    """
    A lightweight view over a single column of a ParquetStorage.
    Delegates all operations to the parent storage.
    """
    single_file_ext: Optional[str] = None

    def __init__(
        self,
        single_instance_space: ParquetSpaceType,
        parent: ParquetStorage,
        source_key: str,
        column_specs: List[Tuple[str, np.dtype, Tuple[int, ...]]],
    ):
        super().__init__(single_instance_space)
        self._parent = parent
        self._source_key = source_key
        self._column_specs = column_specs
        self.capacity = parent.capacity

    @property
    def is_mutable(self) -> bool:
        return self._parent.is_mutable

    @property
    def is_multiprocessing_safe(self) -> bool:
        return self._parent.is_multiprocessing_safe

    @property
    def cache_filename(self) -> Optional[Union[str, os.PathLike]]:
        return self._parent.cache_filename

    def __len__(self) -> int:
        return len(self._parent)

    def get(self, index):
        full_result = self._parent.get(index)
        if isinstance(full_result, dict):
            return full_result[self._source_key]
        return full_result

    def set(self, index, value):
        # Need to get the full value, replace the column, then set
        full_value = self._parent.get(index)
        if isinstance(full_value, dict):
            full_value[self._source_key] = value
            self._parent.set(index, full_value)
        else:
            self._parent.set(index, value)

    def dumps(self, path):
        self._parent.dumps(path)

    def close(self):
        pass


# ========== Module-level helpers ==========

def _allocate_piece_arrays(
    column_specs: List[Tuple[str, np.dtype, Tuple[int, ...]]],
    num_rows: int,
) -> Dict[str, np.ndarray]:
    """Allocate zero-initialized numpy arrays for a piece."""
    data = {}
    for col_name, dtype, shape in column_specs:
        if dtype == np.dtype(object):
            data[col_name] = np.empty((num_rows, *shape), dtype=object)
            data[col_name][:] = ""
        else:
            data[col_name] = np.zeros((num_rows, *shape), dtype=dtype)
    return data


def _write_piece_to_disk(
    storage_dir: str,
    piece_idx: int,
    piece_data: Dict[str, np.ndarray],
    column_specs: List[Tuple[str, np.dtype, Tuple[int, ...]]],
    num_rows: int,
    column_shapes: Dict[str, Tuple[int, ...]],
    compression: Union[str, Dict[str, str], None],
) -> None:
    """Write a piece's data to a parquet file."""
    column_shapes_serializable = {k: list(v) for k, v in column_shapes.items()}
    table = _numpy_arrays_to_arrow_table(
        piece_data, column_specs, num_rows, column_shapes_serializable,
    )

    piece_path = os.path.join(storage_dir, _piece_filename(piece_idx))
    if isinstance(compression, dict):
        # Arrow applies per-column compression to physical parquet column paths.
        # Fixed-size/list-encoded tensors use "<name>.list.element" as the physical path.
        resolved_compression: Dict[str, str] = {}
        spec_map = {name: (dtype, shape) for name, dtype, shape in column_specs}
        for col_name, codec in compression.items():
            if col_name in spec_map:
                dtype, shape = spec_map[col_name]
                n_elements = prod(shape) if shape else 0
                if dtype != np.dtype(object) and n_elements >= 1:
                    resolved_compression[f"{col_name}.list.element"] = codec
                else:
                    resolved_compression[col_name] = codec
            else:
                # Keep raw key for callers that already provide physical paths.
                resolved_compression[col_name] = codec
        pq.write_table(table, piece_path, compression=resolved_compression)
    elif compression is not None:
        pq.write_table(table, piece_path, compression=compression)
    else:
        pq.write_table(table, piece_path, compression="NONE")


def _write_metadata(
    storage_dir: str,
    piece_size: int,
    num_rows: int,
    column_shapes: Dict[str, Tuple[int, ...]],
    compression: Union[str, Dict[str, str], None],
    column_dtypes: Optional[Dict[str, str]] = None,
) -> None:
    """Write the storage metadata file."""
    metadata = {
        "piece_size": piece_size,
        "num_rows": num_rows,
        "column_shapes": {k: list(v) for k, v in column_shapes.items()},
        "compression": compression,
    }
    if column_dtypes is not None:
        metadata["column_dtypes"] = column_dtypes
    metadata_path = os.path.join(storage_dir, "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _read_metadata(storage_dir: str) -> dict:
    """Read the storage metadata file."""
    metadata_path = os.path.join(storage_dir, "_metadata.json")
    assert os.path.exists(metadata_path), \
        f"Metadata file not found: {metadata_path}"
    with open(metadata_path, "r") as f:
        return json.load(f)


def _schema_to_space(
    schema: pa.Schema,
    column_shapes: Dict[str, Tuple[int, ...]],
    default_key: str,
) -> ParquetSpaceType:
    """Infer a Space from a parquet schema and column shapes."""
    if len(schema) == 1 and schema.field(0).name == default_key:
        # Single primary space
        return _field_to_space(schema.field(0), column_shapes.get(default_key, ()))
    else:
        # Flat DictSpace
        spaces = {}
        for i in range(len(schema)):
            field = schema.field(i)
            shape = column_shapes.get(field.name, ())
            spaces[field.name] = _field_to_space(field, shape)
        return DictSpace(NumpyComputeBackend, spaces, device=None)


def _column_metadata_to_space(
    column_dtypes: Dict[str, np.dtype],
    column_shapes: Dict[str, Tuple[int, ...]],
    default_key: str,
) -> ParquetSpaceType:
    """Infer a Space from column dtype/shape metadata without parquet piece files."""
    def _dtype_shape_to_space(dtype: np.dtype, shape: Tuple[int, ...]) -> ParquetSpaceType:
        if dtype == np.dtype(object):
            return TextSpace(
                NumpyComputeBackend,
                max_length=4096,
                dtype=str,
                device=None,
            )
        if NumpyComputeBackend.dtype_is_boolean(dtype):
            return BinarySpace(
                NumpyComputeBackend,
                shape=shape,
                dtype=dtype,
                device=None,
            )
        return BoxSpace(
            NumpyComputeBackend,
            low=-np.inf,
            high=np.inf,
            shape=shape,
            dtype=dtype,
            device=None,
        )

    if len(column_dtypes) == 1 and default_key in column_dtypes:
        return _dtype_shape_to_space(
            column_dtypes[default_key],
            column_shapes.get(default_key, ()),
        )

    spaces = {}
    for col_name, dtype in column_dtypes.items():
        spaces[col_name] = _dtype_shape_to_space(dtype, column_shapes.get(col_name, ()))
    return DictSpace(NumpyComputeBackend, spaces, device=None)


def _field_to_space(field: pa.Field, shape: Tuple[int, ...]) -> ParquetSpaceType:
    """Convert a single Arrow field to a Space."""
    arrow_type = field.type

    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return TextSpace(
            NumpyComputeBackend,
            max_length=4096,
            dtype=str,
            device=None,
        )
    elif pa.types.is_boolean(arrow_type):
        return BinarySpace(
            NumpyComputeBackend,
            shape=shape,
            dtype=NumpyComputeBackend.default_boolean_dtype,
            device=None,
        )
    elif pa.types.is_fixed_size_list(arrow_type):
        inner_type = arrow_type.value_type
        if pa.types.is_boolean(inner_type):
            return BinarySpace(
                NumpyComputeBackend,
                shape=shape,
                dtype=NumpyComputeBackend.default_boolean_dtype,
                device=None,
            )
        else:
            numpy_dtype = _arrow_type_to_numpy_dtype(inner_type)
            return BoxSpace(
                NumpyComputeBackend,
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=numpy_dtype,
                device=None,
            )
    else:
        # Scalar numeric type
        numpy_dtype = _arrow_type_to_numpy_dtype(arrow_type)
        if NumpyComputeBackend.dtype_is_boolean(numpy_dtype):
            return BinarySpace(
                NumpyComputeBackend,
                shape=shape,
                dtype=numpy_dtype,
                device=None,
            )
        else:
            return BoxSpace(
                NumpyComputeBackend,
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=numpy_dtype,
                device=None,
            )
