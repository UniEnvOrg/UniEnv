from typing import Dict, Any, Optional, Tuple, Union, Generic, TypeVar, List
import dataclasses

from unienv_interface.space import Space
from unienv_interface.backends import (
    ComputeBackend,
    BArrayType,
    BDeviceType,
    BDtypeType,
    BRNGType,
)

DataT = TypeVar("DataT")


@dataclasses.dataclass(frozen=True)
class SpaceDataQueueState(Generic[DataT]):
    """State containing collected data as a list of samples."""

    data: List[
        DataT
    ]  # List of transition dicts: [{'obs': ..., 'action': ..., ...}, ...]

    def replace(self, **changes: Any) -> "SpaceDataQueueState":
        return dataclasses.replace(self, **changes)


class FuncSpaceDataQueue(Generic[DataT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """Functional interface for collecting trajectory data."""

    def __init__(
        self,
        space: Dict[str, Space[Any, BDeviceType, BDtypeType, BRNGType]],
        batch_size: Optional[int],
    ) -> None:
        """
        Initialize data collection queue.

        Args:
            space: Dictionary mapping field names to their spaces
                   e.g., {'observation': obs_space, 'action': action_space, ...}
            batch_size: None for single env, int for batched env
        """
        assert batch_size is None or batch_size > 0, (
            "Batch size must be greater than 0 if provided"
        )

        self.spaces = space
        self._batch_size = batch_size

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def backend(self) -> ComputeBackend:
        # Get backend from any non-None space (they should all use the same)
        for space in self.spaces.values():
            if space is not None:
                return space.backend
        raise ValueError("No valid space found to determine backend")

    @property
    def device(self) -> Optional[BDeviceType]:
        # Get device from any non-None space
        for space in self.spaces.values():
            if space is not None:
                return space.device
        return None

    def init(self) -> SpaceDataQueueState:
        """Initialize empty queue state."""
        return SpaceDataQueueState(data=[])

    def reset(
        self,
        state: SpaceDataQueueState,
        mask: Optional[BArrayType] = None,
    ) -> SpaceDataQueueState:
        """
        Reset the queue.

        For single env: clears all data.
        For batched env with mask: clears only masked environments.
        """
        if self._batch_size is None:
            # Single env: clear all data
            return state.replace(data=[])
        else:
            # Batched env: filter out data from reset environments
            if mask is not None:
                # mask is (B,) boolean array
                # Keep only data where mask[i] is False
                reset_indices = set(self.backend.nonzero(mask).tolist())
                new_data = [
                    sample
                    for idx, sample in enumerate(state.data)
                    if idx not in reset_indices
                ]
                return state.replace(data=new_data)
            else:
                # Full reset
                return state.replace(data=[])

    def add(
        self, state: SpaceDataQueueState, data: Dict[str, DataT]
    ) -> SpaceDataQueueState:
        """
        Add a transition to the queue.

        Args:
            state: Current queue state
            data: Dict with keys matching self.spaces, containing transition data
                  For batched env: each value has shape (B, ...)

        Returns:
            Updated state with new data appended
        """
        if self._batch_size is None:
            # Single env: append single transition dict
            new_data = state.data + [data]
        else:
            # Batched env: append batched transition
            # data has shape (B, ...) for each field
            new_data = state.data + [data]

        return state.replace(data=new_data)

    def get_output_data(
        self, state: SpaceDataQueueState, batch_as_list: bool = True
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Get all collected data.

        Args:
            state: Current queue state
            batch_as_list: If True (batched env), return list of per-env data
                          If False, return stacked tensor

        Returns:
            For single env: List of transition dicts
            For batched env: List of (B,) transition dicts or single stacked dict
        """
        if not batch_as_list or self._batch_size is None:
            return state.data
        else:
            # Batched env: convert list of batched samples to per-env lists
            # state.data is [sample_0, sample_1, ...] where each sample has (B, ...) shape
            # We want: [traj_0, traj_1, ..., traj_B] where traj_i is list of samples for env i
            num_samples = len(state.data)
            if num_samples == 0:
                return [[] for _ in range(self._batch_size)]

            # For each environment, collect its trajectory
            trajectories = []
            for env_idx in range(self._batch_size):
                traj = []
                for sample in state.data:
                    # Extract env_idx from each field
                    env_sample = {}
                    for key, value in sample.items():
                        if isinstance(value, dict):
                            env_sample[key] = self._get_batch_item(value, env_idx)
                        else:
                            env_sample[key] = value[env_idx]
                    traj.append(env_sample)
                trajectories.append(traj)

            return trajectories

    def _get_batch_item(self, data: Any, idx: int) -> Any:
        """Extract item at index from batched data structure."""
        if isinstance(data, dict):
            return {k: self._get_batch_item(v, idx) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return data[idx]
        else:
            # Assume array with batch dimension
            return data[idx]


class SpaceDataQueue(Generic[DataT, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """Stateful wrapper for FuncSpaceDataQueue."""

    def __init__(
        self,
        spaces: Dict[str, Space[Any, BDeviceType, BDtypeType, BRNGType]],
        batch_size: Optional[int],
    ) -> None:
        self.func_queue = FuncSpaceDataQueue(spaces, batch_size)
        self._state: Optional[SpaceDataQueueState] = None

    @property
    def spaces(self) -> Dict[str, Space[Any, BDeviceType, BDtypeType, BRNGType]]:
        return self.func_queue.spaces

    @property
    def batch_size(self) -> Optional[int]:
        return self.func_queue.batch_size

    @property
    def backend(self) -> ComputeBackend:
        return self.func_queue.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.func_queue.device

    def reset(self, mask: Optional[BArrayType] = None) -> None:
        """Reset the queue."""
        if self._state is None:
            assert mask is None, "Mask should not be provided on first reset"
            self._state = self.func_queue.init()
        else:
            self._state = self.func_queue.reset(self._state, mask)

    def add(self, data: Dict[str, Any]) -> None:
        """Add a transition to the queue."""
        assert self._state is not None, "Queue must be reset before adding data"
        self._state = self.func_queue.add(self._state, data)

    def get_output_data(
        self, batch_as_list: bool = True
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Get all collected data."""
        assert self._state is not None, "Queue must be reset before getting data"
        return self.func_queue.get_output_data(self._state, batch_as_list)

    def clear(self) -> None:
        """Clear all collected data."""
        if self._state is not None:
            self._state = self._state.replace(data=[])

    def __len__(self) -> int:
        """Return number of transitions collected."""
        if self._state is None:
            return 0
        return len(self._state.data)
