from typing import Any, Optional, Dict, List, Tuple
from unienv_interface.env_base.env import Env
from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.utils.space_data_queue import SpaceDataQueue


class DataCollectionEnvWrapper(Wrapper):
    """
    Environment wrapper for collecting trajectory data.

    Wraps any UniEnv environment to collect transition data during interaction.
    Works with both single environments (batch_size=None) and batched
    environments (batch_size=int).

    Uses SpaceDataQueue for efficient data collection with support for
    partial resets in batched environments.
    
    When used with a TrajectoryReplayBuffer, episode context will be stored
    along with step data.
    """

    def __init__(
        self,
        env: Env,
        additional_data_spaces: Optional[Dict[str, Any]] = None,
        store_on_step: bool = True,
        batch: Optional[Any] = None,
    ):
        """
        Initialize the data collection wrapper.

        Args:
            env: The base UniEnv environment to wrap
            additional_data_spaces: Additional data fields to collect beyond
                                   observation, action, reward, terminated, truncated
                                   e.g., {'context': env.context_space}
            store_on_step: Whether to store data immediately on each step
            batch: Optional batch storage (e.g., TrajectoryReplayBuffer). If provided,
                   episode context will be stored when using TrajectoryReplayBuffer.
        """
        super().__init__(env)
        self.store_on_step = store_on_step
        self._batch = batch
        
        # Check if batch is a TrajectoryReplayBuffer
        self._is_trajectory_buffer = self._check_is_trajectory_buffer(batch)

        # batch_size from inner env: None = single, int = batched
        self._num_envs = 1 if env.batch_size is None else env.batch_size

        # Setup data collection spaces
        self._data_spaces = {
            "observation": env.observation_space,
            "action": env.action_space,
            "reward": None,  # Scalar, handled specially
            "terminated": None,  # Scalar boolean
            "truncated": None,  # Scalar boolean
        }
        if additional_data_spaces:
            self._data_spaces.update(additional_data_spaces)

        # Initialize SpaceDataQueue for collecting transitions
        self._data_queue = SpaceDataQueue(
            spaces=self._data_spaces,
            batch_size=self.batch_size,
        )

        # Track previous observation for transition construction
        self._prev_obs: Optional[Any] = None
        self._prev_context: Optional[Any] = None
        
        # Track step_id for each environment (for batched envs)
        self._step_ids: Optional[List[int]] = None
    
    def _check_is_trajectory_buffer(self, batch: Optional[Any]) -> bool:
        """Check if the provided batch is a TrajectoryReplayBuffer."""
        if batch is None:
            return False
        # Avoid circular imports by checking class name
        return type(batch).__name__ == "TrajectoryReplayBuffer"

    def reset(self, *, mask=None, seed=None, **kwargs):
        """
        Reset environment(s) and reset data queue.

        For batched envs, handles partial resets via mask parameter.
        """
        context, obs, info = self.env.reset(mask=mask, seed=seed, **kwargs)

        # Reset data queue (with mask for partial resets)
        self._data_queue.reset(mask=mask)

        # Initialize or reset step_ids
        if self._step_ids is None:
            self._step_ids = [0] * self._num_envs
        
        if mask is not None:
            # Partial reset: reset step_ids for masked environments
            mask_arr = self.backend.asarray(mask)
            reset_indices = self.backend.nonzero(mask_arr).tolist()
            for idx in reset_indices:
                self._step_ids[idx] = 0
        else:
            # Full reset: reset all step_ids
            self._step_ids = [0] * self._num_envs

        # Store initial observation
        self._prev_obs = obs
        self._prev_context = context

        return context, obs, info

    def step(self, action):
        """
        Execute action(s) and collect transition data.

        Args:
            action: Single action (single env) or batched actions (batched env)

        Returns:
            Standard UniEnv step return: (obs, reward, terminated, truncated, info)
        """
        # Execute step in wrapped environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.store_on_step:
            # Build transition data
            transition = {
                "observation": self._prev_obs,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }

            # Add context if available
            if self._prev_context is not None:
                transition["context"] = self._prev_context

            # Add any additional fields from info
            transition["info"] = info

            # Add step_id for each environment
            if self.batch_size is None:
                assert self._step_ids is not None
                transition["step_id"] = self._step_ids[0]
                self._step_ids[0] += 1
            else:
                # For batched envs, step_id is a list/array
                assert self._step_ids is not None
                transition["step_id"] = self.backend.asarray(self._step_ids, dtype=self.backend.default_integer_dtype, device=self.device)
                # Increment step_ids for environments that are not done
                done = self.backend.logical_or(terminated, truncated)
                for i in range(self._num_envs):
                    if not done[i]:
                        self._step_ids[i] += 1
                    else:
                        self._step_ids[i] = 0

            # Add to queue
            self._data_queue.add(transition)
            
            # If using TrajectoryReplayBuffer and episode ended, store episode context
            if self._is_trajectory_buffer and self._batch is not None:
                self._store_episode_context_if_done(terminated, truncated)

            # Store current obs for next transition
            self._prev_obs = obs

        return obs, reward, terminated, truncated, info
    
    def _store_episode_context_if_done(self, terminated, truncated) -> None:
        """
        Store episode context in TrajectoryReplayBuffer if episodes have ended.
        
        This method checks if any environments have finished episodes and stores
        the episode context (if available) in the TrajectoryReplayBuffer.
        """
        assert self._batch is not None, "Batch must be provided to store episode context"
        
        if self.batch_size is None:
            # Single environment
            done = terminated or truncated
            if done and self._prev_context is not None:
                # Type ignore: we know _batch has these methods because _is_trajectory_buffer is True
                self._batch.set_current_episode_data(self._prev_context)  # type: ignore
                self._batch.mark_episode_end()  # type: ignore
        else:
            # Batched environment
            done = self.backend.logical_or(terminated, truncated)
            if self.backend.any(done):
                # For batched envs, we need to handle each environment separately
                # Only store episode context if we have it and episodes ended
                if self._prev_context is not None:
                    # Get indices of done environments
                    done_indices = self.backend.nonzero(done).tolist()
                    for idx in done_indices:
                        # Extract context for this specific environment
                        env_context = self._get_env_item(self._prev_context, idx)
                        # Type ignore: we know _batch has these methods because _is_trajectory_buffer is True
                        self._batch.set_current_episode_data(env_context)  # type: ignore
                        self._batch.mark_episode_end()  # type: ignore
    
    def _get_env_item(self, data: Any, idx: int) -> Any:
        """Extract item at index from batched data structure."""
        if isinstance(data, dict):
            return {k: self._get_env_item(v, idx) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return data[idx]
        else:
            # Assume array with batch dimension
            return data[idx]

    def get_trajectories(self, batch_as_list: bool = True):
        """
        Get collected trajectories from the data queue.

        Args:
            batch_as_list: If True (batched env), return list of per-env trajectories

        Returns:
            Collected trajectory data in the format specified by SpaceDataQueue
        """
        return self._data_queue.get_output_data(batch_as_list=batch_as_list)

    def get_transitions(self) -> List[Dict[str, Any]]:
        """
        Get all collected transitions as a flat list.

        Returns:
            List of transition dictionaries
        """
        data = self._data_queue.get_output_data(batch_as_list=True)

        if self.batch_size is None:
            # Single env: data is already a list of transitions
            return data if isinstance(data, list) else []
        else:
            # Batched env: data is list of trajectories, flatten them
            all_transitions = []
            for traj in data:
                all_transitions.extend(traj)
            return all_transitions

    def clear(self):
        """Clear all stored data."""
        self._data_queue.clear()
        self._prev_obs = None
        self._prev_context = None
