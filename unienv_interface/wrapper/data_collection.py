from typing import Any, Optional, Dict, List, Tuple
from unienv_interface.env_base.env import Env
from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.utils.space_data_queue import SpaceDataQueue
from unienv_data.replay_buffer import TrajectoryReplayBuffer
from unienv_interface.space.space_utils import batch_utils as sbu


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
        batch: Any,
        additional_data_spaces: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the data collection wrapper.

        Args:
            env: The base UniEnv environment to wrap
            batch: Batch storage (e.g., TrajectoryReplayBuffer). Episode context will 
                   be stored when using TrajectoryReplayBuffer.
            additional_data_spaces: Additional data fields to collect beyond
                                   observation, action, reward, terminated, truncated
                                   e.g., {'context': env.context_space}
        """
        super().__init__(env)
        self.batch = batch
        
        # Check if batch is a TrajectoryReplayBuffer
        self._is_trajectory_buffer = isinstance(batch, TrajectoryReplayBuffer)

        # Setup data collection spaces
        self._data_spaces = {
            "observation": env.observation_space,
            "action": env.action_space,
            "reward": None,  # Scalar, handled specially
            "terminated": None,  # Scalar boolean
            "truncated": None,  # Scalar boolean
            "step_id": None,  # Scalar integer
        }
        
        # Add episode_id if not using TrajectoryReplayBuffer (it stores its own)
        if not self._is_trajectory_buffer:
            self._data_spaces["episode_id"] = None  # Scalar integer
            
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
        
        # Track step_id for each environment (backend array)
        self._step_ids: Optional[Any] = None
        
        # Track episode_id counter (integer, incremented per episode)
        self._episode_id: int = 0
        
        # Track episode_id for each environment (backend array)
        self._episode_ids: Optional[Any] = None

    def should_save_episode(
        self, 
        env_idx: int, 
        traj_data: List[Dict[str, Any]], 
        truncation: bool, 
        termination: bool, 
        info: Dict[str, Any]
    ) -> bool:
        """
        Determine whether to save an episode's trajectory data.
        
        Override this method to implement custom filtering logic, e.g.,
        only storing successful demonstrations.
        
        Args:
            env_idx: Index of the environment (for batched envs)
            traj_data: List of transition dictionaries for this episode
            truncation: Whether the episode was truncated
            termination: Whether the episode terminated naturally
            info: Info dict from the final step
            
        Returns:
            True if the episode should be saved, False otherwise
        """
        return True

    def reset(self, *, mask=None, seed=None, **kwargs):
        """
        Reset environment(s) and reset data queue.

        For batched envs, handles partial resets via mask parameter.
        """
        context, obs, info = self.env.reset(mask=mask, seed=seed, **kwargs)

        # Reset data queue (with mask for partial resets)
        self._data_queue.reset(mask=mask)

        # Initialize step_ids as backend array
        if self._step_ids is None:
            if self.batch_size is None:
                self._step_ids = self.backend.asarray(0, dtype=self.backend.default_integer_dtype)
            else:
                self._step_ids = self.backend.zeros(
                    (self.batch_size,), 
                    dtype=self.backend.default_integer_dtype, 
                    device=self.device
                )
        
        # Initialize episode_ids as backend array
        if self._episode_ids is None and not self._is_trajectory_buffer:
            if self.batch_size is None:
                self._episode_ids = self.backend.asarray(0, dtype=self.backend.default_integer_dtype)
            else:
                self._episode_ids = self.backend.zeros(
                    (self.batch_size,), 
                    dtype=self.backend.default_integer_dtype, 
                    device=self.device
                )
        
        if mask is not None:
            # Partial reset: reset step_ids for masked environments
            mask_arr = self.backend.asarray(mask)
            reset_indices = self.backend.nonzero(mask_arr)
            for i in range(reset_indices.shape[0]):
                idx = int(reset_indices[i])
                self._step_ids = self.backend.index_update(
                    self._step_ids, 
                    idx, 
                    self.backend.asarray(0, dtype=self.backend.default_integer_dtype)
                )
                if not self._is_trajectory_buffer:
                    self._episode_ids = self.backend.index_update(
                        self._episode_ids,
                        idx,
                        self.backend.asarray(self._episode_id, dtype=self.backend.default_integer_dtype)
                    )
        else:
            # Full reset: reset all step_ids and assign new episode_ids
            if self.batch_size is None:
                self._step_ids = self.backend.asarray(0, dtype=self.backend.default_integer_dtype)
                if not self._is_trajectory_buffer:
                    self._episode_ids = self.backend.asarray(self._episode_id, dtype=self.backend.default_integer_dtype)
            else:
                self._step_ids = self.backend.zeros(
                    (self.batch_size,), 
                    dtype=self.backend.default_integer_dtype, 
                    device=self.device
                )
                if not self._is_trajectory_buffer:
                    self._episode_ids = self.backend.full(
                        (self.batch_size,),
                        self._episode_id,
                        dtype=self.backend.default_integer_dtype,
                        device=self.device
                    )

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

        # Build transition data
        transition = {
            "observation": self._prev_obs,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "step_id": self._step_ids,
        }
        
        # Add episode_id if not using TrajectoryReplayBuffer
        if not self._is_trajectory_buffer:
            transition["episode_id"] = self._episode_ids

        # Add any additional fields from info
        transition["info"] = info

        # Add to queue
        self._data_queue.add(transition)
        
        # Post-step check: store episode context if episodes ended
        self._post_step_check(terminated, truncated, info)

        # Store current obs for next transition
        self._prev_obs = obs

        return obs, reward, terminated, truncated, info
    
    def _post_step_check(self, terminated, truncated, info) -> None:
        """
        Post-step processing: update step_ids, episode_ids, and store episode context.
        
        This method handles:
        - Incrementing step_ids for active episodes
        - Resetting step_ids and updating episode_ids for finished episodes
        - Storing episode context in TrajectoryReplayBuffer when episodes end
        """
        if self.batch_size is None:
            # Single environment
            done = terminated or truncated
            if done:
                # Check if we should save this episode
                if self.should_save_episode(0, [], truncated, terminated, info):
                    # Store episode context if using TrajectoryReplayBuffer
                    if self._is_trajectory_buffer and self._prev_context is not None:
                        self.batch.set_current_episode_data(self._prev_context)
                        self.batch.mark_episode_end()
                
                # Reset step_id and increment episode_id
                self._step_ids = self.backend.asarray(0, dtype=self.backend.default_integer_dtype)
                if not self._is_trajectory_buffer:
                    self._episode_id += 1
                    self._episode_ids = self.backend.asarray(self._episode_id, dtype=self.backend.default_integer_dtype)
            else:
                # Increment step_id
                self._step_ids = self._step_ids + self.backend.asarray(1, dtype=self.backend.default_integer_dtype)
        else:
            # Batched environment
            done = self.backend.logical_or(terminated, truncated)
            
            if self.backend.any(done):
                # Handle finished episodes
                done_indices = self.backend.nonzero(done)
                for i in range(done_indices.shape[0]):
                    idx = int(done_indices[i])
                    
                    # Check if we should save this episode
                    # For batched envs, we don't have the full trajectory yet, so pass empty list
                    # Users can override should_save_episode to use info for filtering
                    if self.should_save_episode(idx, [], bool(truncated[idx]), bool(terminated[idx]), info):
                        # Store episode context if using TrajectoryReplayBuffer
                        if self._is_trajectory_buffer and self._prev_context is not None:
                            env_context = sbu.get_at(self.env.context_space, self._prev_context, idx)
                            self.batch.set_current_episode_data(env_context)
                            self.batch.mark_episode_end()
                    
                    # Reset step_id and assign new episode_id for this env
                    self._step_ids = self.backend.index_update(
                        self._step_ids,
                        idx,
                        self.backend.asarray(0, dtype=self.backend.default_integer_dtype)
                    )
                    if not self._is_trajectory_buffer:
                        self._episode_id += 1
                        self._episode_ids = self.backend.index_update(
                            self._episode_ids,
                            idx,
                            self.backend.asarray(self._episode_id, dtype=self.backend.default_integer_dtype)
                        )
            
            # Increment step_ids for environments that are not done
            self._step_ids = self.backend.where(
                done,
                self._step_ids,
                self._step_ids + self.backend.asarray(1, dtype=self.backend.default_integer_dtype)
            )

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
