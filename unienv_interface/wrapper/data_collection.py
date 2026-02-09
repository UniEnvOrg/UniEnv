from typing import Any, Optional, Dict, List, Tuple
from unienv_interface.env_base.env import Env
from unienv_interface.utils.space_data_queue import SpaceDataQueue


class DataCollectionEnvWrapper(Env):
    """
    Environment wrapper for collecting trajectory data.

    Wraps any UniEnv environment to collect transition data during interaction.
    Works with both single environments (batch_size=None) and batched
    environments (batch_size=int).

    Uses SpaceDataQueue for efficient data collection with support for
    partial resets in batched environments.
    """

    def __init__(
        self,
        env: Env,
        additional_data_spaces: Optional[Dict[str, Any]] = None,
        store_on_step: bool = True,
    ):
        """
        Initialize the data collection wrapper.

        Args:
            env: The base UniEnv environment to wrap
            additional_data_spaces: Additional data fields to collect beyond
                                   observation, action, reward, terminated, truncated
                                   e.g., {'context': env.context_space}
            store_on_step: Whether to store data immediately on each step
        """
        self.env = env
        self.store_on_step = store_on_step

        # Inherit attributes from wrapped environment
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.context_space = env.context_space
        self.backend = env.backend
        self.device = env.device
        self.metadata = env.metadata
        self.render_mode = env.render_mode
        self.render_fps = env.render_fps
        self.rng = env.rng

        # batch_size from inner env: None = single, int = batched
        self.batch_size = env.batch_size
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

    def reset(self, *, mask=None, seed=None, **kwargs):
        """
        Reset environment(s) and reset data queue.

        For batched envs, handles partial resets via mask parameter.
        """
        context, obs, info = self.env.reset(mask=mask, seed=seed, **kwargs)

        # Reset data queue (with mask for partial resets)
        self._data_queue.reset(mask=mask)

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

            # Add to queue
            self._data_queue.add(transition)

            # Store current obs for next transition
            self._prev_obs = obs

        return obs, reward, terminated, truncated, info

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
            return data
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

    def render(self):
        """Delegate to wrapped environment."""
        return self.env.render()

    def close(self):
        """Close wrapped environment."""
        return self.env.close()
