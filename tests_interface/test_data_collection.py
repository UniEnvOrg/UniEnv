"""Tests for SpaceDataQueue and DataCollectionEnvWrapper."""

import pytest
import numpy as np
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace
from unienv_interface.utils.space_data_queue import SpaceDataQueue
from unienv_interface.wrapper.data_collection import DataCollectionEnvWrapper
from unienv_interface.env_base.env import Env


class MockEnv(Env):
    """A simple mock environment for testing."""

    def __init__(self, batch_size=None):
        super().__init__()
        self.batch_size = batch_size
        self.backend = NumpyComputeBackend()
        self.device = None
        self.metadata = {"render_modes": []}
        self.render_mode = None
        self.render_fps = None
        self.rng = None

        # Create simple box spaces
        self._obs_shape = (4,)
        self._action_shape = (2,)

        self.observation_space = BoxSpace(
            self.backend,
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )

        self.action_space = BoxSpace(
            self.backend,
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )

        self.context_space = None

        self._step_count = 0
        self._max_steps = 10

    def reset(self, *, mask=None, seed=None, **kwargs):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        if self.batch_size is None:
            obs = np.random.randn(*self._obs_shape).astype(np.float32)
            context = None
        else:
            obs = np.random.randn(self.batch_size, *self._obs_shape).astype(np.float32)
            context = None

        self._step_count = 0
        info = {}

        return context, obs, info

    def step(self, action):
        """Take a step in the environment."""
        self._step_count += 1

        if self.batch_size is None:
            obs = np.random.randn(*self._obs_shape).astype(np.float32)
            reward = np.random.randn()
            terminated = self._step_count >= self._max_steps
            truncated = False
        else:
            obs = np.random.randn(self.batch_size, *self._obs_shape).astype(np.float32)
            reward = np.random.randn(self.batch_size)
            terminated = np.full(self.batch_size, self._step_count >= self._max_steps)
            truncated = np.full(self.batch_size, False)

        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass


class TestSpaceDataQueue:
    """Test the SpaceDataQueue class."""

    def test_init_single_env(self):
        """Test initialization for single environment."""
        backend = NumpyComputeBackend()
        obs_space = BoxSpace(
            backend,
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )
        action_space = BoxSpace(
            backend,
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )

        spaces = {
            "observation": obs_space,
            "action": action_space,
            "reward": None,
            "terminated": None,
            "truncated": None,
        }

        queue = SpaceDataQueue(spaces=spaces, batch_size=None)

        assert queue.batch_size is None
        assert queue.backend == backend
        assert queue.device is None

    def test_init_batched_env(self):
        """Test initialization for batched environment."""
        backend = NumpyComputeBackend()
        obs_space = BoxSpace(
            backend,
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )
        action_space = BoxSpace(
            backend,
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )

        spaces = {
            "observation": obs_space,
            "action": action_space,
            "reward": None,
            "terminated": None,
            "truncated": None,
        }

        queue = SpaceDataQueue(spaces=spaces, batch_size=4)

        assert queue.batch_size == 4
        assert queue.backend == backend

    def test_single_env_data_collection(self):
        """Test data collection for single environment."""
        backend = NumpyComputeBackend()
        obs_space = BoxSpace(
            backend,
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )
        action_space = BoxSpace(
            backend,
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )

        spaces = {
            "observation": obs_space,
            "action": action_space,
            "reward": None,
            "terminated": None,
            "truncated": None,
        }

        queue = SpaceDataQueue(spaces=spaces, batch_size=None)
        queue.reset()

        # Add some transitions
        for i in range(5):
            transition = {
                "observation": np.random.randn(4).astype(np.float32),
                "action": np.random.randn(2).astype(np.float32),
                "reward": np.random.randn(),
                "terminated": False,
                "truncated": False,
            }
            queue.add(transition)

        assert len(queue) == 5

        data = queue.get_output_data(batch_as_list=True)
        assert len(data) == 5
        assert all("observation" in d for d in data)
        assert all("action" in d for d in data)

    def test_batched_env_data_collection(self):
        """Test data collection for batched environment."""
        backend = NumpyComputeBackend()
        batch_size = 4

        obs_space = BoxSpace(
            backend,
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )
        action_space = BoxSpace(
            backend,
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )

        spaces = {
            "observation": obs_space,
            "action": action_space,
            "reward": None,
            "terminated": None,
            "truncated": None,
        }

        queue = SpaceDataQueue(spaces=spaces, batch_size=batch_size)
        queue.reset()

        # Add some transitions
        for i in range(5):
            transition = {
                "observation": np.random.randn(batch_size, 4).astype(np.float32),
                "action": np.random.randn(batch_size, 2).astype(np.float32),
                "reward": np.random.randn(batch_size),
                "terminated": np.full(batch_size, False),
                "truncated": np.full(batch_size, False),
            }
            queue.add(transition)

        assert len(queue) == 5

        # Test batch_as_list=True
        data = queue.get_output_data(batch_as_list=True)
        assert len(data) == batch_size
        assert all(len(traj) == 5 for traj in data)

    def test_reset_single_env(self):
        """Test reset for single environment."""
        backend = NumpyComputeBackend()
        obs_space = BoxSpace(
            backend,
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )

        spaces = {"observation": obs_space}
        queue = SpaceDataQueue(spaces=spaces, batch_size=None)

        queue.reset()
        queue.add({"observation": np.array([1.0, 2.0, 3.0, 4.0])})
        assert len(queue) == 1

        queue.reset()
        assert len(queue) == 0

    def test_clear(self):
        """Test clearing data."""
        backend = NumpyComputeBackend()
        obs_space = BoxSpace(
            backend,
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
            device=None,
        )

        spaces = {"observation": obs_space}
        queue = SpaceDataQueue(spaces=spaces, batch_size=None)

        queue.reset()
        for i in range(3):
            queue.add({"observation": np.array([1.0, 2.0, 3.0, 4.0])})

        assert len(queue) == 3
        queue.clear()
        assert len(queue) == 0


class TestDataCollectionEnvWrapper:
    """Test the DataCollectionEnvWrapper class."""

    def test_init_single_env(self):
        """Test initialization with single environment."""
        env = MockEnv(batch_size=None)
        wrapped = DataCollectionEnvWrapper(env)

        assert wrapped.env == env
        assert wrapped.batch_size is None
        assert wrapped.backend == env.backend
        assert wrapped.observation_space == env.observation_space
        assert wrapped.action_space == env.action_space

    def test_init_batched_env(self):
        """Test initialization with batched environment."""
        batch_size = 4
        env = MockEnv(batch_size=batch_size)
        wrapped = DataCollectionEnvWrapper(env)

        assert wrapped.env == env
        assert wrapped.batch_size == batch_size
        assert wrapped._num_envs == batch_size

    def test_reset(self):
        """Test reset functionality."""
        env = MockEnv(batch_size=None)
        wrapped = DataCollectionEnvWrapper(env)

        context, obs, info = wrapped.reset()

        assert obs is not None
        assert isinstance(info, dict)
        assert wrapped._prev_obs is not None

    def test_single_step_data_collection(self):
        """Test data collection on single step."""
        env = MockEnv(batch_size=None)
        wrapped = DataCollectionEnvWrapper(env)

        # Reset environment
        context, obs, info = wrapped.reset()

        # Take a step
        action = np.array([0.5, -0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapped.step(action)

        # Check that data was collected
        transitions = wrapped.get_transitions()
        assert len(transitions) == 1
        assert "observation" in transitions[0]
        assert "action" in transitions[0]
        assert "reward" in transitions[0]

    def test_multiple_steps_data_collection(self):
        """Test data collection over multiple steps."""
        env = MockEnv(batch_size=None)
        wrapped = DataCollectionEnvWrapper(env)

        # Reset and take multiple steps
        context, obs, info = wrapped.reset()

        for i in range(5):
            action = np.random.randn(2).astype(np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)

        transitions = wrapped.get_transitions()
        assert len(transitions) == 5

    def test_clear_data(self):
        """Test clearing collected data."""
        env = MockEnv(batch_size=None)
        wrapped = DataCollectionEnvWrapper(env)

        context, obs, info = wrapped.reset()

        for i in range(3):
            action = np.random.randn(2).astype(np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)

        assert len(wrapped.get_transitions()) == 3

        wrapped.clear()

        assert len(wrapped.get_transitions()) == 0
        assert wrapped._prev_obs is None

    def test_render_delegation(self):
        """Test that render is delegated to wrapped environment."""
        env = MockEnv(batch_size=None)
        wrapped = DataCollectionEnvWrapper(env)

        # Should not raise an error
        result = wrapped.render()
        assert result is None

    def test_close_delegation(self):
        """Test that close is delegated to wrapped environment."""
        env = MockEnv(batch_size=None)
        wrapped = DataCollectionEnvWrapper(env)

        # Should not raise an error
        wrapped.close()

    def test_store_on_step_false(self):
        """Test behavior when store_on_step is False."""
        env = MockEnv(batch_size=None)
        wrapped = DataCollectionEnvWrapper(env, store_on_step=False)

        context, obs, info = wrapped.reset()

        action = np.array([0.5, -0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapped.step(action)

        transitions = wrapped.get_transitions()
        assert len(transitions) == 0
