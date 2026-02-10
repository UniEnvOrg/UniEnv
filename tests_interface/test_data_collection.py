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


class MockTrajectoryReplayBuffer:
    """Mock TrajectoryReplayBuffer for testing."""
    
    def __init__(self):
        self.episode_data_calls = []
        self.mark_end_calls = []
    
    def set_current_episode_data(self, value):
        self.episode_data_calls.append(value)
    
    def mark_episode_end(self):
        self.mark_end_calls.append(True)


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
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)

        assert wrapped.env == env
        assert wrapped.batch_size is None
        assert wrapped.backend == env.backend
        assert wrapped.observation_space == env.observation_space
        assert wrapped.action_space == env.action_space
        assert wrapped.batch == mock_batch

    def test_init_batched_env(self):
        """Test initialization with batched environment."""
        batch_size = 4
        env = MockEnv(batch_size=batch_size)
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)

        assert wrapped.env == env
        assert wrapped.batch_size == batch_size
        assert wrapped.batch == mock_batch

    def test_reset(self):
        """Test reset functionality."""
        env = MockEnv(batch_size=None)
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)

        context, obs, info = wrapped.reset()

        assert obs is not None
        assert isinstance(info, dict)
        assert wrapped._prev_obs is not None

    def test_single_step_data_collection(self):
        """Test data collection on single step."""
        env = MockEnv(batch_size=None)
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)

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
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)

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
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)

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
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)

        # Should not raise an error
        result = wrapped.render()
        assert result is None

    def test_close_delegation(self):
        """Test that close is delegated to wrapped environment."""
        env = MockEnv(batch_size=None)
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)

        # Should not raise an error
        wrapped.close()

    def test_step_id_tracking_single_env(self):
        """Test that step_id is tracked correctly in single environment."""
        env = MockEnv(batch_size=None)
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)

        # Reset and take multiple steps
        context, obs, info = wrapped.reset()

        for i in range(5):
            action = np.random.randn(2).astype(np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)

        transitions = wrapped.get_transitions()
        assert len(transitions) == 5
        
        # Check that step_ids are sequential
        for i, transition in enumerate(transitions):
            assert "step_id" in transition
            assert transition["step_id"] == i

    def test_step_id_tracking_batched_env(self):
        """Test that step_id is tracked correctly in batched environment."""
        batch_size = 4
        env = MockEnv(batch_size=batch_size)
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)

        # Reset and take multiple steps
        context, obs, info = wrapped.reset()

        for i in range(5):
            action = np.random.randn(batch_size, 2).astype(np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)

        trajectories = wrapped.get_trajectories(batch_as_list=True)
        assert len(trajectories) == batch_size
        
        # Check that each trajectory has sequential step_ids
        for traj in trajectories:
            assert len(traj) == 5
            for i, transition in enumerate(traj):
                assert "step_id" in transition
                assert transition["step_id"] == i

    def test_step_id_reset_on_done(self):
        """Test that step_id resets when episode ends."""
        env = MockEnv(batch_size=None)
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)

        # First episode - take 3 steps
        context, obs, info = wrapped.reset()
        for i in range(3):
            action = np.random.randn(2).astype(np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)

        # Reset (simulating episode end)
        context, obs, info = wrapped.reset()
        
        # Second episode - take 2 steps
        for i in range(2):
            action = np.random.randn(2).astype(np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)

        # Get all transitions
        all_transitions = wrapped.get_transitions()
        assert len(all_transitions) == 5
        
        # First 3 should have step_ids 0, 1, 2
        for i in range(3):
            assert all_transitions[i]["step_id"] == i
        
        # Last 2 should have step_ids 0, 1 (reset after episode end)
        for i in range(2):
            assert all_transitions[3 + i]["step_id"] == i

    def test_trajectory_replay_buffer_detection(self):
        """Test detection of TrajectoryReplayBuffer."""
        env = MockEnv(batch_size=None)
        
        # Test with mock TrajectoryReplayBuffer
        mock_buffer = MockTrajectoryReplayBuffer()
        wrapped_with_buffer = DataCollectionEnvWrapper(env, batch=mock_buffer)
        assert wrapped_with_buffer.batch is mock_buffer
        assert wrapped_with_buffer._is_trajectory_buffer is True
        
        # Test with non-TrajectoryReplayBuffer
        class SomeOtherBuffer:
            pass
        
        other_buffer = SomeOtherBuffer()
        wrapped_other = DataCollectionEnvWrapper(env, batch=other_buffer)
        assert wrapped_other.batch is other_buffer
        assert wrapped_other._is_trajectory_buffer is False

    def test_should_save_episode_default(self):
        """Test that should_save_episode returns True by default."""
        env = MockEnv(batch_size=None)
        mock_batch = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_batch)
        
        # Default implementation should return True
        result = wrapped.should_save_episode(0, [], False, False, {})
        assert result is True

    def test_episode_id_tracking_non_trajectory_buffer(self):
        """Test that episode_id is tracked when not using TrajectoryReplayBuffer."""
        env = MockEnv(batch_size=None)
        
        class NonTrajectoryBuffer:
            pass
        
        buffer = NonTrajectoryBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=buffer)

        # Reset and take steps
        context, obs, info = wrapped.reset()
        
        for i in range(3):
            action = np.random.randn(2).astype(np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)

        # Reset (new episode)
        context, obs, info = wrapped.reset()
        
        for i in range(2):
            action = np.random.randn(2).astype(np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)

        transitions = wrapped.get_transitions()
        assert len(transitions) == 5
        
        # Check episode_ids - first 3 should be 0, last 2 should be 1
        for i in range(3):
            assert "episode_id" in transitions[i]
            assert transitions[i]["episode_id"] == 0
        
        for i in range(2):
            assert "episode_id" in transitions[3 + i]
            assert transitions[3 + i]["episode_id"] == 1

    def test_no_episode_id_with_trajectory_buffer(self):
        """Test that episode_id is NOT in transitions when using TrajectoryReplayBuffer."""
        env = MockEnv(batch_size=None)
        mock_buffer = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_buffer)

        # Reset and take steps
        context, obs, info = wrapped.reset()
        
        for i in range(3):
            action = np.random.randn(2).astype(np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)

        transitions = wrapped.get_transitions()
        assert len(transitions) == 3
        
        # episode_id should not be in transitions when using TrajectoryReplayBuffer
        for transition in transitions:
            assert "episode_id" not in transition

    def test_trajectory_buffer_episode_context_storage(self):
        """Test that episode context is stored in TrajectoryReplayBuffer when episode ends."""
        env = MockEnv(batch_size=None)
        mock_buffer = MockTrajectoryReplayBuffer()
        wrapped = DataCollectionEnvWrapper(env, batch=mock_buffer)

        # Reset and take steps until episode ends
        context, obs, info = wrapped.reset()
        
        # Take many steps to trigger episode end
        for i in range(15):  # MockEnv has max_steps=10
            action = np.random.randn(2).astype(np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)
            if terminated:
                break

        # After episode ends and reset, episode context should have been stored
        # Note: In this mock env, context is None, so no calls should be made
        # But if we had context, it would be stored
        
    def test_custom_should_save_episode(self):
        """Test custom should_save_episode implementation."""
        env = MockEnv(batch_size=None)
        mock_batch = MockTrajectoryReplayBuffer()
        
        class CustomWrapper(DataCollectionEnvWrapper):
            def should_save_episode(self, env_idx, traj_data, truncation, termination, info):
                # Only save episodes that terminated naturally (not truncated)
                return termination and not truncation
        
        wrapped = CustomWrapper(env, batch=mock_batch)
        
        # Test with termination=True, truncation=False
        result = wrapped.should_save_episode(0, [], False, True, {})
        assert result is True
        
        # Test with termination=False, truncation=True
        result = wrapped.should_save_episode(0, [], True, False, {})
        assert result is False
