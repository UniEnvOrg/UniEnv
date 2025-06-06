import pytest
import typing
from unienv_maniskill import FromManiSkillEnv
from unienv_interface.space.space_utils.batch_utils import *

import os.path
import gymnasium as gym
import numpy as np
import torch
import copy

# Env Imports
import mani_skill.envs.tasks.tabletop.pick_cube
import mani_skill.envs.tasks.control.cartpole

TEST_ENV_NAMES = [
    "PickCube-v1",
    "MS-CartpoleBalance-v1"
]
TEST_EPISODES = 2
TEST_MAX_STEPS = 100
TEST_GPU_SIM = True if torch.cuda.is_available() else False

@pytest.mark.parametrize("env_name", TEST_ENV_NAMES)
@pytest.mark.parametrize("obs_mode", ["state", "state_dict", "sensor_data", "rgb", "depth", "segmentation", "rgbd", "rgb+depth", "rgb+depth+segmentation", "rgb+segmentation", "depth+segmentation", "pointcloud"])
@pytest.mark.parametrize("render_mode", ["rgb_array"])
@pytest.mark.parametrize(
    ("sim_backend", "num_envs"), [
        ("cpu", 1)
    ] if not TEST_GPU_SIM else [
        ("gpu", 10),
        ("auto", 5)
    ]
)
def test_maniskill_env(
    env_name : str,
    obs_mode : str,
    render_mode : str,
    num_envs : int,
    sim_backend : str,
    render_backend : str = "gpu" if TEST_GPU_SIM else "cpu",
    max_steps : int = TEST_MAX_STEPS,
    episodes : int = TEST_EPISODES,
):
    env = gym.make(
        env_name, 
        obs_mode=obs_mode,
        render_mode=render_mode,
        sim_backend=sim_backend,
        render_backend=render_backend,
        num_envs=num_envs,
        max_episode_steps=max_steps
    )
    env = FromManiSkillEnv(env)
    done = None
    
    for _ in range(episodes):
        if done is None:
            _, obs, info = env.reset()
        else:
            _, part_obs, info = env.reset(mask=done)
            obs = env.update_observation_post_reset(
                obs, part_obs, done
            )
        
        done = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

        assert obs in env.observation_space
        for _ in range(max_steps):
            action = env.sample_action()
            assert action in env.action_space
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs in env.observation_space
            done = torch.logical_or(terminated, truncated)
            if torch.any(done):
                break