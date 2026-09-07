"""Small deterministic world shared by the runnable remote examples."""

import numpy as np

from unienv_interface.backends import NumpyComputeBackend as Backend
from unienv_interface.space import BoxSpace
from unienv_interface.world import World, WorldNode, WorldEnv


class CounterWorld(World):
    backend = Backend
    device = None
    batch_size = None
    world_timestep = 0.01

    def __init__(self):
        self.value = np.zeros(2, np.float32)
        self.action = np.zeros(2, np.float32)

    def reset(self, *, seed=None, mask=None, **kwargs):
        self.value[:] = 0
        self.action[:] = 0

    def step(self):
        self.value += self.action
        return self.world_timestep


class CounterNode(WorldNode):
    name = "counter"
    control_timestep = 0.02
    update_timestep = 0.01
    has_reward = True

    def __init__(self, world):
        self.world = world
        self.observation_space = BoxSpace(Backend, -np.inf, np.inf, np.float32, shape=(2,))
        self.action_space = BoxSpace(Backend, -1, 1, np.float32, shape=(2,))

    def set_next_action(self, action):
        self.world.action[:] = action

    def get_observation(self):
        return self.world.value

    def get_reward(self):
        return float(self.world.value.sum())


def make_env():
    world = CounterWorld()
    return WorldEnv(world, CounterNode(world))
