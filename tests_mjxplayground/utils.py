from mujoco_playground import registry, MjxEnv
from unienv_mjxplayground import FromMJXPlaygroundEnv
from unienv_interface.env_base import FuncEnvBasedEnv, Env
from unienv_interface.backends.jax import JaxComputeBackend
import jax

def construct_env_from_name(
    name : str,
    num_envs : int,
    jit = True,
    seed : int = 0
) -> FuncEnvBasedEnv:
    mjxenv = registry.load(name)
    funcenv = FromMJXPlaygroundEnv(
        mjxenv,
        num_envs,
        None,
        jit=jit
    )
    env = FuncEnvBasedEnv(
        funcenv,
        rng=JaxComputeBackend.random.random_number_generator(seed)
    )
    return env

def perform_env_test(
    env : Env,
    episodes : int,
    max_steps : int
):
    done = None
    for _ in range(episodes):
        if done is None:
            _, obs, info = env.reset()
        else:
            _, part_obs, info = env.reset(mask=done)
            obs = env.update_observation_post_reset(
                obs, part_obs, done
            )
        
        assert obs in env.observation_space
        for _ in range(max_steps):
            action = env.sample_action()
            assert action in env.action_space
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs in env.observation_space
            done = env.backend.logical_or(terminated, truncated)
            if env.backend.any(done):
                break