from utils import construct_env_from_name, perform_env_test
import pytest

@pytest.mark.parametrize("env_name", ["CartpoleBalance"])
@pytest.mark.parametrize("n_envs", [16, 32])
@pytest.mark.parametrize("seed", [0, 1024])
@pytest.mark.parametrize("jit", [True]) # If not traced it will be tooooooo slow
def test_dmsuite(
    env_name : str,
    n_envs : int,
    seed : int,
    jit : bool,
    episodes : int = 5,
    max_steps : int = 200
):
    env = construct_env_from_name(env_name, n_envs, jit, seed=seed)
    perform_env_test(env, episodes, max_steps)
    env.close()