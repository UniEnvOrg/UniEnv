# UniEnvPy
Universal Single / Parallel Environment Interfaces

## Support for multiple tensor backends

Current backends:

- numpy
- pytorch
- jax

Also supports on-the-fly conversion between backend data.

## Support for multiple simulation environments

Current environments:
- Mujoco
- MJX (Env Wrapper Only, based on [Mujoco-Playground](https://github.com/google-deepmind/mujoco_playground))
- ManiSkill 3 (Env Wrapper Only)

## Support for gymnasium environments

Comes with gymnasium environment translation out-of-the-box.