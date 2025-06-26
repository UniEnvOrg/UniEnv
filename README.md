# UniEnvPy

TLDR: Gymnasium Library replacement with support for multiple tensor backends

Provides an universal interface for single / parallel state-based or function-based environments. Also contains a set of utilities (such as replay buffers, wrappers, etc.) to facilitate the training of reinforcement learning agents.

## Support for multiple tensor backends

Current backends:

- numpy
- pytorch
- jax

Also supports on-the-fly conversion between backend data (using DLPack, so that the data is not copied).

## Support for multiple simulation environments

Current environments:
- Any Environment defined in Gymnasium interface
- <s>Mujoco</s> (New code will be added in the future, but I'm currently working on refractoring World based environments)
- MJX based on [Mujoco-Playground](https://github.com/google-deepmind/mujoco_playground)
- [ManiSkill 3](https://github.com/haosulab/ManiSkill/)
