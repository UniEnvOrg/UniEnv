# UniEnvPy

TLDR: Gymnasium Library replacement with support for multiple tensor backends

Provides an universal interface for single / parallel state-based or function-based environments. Also contains a set of utilities (such as replay buffers, wrappers, etc.) to facilitate the training of reinforcement learning agents.

## Cross-backend Support

UniEnvPy supports multiple tensor backends with zero-copy translation layers through the DLPack protocol, and allows you to use the same abstract compute backend interface to write custom data transformation layers, environment wrappers and other utilities. This is powered by the [xbarray](https://github.com/realquantumcookie/xbarray) package, which builts on top of the Array API Standard, and supports the following backends:

- numpy
- pytorch
- jax

We also support diverse simulation environments and real robots, built on top of the abstract environment / world interface. This allows you to reuse code across different sim and real robots.

Current supported simulation environments:
- Any Environment defined in Gymnasium interface
- <s>Mujoco</s> (New code will be added in the future, but I'm currently working on refractoring World based environments)
- MJX based on [Mujoco-Playground](https://github.com/google-deepmind/mujoco_playground)
- [ManiSkill 3](https://github.com/haosulab/ManiSkill/)

Current supported real robots:
- Franka Research 3 + RobotiQ Gripper in Droid Setup
- OyMotion OHand

## Installation

Install the package with pip

```bash
pip install unienv
```

You can install optional dependencies such as `gymnasium`, `mjx`, `maniskill`, `video` by running

```bash
pip install unienv[gymnasium,mjx,maniskill,video]
```

## Acknowledgements

The idea of this project is inspired by [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and its predecessor [OpenAI Gym](https://github.com/openai/gym). 
This library is impossible without the great work of DataAPIs Consortium and their work on the [Array API Standard](https://data-apis.org/array-api/latest/). The zero-copy translation layers are powered by the [DLPack](https://github.com/dmlc/dlpack) project.