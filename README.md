# UniEnv

Framework unifying robot environments and data APIs. UniEnv provides an universal interface for robot actors, sensors, environments, and data. 

## Tensor library cross-backend Support

UniEnv supports multiple tensor backends with zero-copy translation layers through the DLPack protocol, and allows you to use the same abstract compute backend interface to write custom data transformation layers, environment wrappers and other utilities. This is powered by the [XBArray](https://github.com/UniEnvOrg/XBArray) package.

## Universal Robot Environment Interface

UniEnv supports diverse simulation environments and real robots, built on top of the abstract environment / world interface. This allows you to reuse code across different sim and real robots.

## Universal Robot Data Interface

UniEnv provides a universal data interface for accessing robot data through the abstract `BatchBase` interface. We also provide a utility `ReplayBuffer` for saving data from various environments with diverse data format support, including `hdf5`, memory-mapped torch tensors, and others.

## Installation

Install the package with pip

```bash
pip install unienv
```

You can install optional dependencies such as `gymnasium` (for Gymnasium-compatible environments), `dev`, or `video` by running

```bash
pip install unienv[gymnasium,video]
```

## Cite

If you use UniEnv in your research, please cite it as follows:

```bibtex
@software{cao_unienv,
  author = {Cao, Yunhao AND Fang, Kuan},
  title = {{UniEnv: Unifying Robot Environments and Data APIs}},
  year = {2025},
  month = oct,
  url = {https://github.com/UniEnvOrg/UniEnv},
  license = {MIT}
}
```

## Acknowledgements

The idea of this project is inspired by [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and its predecessor [OpenAI Gym](https://github.com/openai/gym). 
This library is impossible without the great work of DataAPIs Consortium and their work on the [Array API Standard](https://data-apis.org/array-api/latest/). The zero-copy translation layers are powered by the [DLPack](https://github.com/dmlc/dlpack) project.