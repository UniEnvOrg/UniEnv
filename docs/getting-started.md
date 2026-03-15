# Getting Started

## Install

Install the base package:

```bash
pip install unienv
```

Install optional extras when you need them:

```bash
pip install "unienv[gymnasium,video]"
```

## Local Development

For local development inside the repository:

```bash
pip install -e .[dev,gymnasium,video]
pytest
```

## Documentation Build

The GitHub Pages workflow builds this site from:

- markdown files under `docs/`
- generated API reference pages for `unienv_interface` and `unienv_data`

To build locally after installing the MkDocs dependencies:

```bash
mkdocs serve
```
