# Development

This page collects the maintenance workflow for `UniEnvPy`.

## Local Setup

From the `UniEnvPy` directory:

```bash
pip install -e .[dev,gymnasium,video]
pytest
```

Install backend- or integration-specific dependencies separately if you are working on those areas.

## Documentation Generation

The docs site combines:

- hand-written pages under `docs/`
- generated API pages created by `docs/gen_ref_pages.py`

The generation step walks the `unienv_interface` and `unienv_data` modules and emits reference pages consumed by `mkdocstrings`.

To serve the docs locally:

```bash
mkdocs serve
```

## What To Keep Stable

For release-facing docs, the most important guarantees to keep visible are:

- the distinction between `unienv_interface` and `unienv_data`
- the difference between `Env` and `FuncEnv`
- the role of `Space` as the schema contract across runtime and storage code
- the storage-backed nature of replay buffers
- the world/node composition model for richer environments

## Before A Release

Check at least the following:

1. `README.md` reflects the current package layout and installation story.
2. The docs landing page matches the actual module boundaries.
3. `mkdocs build` succeeds without broken references.
4. Tests covering wrappers, spaces, transformations, and replay buffers still pass.
