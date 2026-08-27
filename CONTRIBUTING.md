# Contributing guide

This document describes how to contribute to `prism-1` in a reliable and reproducible way.

## 1) Before you start

- Use Python `>=3.12,<3.13`.
- Install dependencies with `poetry install`.
- Create a dedicated branch per topic (e.g. `feat/xxx`, `fix/yyy`, `docs/zzz`).
- Never version secrets (`.env`, AWS keys, tokens).
- Do not version patient extracts, real identifier lists, or internal lab notebooks.
- Read `CODE_OF_CONDUCT.md` and `SECURITY.md`.

## 2) Working conventions

- Keep commits small, coherent, and descriptive.
- Prefer one atomic change per commit (code plus related documentation).
- Keep the reference configuration in `configs/article_stats.yaml` and document every new parameter.

## 3) Code quality

Install hooks (once):

```bash
poetry run pre-commit install
```

Before pushing:

```bash
poetry run pre-commit run --all-files
```

## 4) Issues

To open a useful issue:

- describe the observed behaviour and the expected behaviour;
- include context (config file, single/batch mode, Python version);
- attach useful traces (error, log excerpt, produced artefacts).

## 5) Pull requests

Each pull request should include:

- a clear goal (why the change is needed);
- the list of impacted files;
- a verification plan that was run locally;
- potential impacts on analytical results.

Recommended checklist:

- [ ] code is readable and documented (docstrings for exposed APIs/functions)
- [ ] documentation is updated (`README.md`, config)
- [ ] no secrets and no unnecessary large files

## 6) Documentation

Any change to:

- the pipeline structure,
- matching methods,
- output formats,

must be reflected in `README.md` and `configs/article_stats.yaml`.
