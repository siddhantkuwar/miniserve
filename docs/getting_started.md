# Getting started

## What is an environment?

MiniServe uses a project-local virtual environment in `.venv/`. A virtual environment keeps this project's Python packages separate from the rest of the computer. `uv` creates and updates that environment from `pyproject.toml` and records exact versions in `uv.lock`.

## First setup

From the repository root:

```bash
uv sync
uv run python tools/hardware_report.py
uv run pytest
```

`uv run` means “run this command using MiniServe's environment.” You do not need to manually activate `.venv` for these commands.

## What the hardware report means

The first report records the machine, Python architecture, storage, and a rough host-memory copy rate. The host copy rate is only a baseline. Later, after MLX is installed, we will add explicit MLX CPU/device measurements and record memory pressure and swap during model runs.

The report is saved under `results/`, which is intentionally ignored by Git because it is specific to the machine that generated it.
