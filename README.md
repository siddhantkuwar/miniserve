# MiniServe

MiniServe is a learning-oriented local inference runtime built from first principles on Apple Silicon. The project starts with a manual MLX decoding loop and progressively adds KV caching, serving, scheduling, batching, quantization, profiling, and cross-architecture kernel work.

The project requirements, learning notes, and architecture documents are intentionally local-only under `docs/` and are not published to GitHub.

## Phase 0 setup

This repository currently contains the Phase 0 foundation. The first milestone is a manual, token-by-token decoder for a small pretrained model. The environment is managed with `uv` and targets native ARM Python 3.12.

```bash
uv sync
uv run python tools/hardware_report.py
uv run pytest
```

The hardware command writes a machine-readable report to `results/hardware.json` and prints the same report as JSON. It does not download model weights.

## Repository layout

```text
src/miniserve/       Original MiniServe Python package
tests/               Unit and integration tests
tools/               Development and hardware utilities
benchmarks/          Reproducible benchmark entry points
docs/                Local-only planning, learning, and architecture notes
results/             Raw local measurements; generated files are ignored
```

## Provenance rule

LeetLLM and `mlx-lm` are learning resources and correctness oracles. MiniServe's generation loop, request state, cache ownership, scheduler, measurements, and kernels must remain original code.
