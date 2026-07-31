# MiniServe

MiniServe is a small, from-scratch language-model inference runtime for Apple
Silicon. It is intentionally focused on the systems path between a tokenized
request and a streamed response: model execution, KV-cache ownership, request
scheduling, batching, memory allocation, measurement, and one custom Metal
attention backend.

The target workload is a pinned 4-bit Qwen2.5 0.5B model on an M2 Pro MacBook
Pro with bounded context and concurrency. `mlx-lm` may load architecture
definitions and weights, but MiniServe owns generation, request state, cache
state, scheduling, measurements, and optimized execution.

## Current push

The project is organized as a six-week engineering push:

1. Run deterministic real-model generation and integrate a correct KV cache.
2. Serve concurrent streaming requests with static and continuous batching.
3. Own memory through a logical block allocator and compare 4-bit/8-bit paths.
4. Build a concurrent-load benchmark and a bounded toy MoE experiment.
5. Integrate a custom Metal decode-attention backend into the serving path.
6. Publish reproducible measurements, architecture, limitations, and a demo.

GitHub issues are the public engineering plan. Private prerequisites, learning
notes, and architecture work remain under `docs/`, which is deliberately
ignored by Git.

## Active foundation work

Four existing learning exercises remain active because they directly unblock
the runtime:

- Assignment 5: understand MLX lazy execution and honest timing.
- Assignment 6: load the pinned model behind a narrow adapter.
- Assignment 7: own deterministic uncached greedy generation.
- Assignment 9: prove exact golden-token parity against an isolated oracle.

The earlier attention, transformer-block, tokenizer, and toy-decoder work stays
as completed foundation code. Later assignment scaffolds were removed; their
requirements now live in product-oriented GitHub issues.

## Setup

The environment is managed with `uv` and targets native ARM Python 3.12.

```bash
uv sync
uv run python tools/hardware_report.py
uv run pytest
```

The hardware command writes a machine-readable report to `results/hardware.json` and prints the same report as JSON. It does not download model weights.

## Repository layout

```text
src/miniserve/engine/       Generation, decode, batching, and scheduling
src/miniserve/models/       Model-specific adapter boundary
src/miniserve/cache/        Contiguous and logical block-managed KV storage
src/miniserve/serving/      Request lifecycle and model-owning executor
src/miniserve/api/          Local streaming HTTP boundary
src/miniserve/quantization/ Reference and MLX-native quantization paths
src/miniserve/kernels/      Custom Metal attention backend
tests/                      Focused unit and end-to-end correctness tests
benchmarks/                 Reproducible latency and throughput entry points
tools/                      Development, oracle, and hardware utilities
docs/                       Local-only learning and architecture notes
results/                    Ignored raw measurements
```

Directories are added when their owning issue begins; the repository does not
carry empty implementation scaffolds for future work.

## Runtime boundaries

- One process loads one model and owns accelerator execution.
- Waiting requests do not own KV-cache capacity.
- Every admitted request has explicit lifecycle and cache ownership.
- Benchmarks force MLX evaluation and preserve raw samples.
- The logical block allocator manages MLX arrays, not macOS physical pages.
- The MLX and Metal attention backends obey one numerical contract.
- `mlx_lm.generate()` and `mlx_lm.server()` are oracles, never runtime paths.

## Initial safety limits

- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- Context: 1,024 tokens
- Generation: 128 tokens
- Active requests: 4
- Waiting requests: 16

These are conservative defaults for the target 16 GB machine. Any increase
must be justified by measured memory pressure and cache accounting.
