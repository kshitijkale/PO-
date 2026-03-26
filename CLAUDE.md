# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is GEPA?

GEPA (Genetic-Pareto) is a Python framework for optimizing text components (AI prompts, code, agent architectures, configurations) using LLM-based reflection and Pareto-efficient evolutionary search. It works by selecting candidates from a Pareto frontier, executing them on minibatches, reflecting on full execution traces via an LLM to diagnose failures, and mutating candidates with targeted fixes.

## Setup

Uses **uv** for dependency management with setuptools as the build backend. All Python execution must go through `uv run`.

```bash
uv sync --extra dev
uv run pre-commit install   # one-time hook setup
```

## Build & Test Commands

```bash
uv run pytest                           # run full test suite
uv run pytest tests/test_state.py       # run a single test file
uv run pytest tests/test_state.py -k "test_name"  # run a single test
uv run ruff check src/                  # lint
uv run ruff format src/                 # format
uv run pyright src/                     # type check (standard mode)
uv run pyright src/gepa/strategies/     # type check a specific module
```

## Code Style

- **Ruff**: line length 120, double quotes, space indent, Python 3.10+ target
- **No relative imports** (enforced by ruff `ban-relative-imports = "all"`)
- `isort` via ruff with `gepa` as first-party, `dspy` as third-party
- `__init__.py` files may have unused imports (F401 ignored)
- Tests allow assert statements, relative imports, and unused fixture args
- Pyright in `standard` mode; several adapter directories and `tests/` are excluded from type checking (see `pyrightconfig.json`)
- Pre-commit hooks run ruff-check (with `--fix`), ruff-format, check-yaml, check-toml, check-added-large-files (3MB max), check-merge-conflict, and debug-statements

## Architecture

### Two Public APIs

1. **`gepa.optimize()`** (`src/gepa/api.py`) — Optimizes a dict of named text components against a dataset using the adapter pattern. The primary API for prompt optimization and DSPy integration.

2. **`gepa.optimize_anything.optimize_anything()`** (`src/gepa/optimize_anything.py`) — Universal API that optimizes a single text artifact (code, prompts, configs, SVGs, etc.) against an evaluator function. Supports three modes: single-task search, multi-task search, and generalization (with train/val split).

### Core Loop (`src/gepa/core/`)

- **`engine.py`** — `GEPAEngine` orchestrates the optimization loop: evaluate seed on valset, then iterate (propose via reflective mutation or merge, evaluate, accept/reject based on Pareto improvement).
- **`state.py`** — `GEPAState` holds the Pareto frontier, candidate history, scores, and budget tracking. Supports serialization for checkpointing/resuming via `run_dir`.
- **`adapter.py`** — `GEPAAdapter` protocol defines the integration point: `evaluate()` runs candidates on data and returns `EvaluationBatch` (scores + trajectories); `make_reflective_dataset()` extracts per-component feedback for the reflection LM.
- **`result.py`** — `GEPAResult` wraps the final optimization output (best candidate, scores, Pareto front).
- **`callbacks.py`** — Event-based callback system for observing optimization progress.

### Proposers (`src/gepa/proposer/`)

- **`reflective_mutation/`** — Core proposal strategy. Selects a candidate from the Pareto frontier, runs it on a training minibatch, builds a reflective dataset (execution traces + scores), and asks the reflection LM to propose improved component text.
- **`merge.py`** — Combines strengths of two Pareto-optimal candidates that excel on different task subsets.

### Strategies (`src/gepa/strategies/`)

Pluggable policies for candidate selection (`pareto`, `current_best`, `epsilon_greedy`, `top_k_pareto`), component selection (`round_robin`, `all`), batch sampling (`epoch_shuffled`), evaluation policy, and instruction proposal templates.

### Adapters (`src/gepa/adapters/`)

Each adapter implements `GEPAAdapter` for a specific system type:
- `default_adapter/` — Single-turn LLM task prompt optimization (used when no custom adapter is provided)
- `optimize_anything_adapter/` — Backs the `optimize_anything` API
- `dspy_adapter/`, `dspy_full_program_adapter/` — DSPy integration
- `generic_rag_adapter/` — RAG pipeline optimization
- `mcp_adapter/` — MCP tool description optimization
- `terminal_bench_adapter/`, `anymaths_adapter/` — Domain-specific adapters

### Other Key Modules

- **`lm.py`** — `LM` wrapper around litellm for making LLM calls
- **`optimize_anything.py`** — The `optimize_anything` function and its config dataclasses (`GEPAConfig`, `EngineConfig`)
- **`utils/stop_condition.py`** — Stopper protocols (`MaxMetricCallsStopper`, `FileStopper`, `TimeoutStopCondition`, `NoImprovementStopper`, `SignalStopper`, `CompositeStopper`)
- **`logging/`** — Experiment tracking (WandB, MLflow) and file/stdout logging
- **`gskill/`** — SWE-bench skill learning pipeline (has its own ruff E402 exception for suppressing logging before imports)
