# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GEPA (Genetic-Pareto) is a Python framework for optimizing textual system components (prompts, code, agent architectures, configurations) using LLM-based reflection and Pareto-efficient evolutionary search. Published on PyPI as `gepa`.

## Commands

All Python commands must use `uv run` prefix. Uses **uv** for dependency management with setuptools as build backend.

```bash
# Setup
uv sync --extra dev

# Tests
uv run pytest tests/                         # all tests
uv run pytest tests/test_batch_sampler.py -v  # single test file
uv run pytest -vv tests/ -k "test_name"       # single test by name

# Lint & format
uv run ruff check src/                        # lint
uv run ruff check --fix src/                  # lint with auto-fix
uv run ruff format src/                       # format

# Type checking
uv run pyright src/                           # full check
uv run pyright src/gepa/strategies/           # specific module

# Pre-commit (runs ruff + yaml/toml checks)
uv run pre-commit run --all-files

# Build
uv run python -m build
```

## Code Style

- Google Python Style Guide with **ruff** (line length 120, double quotes, 4-space indent)
- Type checking via **pyright** (strict)
- No relative imports (enforced by ruff `ban-relative-imports = "all"`)
- Python 3.10+ target (must work on 3.10-3.14)
- Pre-commit hooks auto-run on commit

## Architecture

### Core Optimization Loop (`src/gepa/core/engine.py` - `GEPAEngine`)

1. **Select** - Pick candidate from Pareto frontier using a `CandidateSelector` strategy
2. **Execute** - Run candidate on minibatch via `GEPAAdapter.evaluate()`, capture traces
3. **Reflect** - LLM reads execution traces via `GEPAAdapter.make_reflective_dataset()` to diagnose failures
4. **Mutate** - `ReflectiveMutationProposer` generates improved candidate text
5. **Accept** - Update Pareto frontier in `GEPAState` if candidate improves

### Key Types

- **Candidate**: `dict[str, str]` mapping component names to component text
- **GEPAAdapter** (`core/adapter.py`): Protocol with 3 responsibilities - `evaluate()`, `make_reflective_dataset()`, `propose_new_texts()` (optional)
- **EvaluationBatch**: Container for outputs, scores, trajectories, and optional objective_scores
- **GEPAState** (`core/state.py`): Tracks Pareto frontier, evaluation cache, and candidate pool
- **GEPAResult** (`core/result.py`): Final output with best candidate and metrics

### Module Layout (`src/gepa/`)

- `api.py` - `gepa.optimize()` main entry point; wires together all components
- `optimize_anything.py` - `gepa.optimize_anything.optimize_anything()` universal API for any text artifact
- `core/` - Engine, adapter protocol, state management, callbacks, data loading
- `proposer/` - `reflective_mutation/` (LLM-based reflect+mutate) and `merge.py` (Pareto-aware merging)
- `adapters/` - Built-in adapters: `default_adapter/` (single-turn LLM), `dspy_adapter/`, `dspy_full_program_adapter/`, `generic_rag_adapter/`, `mcp_adapter/`, `terminal_bench_adapter/`, `anymaths_adapter/`, `optimize_anything_adapter/`
- `strategies/` - Pluggable strategies for batch sampling, candidate selection (pareto/current_best/epsilon_greedy/top_k_pareto), component selection (round_robin/all), evaluation policies
- `logging/` - Experiment tracking (wandb, mlflow) and logger interface
- `utils/` - Stop conditions, code execution, stdio capture
- `gskill/` - Skill discovery module (separate optional dep)

### Integration Points

GEPA is used as the optimization engine inside:
- **DSPy**: `dspy.GEPA` optimizer
- **MLflow**: `mlflow.genai.optimize_prompts()`
- **Comet ML Opik**: Agent Optimizer

### Dependencies

Zero core dependencies by default. Features gated behind optional extras (`full`, `dspy`, `dev`, `gskill`). The `dspy` extra is intentionally empty (signals DSPy integration support without pulling deps).

### CI

GitHub Actions runs on every push/PR: ruff lint, pyright type check, pytest on Python 3.10-3.14, and package build validation.
