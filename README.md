<p align="center">
  <img src="assets/logo.png" alt="PDEAgent-Bench logo" width="600" />
</p>

<div align="center">

# PDEAgent-Bench

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://zeroeclipse00.github.io/pde-agent-bench-github-pages/)&#160;
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-gray)](https://huggingface.co/datasets/eclipse00/PDEAgent-Bench)&#160;
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![DOLFINx](https://img.shields.io/badge/DOLFINx-0.10.0-orange.svg)]()
[![Firedrake](https://img.shields.io/badge/Firedrake-2025-green.svg)]()
[![deal.II](https://img.shields.io/badge/deal.II-9.x-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A Multi-Metric, Multi-Library Benchmark for PDE Solver Generation**

[Quick Start](#-quick-start) | [How It Works](#️-how-it-works) | [PDE Types](#-pde-types-covered) | [Leaderboard](#-leaderboard) | [Dataset](#-datasets) | [Citation](#-citation)

</div>

---

PDEAgent-Bench is a comprehensive benchmark that evaluates whether AI agents can **work like computational scientists** — given a PDE problem described in natural language, generate complete, correct, and efficient finite element solver code. It covers three major FEM frameworks and 11 PDE types, measuring both solution accuracy (relative L2 error) and runtime efficiency against oracle reference solutions.

## Overview

### ✨ Highlights

<table>
<tr>
<td align="center" width="25%">🔢<br/><b>Multi-Library Support</b><br/><sub>DOLFINx (Python), Firedrake (Python), deal.II (C++)</sub></td>
<td align="center" width="25%">📐<br/><b>11 PDE Types</b><br/><sub>From Poisson to Navier-Stokes, covering classical and multi-physics</sub></td>
<td align="center" width="25%">📊<br/><b>Two-Metric Evaluation</b><br/><sub>Accuracy (rel. L2 error) + efficiency (runtime) vs oracle</sub></td>
<td align="center" width="25%">🤖<br/><b>Multi-Agent & Multi-LLM</b><br/><sub>OpenAI, Anthropic, Google, Qwen and code agent frameworks</sub></td>
</tr>
<tr>
<td align="center">📦<br/><b>Two Dataset Versions</b><br/><sub>v1: 241 cases (DOLFINx) · v2: 645 cases (all libraries)</sub></td>
<td align="center">🐳<br/><b>Docker-Ready</b><br/><sub>Firedrake and deal.II run in containers, zero local setup</sub></td>
<td align="center">🔁<br/><b>Multi-Attempt Mode</b><br/><sub>Agents can self-correct with error feedback across attempts</sub></td>
<td align="center">⚡<br/><b>Single Entry Point</b><br/><sub>One script to run, compare, and reproduce all experiments</sub></td>
</tr>
</table>

## Understanding the Benchmark

### ⚙️ How It Works

PDEAgent-Bench follows a two-stage pipeline:

```
┌─────────────────────────────────┐    ┌──────────────────────────────────┐
│  Stage 1 — Code Generation      │    │  Stage 2 — Evaluation            │
│                                 │    │                                  │
│  Natural language PDE problem   │    │  Generated solver                │
│      +  API reference guide     │───▶│      vs Oracle reference solution│
│                                 │    │                                  │
│  AI Agent / LLM                 │    │  Metrics: rel. L2 error + runtime│
└─────────────────────────────────┘    └──────────────────────────────────┘
```

**Stage 1 — Code Generation:**  The agent receives a PDE problem description in natural language together with an injected API reference guide for the target solver library (DOLFINx, Firedrake, or deal.II). It must produce complete, runnable solver code — no scaffolding, no partial templates.

**Stage 2 — Evaluation:**  The generated solver is executed in a sandboxed environment. Its numerical output is compared against an oracle reference solution using two metrics:
- **Accuracy**: relative L2 error on a uniform grid (tolerance configurable per case)
- **Efficiency**: wall-clock runtime relative to the oracle (tolerance configurable per case)

Both metrics must pass their thresholds for a case to count as solved.

### 🧮 PDE Types Covered

PDEAgent-Bench v2 covers 11 PDE categories spanning classical and multi-physics problems:

| PDE Type | Description | Libraries |
|:---|:---|:---:|
| **Poisson** | Elliptic boundary value problem | DOLFINx · Firedrake · deal.II |
| **Heat** | Parabolic time-dependent diffusion | DOLFINx · Firedrake · deal.II |
| **Wave** | Hyperbolic second-order wave equation | DOLFINx · Firedrake · deal.II |
| **Linear Elasticity** | Structural deformation | DOLFINx · Firedrake · deal.II |
| **Stokes** | Incompressible slow-flow | DOLFINx · Firedrake · deal.II |
| **Navier-Stokes** | Incompressible viscous flow | DOLFINx · Firedrake |
| **Advection-Diffusion** | Transport with diffusion | DOLFINx · Firedrake · deal.II |
| **Biharmonic** | Fourth-order plate problems | DOLFINx · Firedrake |
| **Hyperelasticity** | Nonlinear elasticity | DOLFINx · Firedrake |
| **Cahn-Hilliard** | Phase-field separation | DOLFINx · Firedrake |
| **Mixed Poisson** | Mixed finite element formulation | DOLFINx · Firedrake · deal.II |

### 🏆 Leaderboard

View the full, up-to-date leaderboard on our **[project website](https://zeroeclipse00.github.io/pde-agent-bench-github-pages/)**.

The leaderboard reports per-library pass rates and aggregated scores across all 11 PDE types, broken down by:
- **Model** (GPT-4o, Claude Sonnet/Opus, Gemini, Qwen, …)
- **Library** (DOLFINx · Firedrake · deal.II)
- **Dataset version** (v1 · v2)
- **Attempt mode** (single-attempt · multi-attempt)

---

## 🚀 Quick Start

### 1. Install the Environment

```bash
# Create a conda environment and install DOLFINx
conda create -n pdebench python=3.11
conda activate pdebench
conda install -c conda-forge fenics-dolfinx=0.10.0 mpich petsc4py

# Install PDEAgent-Bench
pip install -e .
```

> **Firedrake / deal.II**: These two libraries run through Docker by default and do not require local installation.  
> Add `--solver-library firedrake` or `--solver-library dealii` at runtime to invoke the corresponding image automatically.

### 2. Configure API Keys

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="..."

# Qwen
export DASHSCOPE_API_KEY="..."
```

### 3. Run Evaluations

```bash
# Evaluate a single LLM using the v2 dataset and the DOLFINx backend
python scripts/run_benchmark.py --agent gpt-4o

# Evaluate and compare multiple LLMs
python scripts/run_benchmark.py --agent gpt-4o sonnet-3.5 gemini

# Use the Firedrake library (Docker invoked automatically)
python scripts/run_benchmark.py --agent gpt-4o --solver-library firedrake

# Use the deal.II library (agent generates C++, Docker invoked automatically)
python scripts/run_benchmark.py --agent gpt-4o --solver-library dealii

# Test only the v1 dataset with 241 DOLFINx cases
python scripts/run_benchmark.py --agent gpt-4o --version v1
```

---

## Evaluation Entry Point

All evaluations run through the single entry point `scripts/run_benchmark.py`:

```
usage: run_benchmark.py [-h] --agent AGENT [AGENT ...] [--output OUTPUT]
                        [--version {v1,v2}] [--cases CASE_ID ...]
                        [--equation-types TYPE ...] [--skip-generation]
                        [--solver-path PATH] [--eval-existing-dir DIR]
                        [--timeout SECONDS] [--max-attempts N]
                        [--solver-library {dolfinx,firedrake,dealii}]
```

### Common Use Cases

```bash
# Test only specific cases
python scripts/run_benchmark.py --agent gpt-4o --cases poisson_basic heat_basic

# Test only specific equation types
python scripts/run_benchmark.py --agent gpt-4o --equation-types poisson heat

# Multi-attempt mode: retry with the previous error message as feedback
python scripts/run_benchmark.py --agent gpt-4o --max-attempts 3

# Skip LLM calls and directly evaluate existing solvers
python scripts/run_benchmark.py --agent gpt-4o --skip-generation

# Evaluate a specific solver file
python scripts/run_benchmark.py --agent gpt-4o \
    --solver-path results/gpt-4o/poisson_basic/solver.py \
    --cases poisson_basic

# Batch-evaluate all solvers in an existing directory without re-calling the LLM
python scripts/run_benchmark.py --agent qwen3-max \
    --eval-existing-dir results/qwen3-max
```

### Arguments

| Argument | Default | Description |
|:---|:---|:---|
| `--agent` | Required | LLM name; multiple agents supported. See the list below. |
| `--version` | `v2` | Dataset version: `v1` (241 cases) or `v2` (645 cases). |
| `--solver-library` | `dolfinx` | Solver library: `dolfinx` \| `firedrake` \| `dealii`. |
| `--cases` | All | Case IDs to evaluate, separated by spaces. |
| `--equation-types` | All | Equation types to evaluate, e.g. `poisson heat`. |
| `--max-attempts` | `1` | Maximum attempts in multi-attempt mode. |
| `--timeout` | `300` | Per-case execution timeout in seconds. |
| `--output` | `results/` | Output directory for results. |
| `--eval-existing-dir` | None | Batch-evaluate an existing solver directory. |
| `--skip-generation` | No | Skip LLM calls and reuse existing solvers. |

---

## Supported LLMs and Agents

### LLMs (Language Models Only)

| Name | Provider |
|:---|:---|
| `gpt-4o`, `gpt-4o-mini` | OpenAI |
| `gpt-5.1`, `gpt-5.2`, `gpt-5.4` | OpenAI |
| `o3-mini` | OpenAI |
| `sonnet-3.5`, `sonnet-3.6` | Anthropic |
| `claude-opus-4.7`, `claude-opus-4.6` | Anthropic |
| `haiku` | Anthropic |
| `gemini`, `gemini-3.0-pro`, `gemini-3.0-flash`, `gemini-3.1-pro` | Google |
| `qwen3-max`, `qwen3.6-plus` | Qwen |

### Code Agents (External Agent Frameworks)

| Name | Description | Config File |
|:---|:---|:---|
| `codepde` | CodePDE Agent | `pdebench/configs/codepde.json` |
| `openhands` | OpenHands Agent | `pdebench/configs/openhands.json` |
| `mini-swe-agent` | MiniSWE-Agent | `pdebench/configs/mini-swe-agent.json` |

Code agents use the same prompts as LLMs. Environment variables and timeouts are configured through `pdebench/configs/{agent}.json`.

---

## 📦 Datasets

The benchmark dataset is available on [Hugging Face](https://huggingface.co/datasets/eclipse00/PDEAgent-Bench).

### Dataset Format (v2, 645 Cases)

`data/benchmark_v2.jsonl` — one JSON object per line, covers all three solver libraries:

```json
{
  "id": "poisson_basic",
  "oracle_config": {
    "pde": { "type": "poisson", "..." },
    "domain": { "type": "unit_square" },
    "mesh": { "resolution": 120 },
    "bc": { "..." },
    "output": { "format": "npz", "grid": { "nx": 50, "ny": 50 } }
  },
  "evaluation_config": {
    "target_metric": "rel_L2_grid",
    "timeout_sec": 300,
    "accuracy_tolerance": 10,
    "time_tolerance": 3
  },
  "supported_libraries": ["dolfinx", "firedrake", "dealii"]
}
```

---

## Project Structure

```
pdebench/
├── data/
│   ├── benchmark_v1.jsonl      # v1 dataset (241 cases, DOLFINx)
│   └── benchmark_v2.jsonl      # v2 dataset (645 cases, multi-backend)
│
├── scripts/
│   └── run_benchmark.py        # Single evaluation entry point
│
├── pdebench/                   # Python package
│   ├── core/
│   │   ├── prompt_builder.py   # Prompt generation with API guide injection
│   │   ├── llm_client.py       # LLM calls for OpenAI/Anthropic/Google/Qwen
│   │   └── feedback_prompt.py  # Multi-attempt feedback prompt
│   │
│   ├── oracle/                 # Oracle reference solution system
│   │   ├── oracle.py           # Unified entry point dispatching by PDE type
│   │   ├── {pde_type}.py       # DOLFINx implementation for each PDE
│   │   ├── firedrake_oracle/   # Firedrake library implementation
│   │   ├── dealii_oracle/      # deal.II library with .cc programs
│   │   └── docker_bridge.py    # Bridge for running inside Docker containers
│   │
│   ├── sandbox/
│   │   ├── executor.py         # Isolated execution for Python solvers
│   │   └── cpp_executor.py     # C++ solver compilation and execution for deal.II
│   │
│   ├── agents/                 # Code agent wrappers
│   │   ├── codepde_wrapper.py
│   │   ├── openhands_wrapper.py
│   │   └── mini_swe_agent_wrapper.py
│   │
│   ├── metrics/
│   │   └── specialized/        # PDE-specific metric computation for 11 classes
│   │
│   ├── analysis/
│   │   ├── gate_analyzer.py    # Pass-rate gate analysis
│   │   └── error_classifier.py # Error classification
│   │
│   ├── configs/                # Code agent configuration
│   │   ├── codepde.json
│   │   ├── openhands.json
│   │   └── mini-swe-agent.json
│   │
│   └── docs/                   # API reference guides injected into prompts
│       ├── DOLFINX_GUIDE.md
│       ├── FIREDRAKE_GUIDE.md
│       └── DEALII_GUIDE.md
│
├── docker/                     # Docker image definitions
├── experiments/                # Experiment run scripts
│   ├── minisweagent.sh
│   └── openhands.sh
│
├── results/                    # Evaluation result output directory
└── tests/                      # Unit tests
```

---

## 🐳 Docker Support

The Firedrake and deal.II libraries run in Docker containers by default:

```bash
# Pull images
docker pull pdebench/firedrake:latest
docker pull pdebench/dealii:latest

# Evaluations invoke Docker automatically
python scripts/run_benchmark.py --agent gpt-4o --solver-library firedrake
python scripts/run_benchmark.py --agent gpt-4o --solver-library dealii
```

---

## 🔧 Developer Guide

### Add a New LLM

Add an entry to the `SUPPORTED_AGENTS` dictionary in `pdebench/core/llm_client.py`:

```python
'my-model': {'provider': 'openai', 'model': 'my-model-id'},
```

### Add a New Code Agent

1. Create `my_agent_wrapper.py` under `pdebench/agents/` and inherit from `BaseAgent`.
2. Implement `generate_solution(prompt, context)` and return an `AgentResponse`.
3. Register it in `pdebench/agents/__init__.py`: `AgentRegistry.register('my-agent', MyAgentWrapper)`.
4. Add its configuration in `pdebench/configs/my-agent.json`.

### Add a New PDE Type

1. Add `my_pde.py` under `pdebench/oracle/` with a DOLFINx implementation.
2. Import it in `oracle/oracle.py` and add a `pde_type` branch.
3. To support Firedrake or deal.II, add implementations in the corresponding subdirectories.

---

## 📜 Citation

If you use PDEAgent-Bench in your research, please cite our paper:

```bibtex
@misc{hang2026pdeagentbench,
  title  = {PDEAgent-Bench: A Multi-Metric, Multi-Library Benchmark for PDE Solver Generation},
  author = {Zhen Hang, Yushan Yashengjiang, Junhui Li, Huanshuo Dong,
            Yang Wei, Zhezheng Hao, Jiangtao Ma, Songlin Bai,
            Zhongkai Hao, Xihang Yue, Gangzong Si, Dongming Jiang,
            Chao Yao, Zhanhua Hu, Jianqing Zhang, Pengwei Liu,
            Yaomin Shen, Xingyu Ren, Lei Liu, Zikang Xu, Han Li,
            Qingsong Yao, Hande Dong, Hong Wang},
  year   = {2026},
  note   = {Under review at NeurIPS 2026},
  url    = {https://github.com/YusanX/pde-agent-bench}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

- The [FEniCSx / DOLFINx](https://fenicsproject.org/) team
- The [Firedrake](https://www.firedrakeproject.org/) team
- The [deal.II](https://www.dealii.org/) team
- [SWE-bench](https://www.swebench.com/) for inspiration in evaluation design

<p align="right"><a href="#top">Back to top</a></p>
