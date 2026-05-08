# PDEAgent-Bench: A Multi-Metric, Multi-Library Benchmark for PDE Solver Generation

**A benchmark system for evaluating the end-to-end PDE solver code generation capabilities of large language models and AI agents.**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![DOLFINx](https://img.shields.io/badge/DOLFINx-0.10.0-orange.svg)]()
[![Firedrake](https://img.shields.io/badge/Firedrake-2025-green.svg)]()
[![deal.II](https://img.shields.io/badge/deal.II-9.x-red.svg)]()

## Project Overview

PDEBench evaluates whether AI agents can "work like computational scientists":

- **From physics to code**: Given a PDE problem described in natural language, the LLM/agent must generate complete FEM solver code.
- **Multi-backend support**: Supports three solver frameworks: DOLFINx (Python), Firedrake (Python), and deal.II (C++).
- **Two datasets**: v1 (241 cases, DOLFINx only) and v2 (645 cases, all backends), covering 11 PDE types.
- **Two-dimensional evaluation**: Measures accuracy with relative L2 error and efficiency with runtime, compared against oracle reference solutions while tolerating mesh differences.

## Quick Start

### 1. Install the Environment

```bash
# Create a conda environment and install DOLFINx
conda create -n pdebench python=3.11
conda activate pdebench
conda install -c conda-forge fenics-dolfinx=0.10.0 mpich petsc4py

# Install PDEBench
pip install -e .
```

> **Firedrake / deal.II**: These two backends run through Docker by default and do not require local installation.  
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

# Use the Firedrake backend with Docker automatically
python scripts/run_benchmark.py --agent gpt-4o --solver-library firedrake

# Use the deal.II backend, where the agent generates C++ code and Docker is invoked automatically
python scripts/run_benchmark.py --agent gpt-4o --solver-library dealii

# Test only the v1 dataset with 241 DOLFINx cases
python scripts/run_benchmark.py --agent gpt-4o --version v1
```

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

# Multi-attempt mode, retrying with the previous error message
python scripts/run_benchmark.py --agent gpt-4o --max-attempts 3

# Skip LLM calls and directly evaluate existing solvers
python scripts/run_benchmark.py --agent gpt-4o --skip-generation

# Evaluate a specified solver file
python scripts/run_benchmark.py --agent gpt-4o \
    --solver-path results/gpt-4o/poisson_basic/solver.py \
    --cases poisson_basic

# Batch-evaluate all solvers in an existing directory without calling the LLM again
python scripts/run_benchmark.py --agent qwen3-max \
    --eval-existing-dir results/qwen3-max
```

### Arguments

| Argument | Default | Description |
|------|--------|------|
| `--agent` | Required | LLM name; multiple agents are supported. See the supported list below. |
| `--version` | `v2` | Dataset version: `v1` (241 cases) or `v2` (645 cases). |
| `--solver-library` | `dolfinx` | Solver backend: `dolfinx` \| `firedrake` \| `dealii`. |
| `--cases` | All | Case IDs to evaluate, separated by spaces. |
| `--equation-types` | All | Equation types to evaluate, such as `poisson heat`. |
| `--max-attempts` | `1` | Maximum number of attempts in multi-attempt mode. |
| `--timeout` | `300` | Per-case execution timeout in seconds. |
| `--output` | `results/` | Output directory for results. |
| `--eval-existing-dir` | None | Batch-evaluate an existing solver directory. |
| `--skip-generation` | No | Skip LLM calls and reuse existing solvers. |

## Supported LLMs and Agents

### LLMs (Language Models Only)

| Name | Provider |
|------|--------|
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
|------|------|----------|
| `codepde` | CodePDE Agent | `pdebench/configs/codepde.json` |
| `openhands` | OpenHands Agent | `pdebench/configs/openhands.json` |
| `mini-swe-agent` | MiniSWE-Agent | `pdebench/configs/mini-swe-agent.json` |

Code agents use the same prompts as LLMs. Environment variables and timeouts are configured through `pdebench/configs/{agent}.json`.

## Datasets

### v2 (Recommended, 645 Cases)

`data/benchmark_v2.jsonl` contains one JSON object per line, covers all three solver backends, and uses the following case structure:

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

### v1 (241 Cases, DOLFINx Only)

`data/benchmark_v1.jsonl` is a smaller subset for compatibility with earlier experiments and supports only the `dolfinx` backend.


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
│   │   ├── firedrake_oracle/   # Firedrake backend implementation
│   │   ├── dealii_oracle/      # deal.II backend implementation with .cc programs
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

## Docker Support

The Firedrake and deal.II backends run in Docker containers by default and do not require local installation:

```bash
# Pull images
docker pull pdebench/firedrake:latest
docker pull pdebench/dealii:latest

# Run evaluations with Docker invoked automatically
python scripts/run_benchmark.py --agent gpt-4o --solver-library firedrake
python scripts/run_benchmark.py --agent gpt-4o --solver-library dealii
```

## Developer Guide

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

## Citation

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


## License

This project is intended for research on evaluating AI scientific programming capabilities.

## Acknowledgements

- The [FEniCSx / DOLFINx](https://fenicsproject.org/) team
- The [Firedrake](https://www.firedrakeproject.org/) team
- The [deal.II](https://www.dealii.org/) team
- [SWE-bench](https://www.swebench.com/) for inspiration in evaluation design
