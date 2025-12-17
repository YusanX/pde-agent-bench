# Agent Task: Optimize Linear Solver Performance

## Background
PDEBench is a benchmark suite for Partial Differential Equations. The current default linear solver strategy (`baseline.py`) is very basic, leading to suboptimal efficiency on complex cases.

We need you to act as a **High-Performance Computing (HPC) Optimization Engineer**. Your task is to modify the solver code to improve performance while maintaining physical correctness.

## Goal
Optimize the `solve_linear` function in `pdebench/pdebench/linsolve/baseline.py` to achieve **lower Total Wall Time** in the benchmark suite.

## Actions You Can Take
1. **Change KSP Type**: Try GMRES (`gmres`), MinRes (`minres`), BiCGSTAB (`bcgs`), etc.
2. **Change Preconditioner**: Try ILU (`ilu`), SOR (`sor`), GAMG (`gamg`), ICC (`icc`), etc.
3. **Parameter Tuning**: Adjust convergence tolerances (`rtol`, `atol`) carefully, ensuring accuracy is preserved.
4. **Adaptive Strategy**: Implement logic to dynamically select the best solver strategy based on matrix properties (e.g., size, symmetry) or PDE type.

## Constraints (Strictly Follow)
1. **ONLY Modify** `pdebench/pdebench/linsolve/baseline.py`.
2. **DO NOT Modify** any JSON case files in `pdebench/cases/`.
3. **DO NOT Modify** any test code in `pdebench/tests/`.
4. **DO NOT Modify** the `solve_linear_direct` function (this is the Ground Truth for reference).
5. You must ensure **100% Pass Rate** for all demo cases.

## How to Verify Your Work

1. **Run Benchmark**:
   ```bash
   python scripts/benchmark_score.py --log-history --experiment-id "<YOUR_MODEL_NAME>_run"
   ```
   **CRITICAL**: You MUST use `--log-history` to record your score. Replace `<YOUR_MODEL_NAME>` with your specific model name (e.g., `gpt4o_run`, `claude4.5_run`).
   
2. **Check Results**:
   - The output will show a `🏆 Final Score Summary`.
   - Focus on **`Total Wall Time`** (Lower is Better) and **`Success Rate`** (Must be 10/10).
   - If the success rate drops, your optimization has compromised accuracy and must be reverted or fixed.

## Expected Output
A modified `pdebench/pdebench/linsolve/baseline.py` that achieves a faster runtime than the original version in `scripts/benchmark_score.py` while passing all tests.
