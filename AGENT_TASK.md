# Task: Solve a Specific PDE Case

You are a scientific computing expert. Your goal is to solve ONE specific PDE problem from the dataset and verify it.

## Steps
1.  **Read the Data:** Read the file `datasets/level_2_1_basic.jsonl` and extract the JSON object from the **first line** (Case ID: `heat_grid_target`).
2.  **Understand the Prompt:** Read the `prompt` field carefully. It contains specific parameters (diffusion coefficient `kappa`, time `T`, `dt`, etc.) and the required Output Grid size (e.g., 50x50).
3.  **Write the Solver:** Create a Python script named `my_solver.py` that solves this SPECIFIC problem.
    *   **CRITICAL:** You MUST **hard-code** the physics parameters (`kappa`, `T`, `dt`, `f`, `g`) and the output grid interpolation logic into your script based on the Prompt description. The evaluation harness will NOT pass these as arguments.
    *   **CRITICAL:** Your script MUST accept `--resolution`, `--degree`, and `--outdir` as command-line arguments.
    *   **CRITICAL:** Save outputs to `solution.npz` (with fields x,y,u) and `meta.json` (empty dict is fine) inside the folder specified by `--outdir`.
    *   **CRITICAL:** You MUST use `dolfinx` version **0.8.0 or higher**. Do NOT use legacy `dolfin` syntax.
4.  **Verify:** Run the evaluation script specifically for this case to check your work.

## How to Verify
Run the following command to evaluate your solver against the specific case:

```bash
python scripts/evaluate_agent.py \
    --dataset datasets/level_2_1_basic.jsonl \
    --agent-script my_solver.py \
    --outdir results/test_run \
    --limit 1
```

If the output shows "SUCCESS" or "PASS", you have completed the task.