# Agent Task: Optimize Linear Solver Performance (PDEBench)

## Goal
Improve runtime in `python scripts/benchmark_score.py` by optimizing ONLY the iterative solver implementation,
while preserving numerical correctness.

## File to modify (STRICT)
- ONLY modify: `pdebench/linsolve/baseline.py`
- DO NOT modify: any file under `cases/`, `tests/`, `pdebench/linsolve/reference.py` (or `solve_linear_direct`), or evaluation code.

## Correctness constraints (MUST HOLD)
For every demo case in `scripts/benchmark_score.py`:
1) KSP must CONVERGE: `converged == True` and `converged_reason > 0`.
2) Accuracy must not degrade: the case must still meet its `targets.metric <= targets.target_error`.
3) Residual quality must not degrade: require `rel_res <= 1e-8` (or the project’s existing threshold).

If any case violates these, the optimization is invalid even if it looks faster.

## Allowed actions
- Change KSP type (cg/gmres/minres/bcgs)
- Change PC (jacobi/ilu/icc/gamg/sor)
- Tune KSP tolerances and max_it, but do NOT weaken correctness constraints above.
- Add a lightweight adaptive heuristic (e.g., choose CG for symmetric SPD-like matrices, GMRES otherwise).

## Benchmark protocol (MUST FOLLOW)
1) First record BASELINE score (no code changes):
   `python scripts/benchmark_score.py --log-history --experiment-id "<MODEL>_baseline"`
2) Then implement your optimization and re-run:
   `python scripts/benchmark_score.py --log-history --experiment-id "<MODEL>_opt"`
3) Your final output must include:
   - The best `Total Wall Time` you achieved (from the logged summary)
   - Confirmation that Success Rate is 10/10 AND all correctness constraints hold

Use `<MODEL>` as a short identifier like `gpt52`, `claude45`, etc.
