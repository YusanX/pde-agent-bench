import argparse
import json
import time
import importlib.util
import numpy as np

def _load_module(path):
    spec = importlib.util.spec_from_file_location("agent_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _get_solver_fn(module):
    if hasattr(module, "solve") and callable(module.solve):
        return module.solve
    if hasattr(module, "solve_case") and callable(module.solve_case):
        return module.solve_case
    raise AttributeError("Expected solve(case_spec) or solve_case(case_spec) in agent script")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    with open(args.case) as f:
        case_spec = json.load(f)

    solver = _get_solver_fn(_load_module(args.script))

    t0 = time.time()
    result = solver(case_spec)
    t1 = time.time()

    if not isinstance(result, dict):
        raise ValueError("solve() must return a dict with keys: u (or u_grid) and solver_info")

    u_grid = result.get("u")
    if u_grid is None:
        u_grid = result.get("u_grid")
    solver_info = result.get("solver_info", {})

    if u_grid is None:
        raise ValueError("solve() returned no solution array")
    if not isinstance(solver_info, dict) or not solver_info:
        raise ValueError("solve() must return non-empty solver_info dict")

    required_keys = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol"]
    missing = [k for k in required_keys if k not in solver_info]
    if missing:
        raise ValueError(f"solver_info missing required keys: {missing}")

    u_grid = np.array(u_grid)

    grid = case_spec["oracle_config"]["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    if u_grid.ndim == 1 and u_grid.size == nx * ny:
        u_grid = u_grid.reshape((nx, ny))
    if u_grid.shape != (nx, ny):
        raise ValueError(f"Expected u_grid shape ({nx}, {ny}), got {u_grid.shape}")

    x = np.linspace(grid["bbox"][0], grid["bbox"][1], nx)
    y = np.linspace(grid["bbox"][2], grid["bbox"][3], ny)
    np.savez(f"{args.outdir}/solution.npz", x=x, y=y, u=u_grid)

    meta = {
        "wall_time_sec": t1 - t0,
        "solver_info": solver_info,
    }
    with open(f"{args.outdir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
