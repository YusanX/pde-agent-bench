"""
dealii_oracle/oracle.py
=======================

Python dispatcher for the deal.II oracle backend.

Workflow for each solve() call:
  1. preprocess_case_spec() – inject _computed_* expression fields
  2. ensure_built()         – cmake + make on first call (cached)
  3. run_oracle_program()   – invoke C++ binary via subprocess
  4. Wrap output in OracleResult
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import sympy as sp

from .._types import OracleResult, compute_rel_L2_grid
from .common import ensure_built, preprocess_case_spec, run_oracle_program

# Paths resolved relative to this file so the oracle works regardless of cwd
_ORACLE_DIR   = Path(__file__).resolve().parent
_PROGRAMS_DIR = _ORACLE_DIR / "programs"
_BUILD_DIR    = _ORACLE_DIR / "build"


def _sample_exact_scalar_grid(
    expr_str: str,
    grid_cfg: Dict[str, Any],
    *,
    t_value: Optional[float] = None,
) -> np.ndarray:
    """Sample a scalar sympy expression on the evaluator grid."""
    sx, sy, st = sp.symbols("x y t", real=True)
    expr = sp.sympify(str(expr_str), locals={"x": sx, "y": sy, "t": st, "pi": sp.pi})
    if t_value is not None:
        expr = expr.subs(st, float(t_value))
    fn = sp.lambdify((sx, sy), expr, modules="numpy")

    bbox = grid_cfg["bbox"]
    nx = int(grid_cfg["nx"])
    ny = int(grid_cfg["ny"])
    x_lin = np.linspace(bbox[0], bbox[1], nx)
    y_lin = np.linspace(bbox[2], bbox[3], ny)
    xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")

    values = np.asarray(fn(xx, yy), dtype=np.float64)
    if values.shape == ():
        values = np.full((ny, nx), float(values), dtype=np.float64)
    return values.reshape(ny, nx)


def _poisson_reference_grid(
    solver: "DealIIOracleSolver",
    case_spec: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Build a reference grid for Poisson:
    - exact grid if manufactured_solution.u exists
    - otherwise a higher-accuracy deal.II solve using reference_config
    """
    pde_cfg = case_spec["pde"]
    manufactured = pde_cfg.get("manufactured_solution", {})
    grid_cfg = case_spec["output"]["grid"]

    if "u" in manufactured:
        return _sample_exact_scalar_grid(manufactured["u"], grid_cfg), {}

    ref_cfg = case_spec.get("reference_config", {})
    if not ref_cfg:
        return None, {}

    ref_case = copy.deepcopy(case_spec)
    ref_case["mesh"] = ref_cfg.get("mesh", case_spec["mesh"])
    ref_case["fem"] = ref_cfg.get("fem", case_spec["fem"])
    ref_case["oracle_solver"] = ref_cfg.get("oracle_solver", case_spec.get("oracle_solver", {}))

    ref_enriched = preprocess_case_spec(ref_case)
    solver._ensure_built("poisson")
    ref_grid, _ = run_oracle_program(
        pde_type="poisson",
        case_spec=ref_enriched,
        build_dir=_BUILD_DIR,
        timeout_sec=solver._timeout,
    )
    return ref_grid, {
        "reference_resolution": ref_case["mesh"].get("resolution"),
        "reference_degree": ref_case["fem"].get("degree"),
    }


def _heat_reference_grid(
    solver: "DealIIOracleSolver",
    case_spec: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Build a reference grid for Heat:
    - exact grid sampled at final time if manufactured_solution.u exists
    - otherwise a higher-accuracy deal.II solve using reference_config
    """
    pde_cfg = case_spec["pde"]
    manufactured = pde_cfg.get("manufactured_solution", {})
    grid_cfg = case_spec["output"]["grid"]
    time_cfg = pde_cfg.get("time", {})
    t_end = float(time_cfg.get("t_end", 1.0))

    if "u" in manufactured:
        return _sample_exact_scalar_grid(manufactured["u"], grid_cfg, t_value=t_end), {}

    ref_cfg = case_spec.get("reference_config", {})
    if not ref_cfg:
        return None, {}

    ref_case = copy.deepcopy(case_spec)
    ref_case["mesh"] = ref_cfg.get("mesh", case_spec["mesh"])
    ref_case["fem"] = ref_cfg.get("fem", case_spec["fem"])
    ref_case["oracle_solver"] = ref_cfg.get("oracle_solver", case_spec.get("oracle_solver", {}))

    if "time" in ref_cfg:
        ref_case.setdefault("pde", {}).setdefault("time", {})
        ref_case["pde"]["time"].update(ref_cfg["time"])

    ref_enriched = preprocess_case_spec(ref_case)
    solver._ensure_built("heat")
    ref_grid, _ = run_oracle_program(
        pde_type="heat",
        case_spec=ref_enriched,
        build_dir=_BUILD_DIR,
        timeout_sec=solver._timeout,
    )
    ref_info: Dict[str, Any] = {
        "reference_resolution": ref_case["mesh"].get("resolution"),
        "reference_degree": ref_case["fem"].get("degree"),
    }
    if "time" in ref_cfg:
        ref_info["reference_dt"] = ref_case["pde"]["time"].get("dt")
    return ref_grid, ref_info


def _helmholtz_reference_grid(
    solver: "DealIIOracleSolver",
    case_spec: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Build a reference grid for Helmholtz:
    - exact grid if manufactured_solution.u exists
    - otherwise a higher-accuracy deal.II solve using reference_config
    """
    pde_cfg = case_spec["pde"]
    manufactured = pde_cfg.get("manufactured_solution", {})
    grid_cfg = case_spec["output"]["grid"]

    if "u" in manufactured:
        return _sample_exact_scalar_grid(manufactured["u"], grid_cfg), {}

    ref_cfg = case_spec.get("reference_config", {})
    if not ref_cfg:
        return None, {}

    ref_case = copy.deepcopy(case_spec)
    ref_case["mesh"] = ref_cfg.get("mesh", case_spec["mesh"])
    ref_case["fem"] = ref_cfg.get("fem", case_spec["fem"])
    ref_case["oracle_solver"] = ref_cfg.get("oracle_solver", case_spec.get("oracle_solver", {}))

    ref_enriched = preprocess_case_spec(ref_case)
    solver._ensure_built("helmholtz")
    ref_grid, _ = run_oracle_program(
        pde_type="helmholtz",
        case_spec=ref_enriched,
        build_dir=_BUILD_DIR,
        timeout_sec=solver._timeout,
    )
    return ref_grid, {
        "reference_resolution": ref_case["mesh"].get("resolution"),
        "reference_degree": ref_case["fem"].get("degree"),
    }


class DealIIOracleSolver:
    """
    Oracle backend that compiles deal.II C++ programs on first use and
    calls the appropriate binary for each PDE type.

    The interface mirrors FiredrakeOracleSolver: accepts oracle_config
    (the 'oracle_config' sub-dict from benchmark.jsonl) and returns an
    OracleResult with the same field semantics.
    """

    def __init__(self, timeout_sec: int = 300):
        self._timeout = timeout_sec
        self._built_pdes: Set[str] = set()

    def _ensure_built(self, pde_type: str) -> None:
        if pde_type not in self._built_pdes:
            ensure_built(_PROGRAMS_DIR, _BUILD_DIR, pde_type)
            self._built_pdes.add(pde_type)

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        """
        Solve one PDE case with the deal.II oracle.

        Args:
            case_spec: oracle_config dict from benchmark.jsonl
                       (same dict passed to OracleSolver.solve()).

        Returns:
            OracleResult with reference grid, baseline_error, baseline_time.
        """
        pde_type = case_spec["pde"]["type"]

        # 1. Inject _computed_* expression fields for C++
        enriched = preprocess_case_spec(case_spec)

        # 2. Compile oracle binaries if not yet done
        self._ensure_built(pde_type)

        # 3. Run C++ binary
        grid, meta = run_oracle_program(
            pde_type   = pde_type,
            case_spec  = enriched,
            build_dir  = _BUILD_DIR,
            timeout_sec = self._timeout,
        )

        # 4. Baseline error:
        baseline_error = 0.0
        reference = grid
        solver_info = {
            "ksp_type":  meta.get("ksp_type", ""),
            "pc_type":   meta.get("pc_type", ""),
            "rtol":      meta.get("rtol", 0.0),
            "library":   "dealii",
        }

        if pde_type == "poisson":
            ref_grid, ref_info = _poisson_reference_grid(self, case_spec)
            if ref_grid is not None:
                baseline_error = compute_rel_L2_grid(grid, ref_grid)
                reference = ref_grid
                solver_info.update(ref_info)
        elif pde_type == "heat":
            ref_grid, ref_info = _heat_reference_grid(self, case_spec)
            if ref_grid is not None:
                baseline_error = compute_rel_L2_grid(grid, ref_grid)
                reference = ref_grid
                solver_info.update(ref_info)
        elif pde_type == "helmholtz":
            ref_grid, ref_info = _helmholtz_reference_grid(self, case_spec)
            if ref_grid is not None:
                baseline_error = compute_rel_L2_grid(grid, ref_grid)
                reference = ref_grid
                solver_info.update(ref_info)

        return OracleResult(
            baseline_error = float(baseline_error),
            baseline_time  = float(meta.get("baseline_time", 0.0)),
            reference      = reference,
            solver_info    = solver_info,
            num_dofs       = int(meta.get("num_dofs", 0)),
        )
