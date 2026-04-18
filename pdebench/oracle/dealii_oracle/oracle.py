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
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import sympy as sp

from .._types import OracleResult, compute_rel_L2_grid
from .common import ensure_built, preprocess_case_spec, run_oracle_program

# Paths resolved relative to this file so the oracle works regardless of cwd
_ORACLE_DIR   = Path(__file__).resolve().parent
_PROGRAMS_DIR = _ORACLE_DIR / "programs"

# 优先使用预编译目录（Docker 镜像内由 ENV DEALII_ORACLE_PREBUILT_DIR 指定），
# 回退到源码旁的 build/ 目录（本地开发环境）
_PREBUILT_ENV = os.environ.get("DEALII_ORACLE_PREBUILT_DIR", "")
_BUILD_DIR    = Path(_PREBUILT_ENV) if _PREBUILT_ENV else _ORACLE_DIR / "build"


def _apply_domain_mask(
    u_fem_grid: Optional[np.ndarray],
    u_exact_grid: np.ndarray,
) -> np.ndarray:
    """将 FEM 采样的域内掩码应用到精确解网格。

    C++ eval_scalar_at_points / eval_vector_at_points 修复后，复杂域的域外点
    在 FEM 采样结果（grid）中为 NaN。将相同位置在精确解中也设为 NaN，使
    误差计算只覆盖域内点。对简单（矩形/立方体）域无域外 NaN 时为空操作。
    """
    if u_fem_grid is None or not np.any(np.isnan(u_fem_grid)):
        return u_exact_grid
    masked = u_exact_grid.copy()
    masked[np.isnan(u_fem_grid)] = np.nan
    return masked


def _sample_exact_scalar_grid(
    expr_str: str,
    grid_cfg: Dict[str, Any],
    *,
    t_value: Optional[float] = None,
) -> np.ndarray:
    """Sample a scalar sympy expression on the evaluator grid."""
    sx, sy, sz, st = sp.symbols("x y z t", real=True)
    expr = sp.sympify(str(expr_str), locals={"x": sx, "y": sy, "z": sz, "t": st, "pi": sp.pi})
    if t_value is not None:
        expr = expr.subs(st, float(t_value))

    bbox = grid_cfg["bbox"]
    nx = int(grid_cfg["nx"])
    ny = int(grid_cfg["ny"])
    is_3d = len(bbox) == 6 and "nz" in grid_cfg

    if is_3d:
        nz = int(grid_cfg["nz"])
        fn = sp.lambdify((sx, sy, sz), expr, modules="numpy")
        x_lin = np.linspace(bbox[0], bbox[1], nx)
        y_lin = np.linspace(bbox[2], bbox[3], ny)
        z_lin = np.linspace(bbox[4], bbox[5], nz)
        zz, yy, xx = np.meshgrid(z_lin, y_lin, x_lin, indexing="ij")
        values = np.asarray(fn(xx, yy, zz), dtype=np.float64)
        if values.shape == ():
            values = np.full((nz, ny, nx), float(values), dtype=np.float64)
        return values.reshape(nz, ny, nx)

    fn = sp.lambdify((sx, sy), expr, modules="numpy")
    x_lin = np.linspace(bbox[0], bbox[1], nx)
    y_lin = np.linspace(bbox[2], bbox[3], ny)
    xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
    values = np.asarray(fn(xx, yy), dtype=np.float64)
    if values.shape == ():
        values = np.full((ny, nx), float(values), dtype=np.float64)
    return values.reshape(ny, nx)


def _sample_exact_vector_magnitude_grid(
    expr_list: list,
    grid_cfg: Dict[str, Any],
    *,
    t_value: Optional[float] = None,
) -> np.ndarray:
    """
    Sample vector field components and return their L2 magnitude grid.
    Matches write_vector_magnitude_grid in grid_writer.h (stokes/navier_stokes/linear_elasticity).
    """
    sx, sy, sz, st = sp.symbols("x y z t", real=True)
    bbox = grid_cfg["bbox"]
    nx = int(grid_cfg["nx"])
    ny = int(grid_cfg["ny"])
    is_3d = len(bbox) == 6 and "nz" in grid_cfg

    if is_3d:
        nz = int(grid_cfg["nz"])
        x_lin = np.linspace(bbox[0], bbox[1], nx)
        y_lin = np.linspace(bbox[2], bbox[3], ny)
        z_lin = np.linspace(bbox[4], bbox[5], nz)
        zz, yy, xx = np.meshgrid(z_lin, y_lin, x_lin, indexing="ij")
        mag_sq = np.zeros((nz, ny, nx), dtype=np.float64)
    else:
        x_lin = np.linspace(bbox[0], bbox[1], nx)
        y_lin = np.linspace(bbox[2], bbox[3], ny)
        xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
        mag_sq = np.zeros((ny, nx), dtype=np.float64)

    for expr_str in expr_list:
        expr = sp.sympify(str(expr_str), locals={"x": sx, "y": sy, "z": sz, "t": st, "pi": sp.pi})
        if t_value is not None:
            expr = expr.subs(st, float(t_value))

        if is_3d:
            fn = sp.lambdify((sx, sy, sz), expr, modules="numpy")
            comp = np.asarray(fn(xx, yy, zz), dtype=np.float64)
            if comp.shape == ():
                comp = np.full((nz, ny, nx), float(comp), dtype=np.float64)
            mag_sq += comp.reshape(nz, ny, nx) ** 2
        else:
            fn = sp.lambdify((sx, sy), expr, modules="numpy")
            comp = np.asarray(fn(xx, yy), dtype=np.float64)
            if comp.shape == ():
                comp = np.full((ny, nx), float(comp), dtype=np.float64)
            mag_sq += comp.reshape(ny, nx) ** 2

    return np.sqrt(mag_sq)


def _reference_time_value(pde_cfg: Dict[str, Any]) -> Optional[float]:
    """Use terminal time when evaluating transient manufactured solutions."""
    time_cfg = pde_cfg.get("time", {})
    if not isinstance(time_cfg, dict):
        return None
    if "t_end" in time_cfg:
        return float(time_cfg["t_end"])
    if "t0" in time_cfg:
        return float(time_cfg["t0"])
    return None


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


def _biharmonic_reference_grid(
    solver: "DealIIOracleSolver",
    case_spec: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Build a reference grid for Biharmonic:
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
    solver._ensure_built("biharmonic")
    ref_grid, _ = run_oracle_program(
        pde_type="biharmonic",
        case_spec=ref_enriched,
        build_dir=_BUILD_DIR,
        timeout_sec=solver._timeout,
    )
    return ref_grid, {
        "reference_resolution": ref_case["mesh"].get("resolution"),
        "reference_degree": ref_case["fem"].get("degree"),
    }


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


def _scalar_pde_reference_grid(
    solver: "DealIIOracleSolver",
    case_spec: Dict[str, Any],
    pde_type: str,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Generic reference grid builder for scalar PDEs (convection_diffusion, reaction_diffusion).
    - exact grid if manufactured_solution.u is a scalar string
    - otherwise a higher-accuracy solve using reference_config
    """
    pde_cfg = case_spec["pde"]
    manufactured = pde_cfg.get("manufactured_solution", {})
    grid_cfg = case_spec["output"]["grid"]
    t_value = _reference_time_value(pde_cfg)

    if "u" in manufactured and isinstance(manufactured["u"], str):
        return _sample_exact_scalar_grid(manufactured["u"], grid_cfg, t_value=t_value), {}

    ref_cfg = case_spec.get("reference_config", {})
    if not ref_cfg:
        return None, {}

    ref_case = copy.deepcopy(case_spec)
    ref_case["mesh"] = ref_cfg.get("mesh", case_spec["mesh"])
    ref_case["fem"] = ref_cfg.get("fem", case_spec["fem"])
    ref_oracle_solver = ref_cfg.get("oracle_solver", case_spec.get("oracle_solver", {}))

    # deal.II quad mesh has ~2x more DoFs than a dolfinx triangle mesh at the
    # same resolution (Q2 vs P2).  For large reference meshes, GMRES+ILU(0)
    # can time out even though dolfinx's PETSc ILU succeeds easily.
    # Override to UMFPACK direct solver when no direct solver is already
    # requested — this matches the Stokes reference override strategy and
    # ensures robustness regardless of problem conditioning.
    ref_ksp = ref_oracle_solver.get("ksp_type", "gmres")
    ref_pc  = ref_oracle_solver.get("pc_type",  "ilu")
    if ref_ksp not in ("preonly",) and ref_pc not in ("lu", "mumps", "direct"):
        ref_oracle_solver = {
            **ref_oracle_solver,
            "ksp_type": "preonly",
            "pc_type":  "lu",
        }

    ref_case["oracle_solver"] = ref_oracle_solver

    ref_enriched = preprocess_case_spec(ref_case)
    solver._ensure_built(pde_type)
    ref_grid, _ = run_oracle_program(
        pde_type=pde_type,
        case_spec=ref_enriched,
        build_dir=_BUILD_DIR,
        timeout_sec=solver._timeout,
    )
    return ref_grid, {
        "reference_resolution": ref_case["mesh"].get("resolution"),
        "reference_degree": ref_case["fem"].get("degree"),
    }


def _vector_magnitude_pde_reference_grid(
    solver: "DealIIOracleSolver",
    case_spec: Dict[str, Any],
    pde_type: str,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Generic reference grid builder for vector PDEs (stokes, navier_stokes, linear_elasticity).
    C++ binary outputs velocity/displacement magnitude via write_vector_magnitude_grid.
    - exact magnitude grid if manufactured_solution.u is a list of expressions
    - otherwise a higher-accuracy solve using reference_config
    """
    pde_cfg = case_spec["pde"]
    manufactured = pde_cfg.get("manufactured_solution", {})
    grid_cfg = case_spec["output"]["grid"]
    t_value = _reference_time_value(pde_cfg)

    if "u" in manufactured and isinstance(manufactured["u"], list):
        return _sample_exact_vector_magnitude_grid(manufactured["u"], grid_cfg, t_value=t_value), {}

    ref_cfg = case_spec.get("reference_config", {})
    if not ref_cfg:
        return None, {}

    ref_case = copy.deepcopy(case_spec)
    ref_case["mesh"] = ref_cfg.get("mesh", case_spec["mesh"])
    ref_case["fem"] = ref_cfg.get("fem", case_spec["fem"])
    ref_oracle_solver = ref_cfg.get("oracle_solver", case_spec.get("oracle_solver", {}))

    # deal.II doesn't have PETSc/Hypre. For Stokes/NS reference solves that
    # specify an AMG-type iterative solver (minres+hypre, gmres+hypre …),
    # unpreconditioned Krylov diverges or times out on large meshes (200+).
    # Override to UMFPACK direct solver on a manageable P2/P1 mesh (<=128)
    # which is still substantially more accurate than the coarse solve.
    if pde_type == "stokes":
        ref_ksp = ref_oracle_solver.get("ksp_type", "")
        ref_pc  = ref_oracle_solver.get("pc_type", "")
        if ref_ksp not in ("preonly",) and ref_pc not in ("lu", "mumps"):
            ref_res = min(ref_case["mesh"].get("resolution", 128), 128)
            ref_case["mesh"] = {**ref_case["mesh"], "resolution": ref_res}
            ref_case["fem"]  = {"degree_u": 2, "degree_p": 1}
            ref_oracle_solver = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "rtol": 1e-12,
                "pressure_fixing": ref_oracle_solver.get("pressure_fixing", "point"),
            }

    if pde_type == "navier_stokes":
        # deal.II NS always uses UMFPACK internally (Picard). The reference
        # mesh can be very large (140-180) causing many slow factorizations.
        # Cap at 96 to keep Picard reference solve within the timeout budget
        # while still providing a substantially finer reference than the
        # baseline (which is typically resolution 32-64).
        ref_res = min(ref_case["mesh"].get("resolution", 96), 96)
        ref_case["mesh"] = {**ref_case["mesh"], "resolution": ref_res,
                             "cell_type": "quadrilateral"}
        ref_case["fem"]  = {"degree_u": 2, "degree_p": 1}
        ref_oracle_solver = {
            **ref_oracle_solver,
            "rtol": 1e-10,
            "atol": 1e-12,
        }

    ref_case["oracle_solver"] = ref_oracle_solver

    ref_enriched = preprocess_case_spec(ref_case)
    solver._ensure_built(pde_type)
    ref_grid, _ = run_oracle_program(
        pde_type=pde_type,
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

    def __init__(self, timeout_sec: int = 900):
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
        grid_cfg = enriched.get("output", {}).get("grid", {})
        is_3d = len(grid_cfg.get("bbox", [])) == 6 and "nz" in grid_cfg

        # 2. Compile oracle binaries if not yet done
        self._ensure_built(pde_type)

        # 2b. For Stokes, deal.II doesn't have a working AMG preconditioner for
        # the full saddle-point system. When the oracle_solver requests an
        # iterative solver with hypre/ilu/none preconditioning, unpreconditioned
        # Krylov diverges or gives inaccurate results on many problems.
        # Override to UMFPACK (direct solver) so the baseline solve is reliable.
        # The agent's submitted SOLUTION (not solver choice) is what gets graded.
        if pde_type == "stokes":
            os_cfg = enriched.get("oracle_solver", {})
            _ksp = os_cfg.get("ksp_type", "preonly")
            _pc  = os_cfg.get("pc_type", "lu")
            if _ksp not in ("preonly",) and _pc not in ("lu", "mumps"):
                enriched = copy.deepcopy(enriched)
                enriched["oracle_solver"] = {
                    **os_cfg,
                    "ksp_type": "preonly",
                    "pc_type":  "lu",
                }

        # 2c. In 3-D Helmholtz, direct LU factorizations can be prohibitively
        # memory-hungry inside the dealii Docker image and may be SIGKILLed
        # even for meshes that are otherwise fine with Krylov solves.
        # Prefer a robust iterative fallback instead of crashing the oracle.
        if pde_type == "helmholtz" and is_3d:
            os_cfg = enriched.get("oracle_solver", {})
            _ksp = os_cfg.get("ksp_type", "gmres")
            _pc  = os_cfg.get("pc_type", "ilu")
            if _ksp in ("preonly",) or _pc in ("lu", "mumps", "direct"):
                enriched = copy.deepcopy(enriched)
                enriched["oracle_solver"] = {
                    **os_cfg,
                    "ksp_type": "gmres",
                    "pc_type": "ilu",
                }

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
                ref_grid = _apply_domain_mask(grid, ref_grid)
                baseline_error = compute_rel_L2_grid(grid, ref_grid)
                reference = ref_grid
                solver_info.update(ref_info)
        elif pde_type == "heat":
            ref_grid, ref_info = _heat_reference_grid(self, case_spec)
            if ref_grid is not None:
                ref_grid = _apply_domain_mask(grid, ref_grid)
                baseline_error = compute_rel_L2_grid(grid, ref_grid)
                reference = ref_grid
                solver_info.update(ref_info)
        elif pde_type == "helmholtz":
            ref_grid, ref_info = _helmholtz_reference_grid(self, case_spec)
            if ref_grid is not None:
                ref_grid = _apply_domain_mask(grid, ref_grid)
                baseline_error = compute_rel_L2_grid(grid, ref_grid)
                reference = ref_grid
                solver_info.update(ref_info)
        elif pde_type == "biharmonic":
            ref_grid, ref_info = _biharmonic_reference_grid(self, case_spec)
            if ref_grid is not None:
                ref_grid = _apply_domain_mask(grid, ref_grid)
                baseline_error = compute_rel_L2_grid(grid, ref_grid)
                reference = ref_grid
                solver_info.update(ref_info)
        elif pde_type in ("convection_diffusion", "reaction_diffusion"):
            ref_grid, ref_info = _scalar_pde_reference_grid(self, case_spec, pde_type)
            if ref_grid is not None:
                ref_grid = _apply_domain_mask(grid, ref_grid)
                baseline_error = compute_rel_L2_grid(grid, ref_grid)
                reference = ref_grid
                solver_info.update(ref_info)
        elif pde_type in ("stokes", "navier_stokes", "linear_elasticity"):
            ref_grid, ref_info = _vector_magnitude_pde_reference_grid(self, case_spec, pde_type)
            if ref_grid is not None:
                ref_grid = _apply_domain_mask(grid, ref_grid)
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
