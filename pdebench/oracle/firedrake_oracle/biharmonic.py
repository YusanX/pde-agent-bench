"""Firedrake Biharmonic oracle (two Poisson solves: -Δw=f, -Δu=w)."""
from __future__ import annotations

import time
from typing import Any, Dict

import sympy as sp

from firedrake import (
    Function, TrialFunction, TestFunction, SpatialCoordinate,
    DirichletBC, Constant, interpolate, inner, grad, dx, solve,
)

from .common import (
    OracleResult, compute_rel_L2_grid,
    create_mesh, create_scalar_space,
    parse_expression,
    build_scalar_bc,
    sample_scalar_on_grid,
    _scalar_solver_params,
    _eval_exact_sym_on_grid,
    _apply_domain_mask,
)


class FiredrakeBiharmonicSolver:
    """Biharmonic oracle via split Poisson: -Δw=f then -Δu=w."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        t_start = time.perf_counter()

        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(msh, case_spec["fem"]["family"], case_spec["fem"]["degree"])
        x = SpatialCoordinate(msh)

        pde_cfg = case_spec["pde"]
        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")

        u_exact_fn = None
        w_exact_fn = None
        f_ufl = None

        if "u" in manufactured:
            sx, sy = sp.symbols("x y", real=True)
            u_sym = sp.sympify(manufactured["u"], locals={"x": sx, "y": sy, "pi": sp.pi})
            lap_u = sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)
            w_sym = -lap_u
            f_sym = -(sp.diff(w_sym, sx, 2) + sp.diff(w_sym, sy, 2))
            f_ufl = parse_expression(f_sym, x)
            u_exact_fn = Function(V)
            u_exact_fn.interpolate(parse_expression(u_sym, x))
            w_exact_fn = Function(V)
            w_exact_fn.interpolate(parse_expression(w_sym, x))
        else:
            if source_expr is not None:
                try:
                    f_ufl = Constant(float(sp.sympify(source_expr)))
                except Exception:
                    f_ufl = parse_expression(source_expr, x)
            else:
                f_ufl = Constant(0.0)

        solver_params = case_spec.get("oracle_solver", {})
        sp_dict = _scalar_solver_params(solver_params)

        # --- Solve -Δw = f ---
        w_trial = TrialFunction(V)
        v1 = TestFunction(V)
        a_w = inner(grad(w_trial), grad(v1)) * dx
        L_w = f_ufl * v1 * dx
        bcs_w = [DirichletBC(V, w_exact_fn if w_exact_fn is not None else Constant(0.0), "on_boundary")]
        w_h = Function(V)
        solve(a_w == L_w, w_h, bcs=bcs_w, solver_parameters=sp_dict)

        # --- Solve -Δu = w ---
        u_trial = TrialFunction(V)
        v2 = TestFunction(V)
        a_u = inner(grad(u_trial), grad(v2)) * dx
        L_u = w_h * v2 * dx
        if u_exact_fn is not None:
            bcs_u = [DirichletBC(V, u_exact_fn, "on_boundary")]
        else:
            bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            bcs_u = [build_scalar_bc(V, bc_cfg.get("value", "0.0"), x)]
        u_h = Function(V)
        solve(a_u == L_u, u_h, bcs=bcs_u, solver_parameters=sp_dict)

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_scalar_on_grid(u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])

        baseline_error = 0.0
        if u_exact_fn is not None:
            u_exact_grid = _apply_domain_mask(u_grid, _eval_exact_sym_on_grid(u_sym, (sx, sy), grid_cfg))
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            u_grid = u_exact_grid
        else:
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_solver = ref_cfg.get("oracle_solver", {})
            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_V = create_scalar_space(ref_msh, ref_fem_spec["family"], ref_fem_spec["degree"])
            ref_x = SpatialCoordinate(ref_msh)
            if source_expr is not None:
                try:
                    ref_f = Constant(float(sp.sympify(source_expr)))
                except Exception:
                    ref_f = parse_expression(source_expr, ref_x)
            else:
                ref_f = Constant(0.0)
            ref_sp = {**sp_dict, "ksp_rtol": ref_solver.get("rtol", 1e-12)}
            ref_w_t = TrialFunction(ref_V)
            ref_v1 = TestFunction(ref_V)
            ref_bcs_w = [DirichletBC(ref_V, Constant(0.0), "on_boundary")]
            ref_w_h = Function(ref_V)
            solve(inner(grad(ref_w_t), grad(ref_v1)) * dx == ref_f * ref_v1 * dx,
                  ref_w_h, bcs=ref_bcs_w, solver_parameters=ref_sp)
            ref_u_t = TrialFunction(ref_V)
            ref_v2 = TestFunction(ref_V)
            ref_bc_v = case_spec.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
            ref_bcs_u = [build_scalar_bc(ref_V, ref_bc_v, ref_x)]
            ref_u_h = Function(ref_V)
            solve(inner(grad(ref_u_t), grad(ref_v2)) * dx == ref_w_h * ref_v2 * dx,
                  ref_u_h, bcs=ref_bcs_u, solver_parameters=ref_sp)
            _, _, ref_grid = sample_scalar_on_grid(
                ref_u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid

        baseline_time = time.perf_counter() - t_start
        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info={
                "ksp_type": sp_dict["ksp_type"],
                "pc_type": sp_dict["pc_type"],
                "rtol": sp_dict["ksp_rtol"],
            },
            num_dofs=V.dof_count,
        )
