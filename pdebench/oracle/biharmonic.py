"""Biharmonic oracle solver (elliptic, 4th order) via a mixed Poisson formulation.

We solve Δ² u = f on the unit square using:
    -Δ w = f
    -Δ u = w

with Dirichlet conditions:
    u = g_u on ∂Ω
    w = g_w on ∂Ω

For manufactured solutions, g_u is the exact u and g_w is the exact w = -Δu.
For no-exact cases, we use u Dirichlet from case_spec.bc (default 0) and w=0.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import sympy as sp
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from .common import (
    OracleResult,
    compute_rel_L2_grid,
    create_mesh,
    create_scalar_space,
    locate_all_boundary_dofs,
    build_dirichlet_bc,
    interpolate_expression,
    parse_expression,
    sample_scalar_on_grid,
)


def _rhs_scalar(msh, x, source_expr: str | None):
    """Robust RHS for Poisson: return fem.Constant for constants to keep domain bound."""
    if source_expr is None:
        return fem.Constant(msh, 0.0)
    try:
        c = float(sp.sympify(source_expr))
        return fem.Constant(msh, c)
    except Exception:
        return parse_expression(source_expr, x)


class BiharmonicSolver:
    """Oracle solver for biharmonic equation using two Poisson solves."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        # ⏱️ 开始计时整个求解流程
        t_start_total = time.perf_counter()
        
        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(msh, case_spec["fem"]["family"], case_spec["fem"]["degree"])

        pde_cfg = case_spec["pde"]
        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")

        x = ufl.SpatialCoordinate(msh)
        u_exact = None
        w_exact = None
        f_expr = None

        if "u" in manufactured:
            sx, sy = sp.symbols("x y", real=True)
            u_sym = sp.sympify(manufactured["u"], locals={"x": sx, "y": sy, "pi": sp.pi})
            lap_u = sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)
            w_sym = -lap_u
            lap_w = sp.diff(w_sym, sx, 2) + sp.diff(w_sym, sy, 2)
            f_sym = -(lap_w)  # -Δw = f

            f_expr = parse_expression(f_sym, x)

            u_exact_expr = parse_expression(u_sym, x)
            w_exact_expr = parse_expression(w_sym, x)
            u_exact = fem.Function(V)
            w_exact = fem.Function(V)
            interpolate_expression(u_exact, u_exact_expr)
            interpolate_expression(w_exact, w_exact_expr)
        else:
            f_expr = _rhs_scalar(msh, x, source_expr)

        solver_params = case_spec.get("oracle_solver", {})
        petsc_options = {
            "ksp_type": solver_params.get("ksp_type", "cg"),
            "pc_type": solver_params.get("pc_type", "hypre"),
            "ksp_rtol": solver_params.get("rtol", 1e-10),
            "ksp_atol": solver_params.get("atol", 1e-12),
        }

        # -----------------------
        # Solve for w: -Δw = f
        # -----------------------
        w = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a_w = ufl.inner(ufl.grad(w), ufl.grad(v)) * ufl.dx
        L_w = f_expr * v * ufl.dx

        if w_exact is not None:
            boundary_dofs = locate_all_boundary_dofs(msh, V)
            bcs_w = [fem.dirichletbc(w_exact, boundary_dofs)]
        else:
            bcs_w = [build_dirichlet_bc(msh, V, "0.0")]

        w_problem = LinearProblem(
            a_w,
            L_w,
            bcs=bcs_w,
            petsc_options=petsc_options,
            petsc_options_prefix="oracle_biharmonic_w_",
        )

        w_h = w_problem.solve()

        # -----------------------
        # Solve for u: -Δu = w
        # -----------------------
        u = ufl.TrialFunction(V)
        q = ufl.TestFunction(V)
        a_u = ufl.inner(ufl.grad(u), ufl.grad(q)) * ufl.dx
        L_u = w_h * q * ufl.dx

        if u_exact is not None:
            boundary_dofs = locate_all_boundary_dofs(msh, V)
            bcs_u = [fem.dirichletbc(u_exact, boundary_dofs)]
        else:
            bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            bc_value = bc_cfg.get("value", "0.0")
            bcs_u = [build_dirichlet_bc(msh, V, bc_value)]

        u_problem = LinearProblem(
            a_u,
            L_u,
            bcs=bcs_u,
            petsc_options=petsc_options,
            petsc_options_prefix="oracle_biharmonic_u_",
        )
        u_h = u_problem.solve()

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_scalar_on_grid(u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])

        baseline_error = 0.0
        solver_info: Dict[str, Any] = {
            "ksp_type": petsc_options["ksp_type"],
            "pc_type": petsc_options["pc_type"],
            "rtol": petsc_options["ksp_rtol"],
        }

        if u_exact is not None:
            _, _, u_exact_grid = sample_scalar_on_grid(
                u_exact, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            u_grid = u_exact_grid
        else:
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_solver = ref_cfg.get("oracle_solver", {})

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_V = create_scalar_space(ref_msh, ref_fem_spec["family"], ref_fem_spec["degree"])
            ref_x = ufl.SpatialCoordinate(ref_msh)
            ref_f = _rhs_scalar(ref_msh, ref_x, source_expr)

            ref_petsc = {
                "ksp_type": ref_solver.get("ksp_type", petsc_options["ksp_type"]),
                "pc_type": ref_solver.get("pc_type", petsc_options["pc_type"]),
                "ksp_rtol": ref_solver.get("rtol", 1e-12),
                "ksp_atol": ref_solver.get("atol", 1e-14),
            }

            # w_ref
            ref_w = ufl.TrialFunction(ref_V)
            ref_v = ufl.TestFunction(ref_V)
            ref_a_w = ufl.inner(ufl.grad(ref_w), ufl.grad(ref_v)) * ufl.dx
            ref_L_w = ref_f * ref_v * ufl.dx
            ref_bcs_w = [build_dirichlet_bc(ref_msh, ref_V, "0.0")]
            ref_w_problem = LinearProblem(
                ref_a_w,
                ref_L_w,
                bcs=ref_bcs_w,
                petsc_options=ref_petsc,
                petsc_options_prefix="oracle_biharmonic_ref_w_",
            )
            ref_w_h = ref_w_problem.solve()

            # u_ref
            ref_u = ufl.TrialFunction(ref_V)
            ref_q = ufl.TestFunction(ref_V)
            ref_a_u = ufl.inner(ufl.grad(ref_u), ufl.grad(ref_q)) * ufl.dx
            ref_L_u = ref_w_h * ref_q * ufl.dx
            ref_bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            ref_bc_value = ref_bc_cfg.get("value", "0.0")
            ref_bcs_u = [build_dirichlet_bc(ref_msh, ref_V, ref_bc_value)]
            ref_u_problem = LinearProblem(
                ref_a_u,
                ref_L_u,
                bcs=ref_bcs_u,
                petsc_options=ref_petsc,
                petsc_options_prefix="oracle_biharmonic_ref_u_",
            )
            ref_u_h = ref_u_problem.solve()

            _, _, ref_grid = sample_scalar_on_grid(
                ref_u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid
            solver_info["reference_resolution"] = ref_mesh_spec.get("resolution")
            solver_info["reference_degree"] = ref_fem_spec.get("degree")

        # ⏱️ 结束计时（包含完整流程）
        baseline_time = time.perf_counter() - t_start_total

        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info=solver_info,
            num_dofs=V.dofmap.index_map.size_global,
        )

