"""Convection-Diffusion oracle solver."""
from __future__ import annotations

import time
from typing import Any, Dict

import sympy as sp
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from .common import (
    OracleResult,
    compute_L2_error,
    compute_rel_L2_grid,
    create_mesh,
    create_scalar_space,
    locate_all_boundary_dofs,
    build_dirichlet_bc,
    interpolate_expression,
    parse_expression,
    sample_scalar_on_grid,
)


class ConvectionDiffusionSolver:
    """Convection-diffusion solver with optional SUPG stabilization."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(msh, case_spec["fem"]["family"], case_spec["fem"]["degree"])

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        epsilon = float(params.get("epsilon", 0.01))
        beta = params.get("beta", [1.0, 1.0])
        beta_vec = ufl.as_vector(beta)
        source_expr = pde_cfg.get("source_term")

        x = ufl.SpatialCoordinate(msh)
        manufactured = pde_cfg.get("manufactured_solution", {})
        u_exact = None
        f_expr = None

        if "u" in manufactured:
            sx, sy = sp.symbols("x y", real=True)
            u_sym = sp.sympify(manufactured["u"], locals={"x": sx, "y": sy})
            bx, by = beta
            f_sym = -epsilon * (sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)) + bx * sp.diff(u_sym, sx) + by * sp.diff(u_sym, sy)
            f_expr = parse_expression(f_sym, x)

            u_exact_expr = parse_expression(u_sym, x)
            u_exact = fem.Function(V)
            interpolate_expression(u_exact, u_exact_expr)
        elif source_expr is not None:
            f_expr = parse_expression(source_expr, x)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta_vec, ufl.grad(u)) * v) * ufl.dx
        L = (f_expr if f_expr is not None else 0.0) * v * ufl.dx

        stabilization = params.get("stabilization", None)
        if stabilization == "supg":
            h = ufl.CellDiameter(msh)
            beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
            tau = h / (2.0 * beta_norm + 1e-12)
            r = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u)) - (f_expr if f_expr is not None else 0.0)
            a += tau * ufl.dot(beta_vec, ufl.grad(v)) * r * ufl.dx
            L += tau * ufl.dot(beta_vec, ufl.grad(v)) * (f_expr if f_expr is not None else 0.0) * ufl.dx

        bcs = []
        if u_exact is not None:
            boundary_dofs = locate_all_boundary_dofs(msh, V)
            bcs = [fem.dirichletbc(u_exact, boundary_dofs)]
        else:
            bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            bc_value = bc_cfg.get("value", "0.0")
            bcs = [build_dirichlet_bc(msh, V, bc_value)]

        solver_params = case_spec.get("oracle_solver", {})
        petsc_options = {
            "ksp_type": solver_params.get("ksp_type", "gmres"),
            "pc_type": solver_params.get("pc_type", "ilu"),
            "ksp_rtol": solver_params.get("rtol", 1e-10),
            "ksp_atol": solver_params.get("atol", 1e-12),
        }

        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="oracle_convdiff_",
        )
        t_start = time.perf_counter()
        u_h = problem.solve()
        baseline_time = time.perf_counter() - t_start

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_scalar_on_grid(
            u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
        )

        baseline_error = 0.0
        solver_info = {
            "ksp_type": petsc_options["ksp_type"],
            "pc_type": petsc_options["pc_type"],
            "rtol": petsc_options["ksp_rtol"],
            "stabilization": stabilization or "none",
        }
        if u_exact is not None:
            _, _, u_exact_grid = sample_scalar_on_grid(
                u_exact, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            # Use exact grid as reference for evaluation alignment.
            u_grid = u_exact_grid
        else:
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_solver = ref_cfg.get("oracle_solver", {})

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_V = create_scalar_space(ref_msh, ref_fem_spec["family"], ref_fem_spec["degree"])
            ref_x = ufl.SpatialCoordinate(ref_msh)
            ref_f_expr = parse_expression(source_expr, ref_x) if source_expr is not None else 0.0
            ref_u = ufl.TrialFunction(ref_V)
            ref_v = ufl.TestFunction(ref_V)
            ref_a = (
                epsilon * ufl.inner(ufl.grad(ref_u), ufl.grad(ref_v))
                + ufl.dot(beta_vec, ufl.grad(ref_u)) * ref_v
            ) * ufl.dx
            ref_L = ref_f_expr * ref_v * ufl.dx
            ref_bcs = [build_dirichlet_bc(ref_msh, ref_V, bc_value)]
            ref_petsc_options = {
                "ksp_type": ref_solver.get("ksp_type", petsc_options["ksp_type"]),
                "pc_type": ref_solver.get("pc_type", petsc_options["pc_type"]),
                "ksp_rtol": ref_solver.get("rtol", 1e-12),
                "ksp_atol": ref_solver.get("atol", 1e-14),
            }
            ref_problem = LinearProblem(
                ref_a,
                ref_L,
                bcs=ref_bcs,
                petsc_options=ref_petsc_options,
                petsc_options_prefix="oracle_convdiff_ref_",
            )
            ref_u_h = ref_problem.solve()
            _, _, ref_grid = sample_scalar_on_grid(
                ref_u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid
            solver_info["reference_resolution"] = ref_mesh_spec.get("resolution")
            solver_info["reference_degree"] = ref_fem_spec.get("degree")

        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info=solver_info,
            num_dofs=V.dofmap.index_map.size_global,
        )
