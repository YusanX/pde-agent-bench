"""Stokes oracle solver (steady incompressible flow)."""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import sympy as sp
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

from .common import (
    OracleResult,
    compute_L2_error,
    compute_rel_L2_grid,
    create_mesh,
    create_mixed_space,
    interpolate_expression,
    parse_expression,
    parse_vector_expression,
    sample_vector_magnitude_on_grid,
)


class StokesSolver:
    """Taylor-Hood mixed solver for Stokes."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        degree_u = case_spec["fem"].get("degree_u", 2)
        degree_p = case_spec["fem"].get("degree_p", 1)
        W = create_mixed_space(msh, degree_u, degree_p)

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        nu = float(params.get("nu", 1.0))

        x = ufl.SpatialCoordinate(msh)
        manufactured = pde_cfg.get("manufactured_solution", {})
        u_exact = None
        p_exact = None
        f_expr = None

        if "u" in manufactured and "p" in manufactured:
            sx, sy = sp.symbols("x y", real=True)
            u_sym = manufactured["u"]
            p_sym = sp.sympify(manufactured["p"], locals={"x": sx, "y": sy})
            u_sym_x = sp.sympify(u_sym[0], locals={"x": sx, "y": sy})
            u_sym_y = sp.sympify(u_sym[1], locals={"x": sx, "y": sy})

            f_x = -nu * (sp.diff(u_sym_x, sx, 2) + sp.diff(u_sym_x, sy, 2)) + sp.diff(p_sym, sx)
            f_y = -nu * (sp.diff(u_sym_y, sx, 2) + sp.diff(u_sym_y, sy, 2)) + sp.diff(p_sym, sy)
            f_expr = parse_vector_expression([f_x, f_y], x)

            u_exact_expr = parse_vector_expression(u_sym, x)
            p_exact_expr = parse_expression(p_sym, x)

            V, Q = W.sub(0).collapse(), W.sub(1).collapse()
            u_exact = fem.Function(V)
            p_exact = fem.Function(Q)
            interpolate_expression(u_exact, u_exact_expr)
            interpolate_expression(p_exact, p_exact_expr)

        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)
        a = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.div(v) * p * ufl.dx
            - q * ufl.div(u) * ufl.dx
        )
        L = ufl.inner(f_expr if f_expr is not None else ufl.as_vector((0.0, 0.0)), v) * ufl.dx

        bcs = []
        if u_exact is not None:
            V = W.sub(0).collapse()
            boundary_dofs = fem.locate_dofs_geometrical(
                (W.sub(0), V), lambda x: (x[0] >= 0)
            )
            bcs = [fem.dirichletbc(u_exact, boundary_dofs, W.sub(0))]
        # Fix pressure at a single point to remove nullspace
        Q = W.sub(1).collapse()
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
        )
        if len(p_dofs) > 0:
            p0 = fem.Function(Q)
            p0.x.array[:] = 0.0
            bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

        solver_params = case_spec.get("oracle_solver", {})
        petsc_options = {
            "ksp_type": solver_params.get("ksp_type", "minres"),
            "pc_type": solver_params.get("pc_type", "hypre"),
            "ksp_rtol": solver_params.get("rtol", 1e-10),
        }

        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="oracle_stokes_",
        )
        t_start = time.perf_counter()
        w_h = problem.solve()
        baseline_time = time.perf_counter() - t_start

        u_h, p_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()

        baseline_error = 0.0
        if u_exact is not None:
            _, _, u_exact_grid = sample_vector_magnitude_on_grid(
                u_exact, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            # Use exact grid as reference for evaluation alignment.
            u_grid = u_exact_grid

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_vector_magnitude_on_grid(
            u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
        )

        solver_info = {
            "ksp_type": petsc_options["ksp_type"],
            "pc_type": petsc_options["pc_type"],
            "rtol": petsc_options["ksp_rtol"],
        }

        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info=solver_info,
            num_dofs=W.dofmap.index_map.size_global,
        )
