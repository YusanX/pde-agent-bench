"""Helmholtz oracle solver (indefinite elliptic)."""
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


def _rhs_expression(msh, x, source_expr: str | None):
    """Build a robust RHS expression bound to the domain.

    For constant sources (especially 0.0), UFL simplification can drop the domain
    and lead to: 'This integral is missing an integration domain.'
    Using fem.Constant(mesh, c) avoids that.
    """
    if source_expr is None:
        return fem.Constant(msh, 0.0)
    try:
        c = float(sp.sympify(source_expr))
        return fem.Constant(msh, c)
    except Exception:
        return parse_expression(source_expr, x)


class HelmholtzSolver:
    """Helmholtz equation solver for oracle ground truth.

    PDE (unit square):
        -Δu - k^2 u = f   in Ω
                 u = g   on ∂Ω

    Notes:
    - This operator is generally indefinite for moderate/large k.
    - Use GMRES+ILU (default) or a direct solve (pc_type=lu, ksp_type=preonly).
    """

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        # ⏱️ 开始计时整个求解流程
        t_start_total = time.perf_counter()
        
        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(
            msh, case_spec["fem"]["family"], case_spec["fem"]["degree"]
        )

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        k = float(params.get("k", params.get("wave_number", 10.0)))

        x = ufl.SpatialCoordinate(msh)
        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")
        u_exact = None
        f_expr = None

        if "u" in manufactured:
            sx, sy = sp.symbols("x y", real=True)
            u_sym = sp.sympify(manufactured["u"], locals={"x": sx, "y": sy, "pi": sp.pi})
            k_sym = sp.sympify(k)
            f_sym = -(sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)) - (k_sym**2) * u_sym
            f_expr = parse_expression(f_sym, x)

            u_exact_expr = parse_expression(u_sym, x)
            u_exact = fem.Function(V)
            interpolate_expression(u_exact, u_exact_expr)
        elif source_expr is not None:
            f_expr = _rhs_expression(msh, x, source_expr)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k**2) * u * v) * ufl.dx
        rhs = f_expr if f_expr is not None else fem.Constant(msh, 0.0)
        L = rhs * v * ufl.dx

        # Boundary conditions
        if u_exact is not None:
            boundary_dofs = locate_all_boundary_dofs(msh, V)
            bcs = [fem.dirichletbc(u_exact, boundary_dofs)]
        else:
            bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            bc_value = bc_cfg.get("value", "0.0")
            bcs = [build_dirichlet_bc(msh, V, bc_value)]

        solver_params = case_spec.get("oracle_solver", {})
        petsc_options: Dict[str, Any] = {
            "ksp_type": solver_params.get("ksp_type", "gmres"),
            "pc_type": solver_params.get("pc_type", "ilu"),
            "ksp_rtol": solver_params.get("rtol", 1e-10),
            "ksp_atol": solver_params.get("atol", 1e-12),
        }
        # Optional hook for direct solvers (e.g., mumps) when pc_type=lu
        if "pc_factor_mat_solver_type" in solver_params:
            petsc_options["pc_factor_mat_solver_type"] = solver_params["pc_factor_mat_solver_type"]

        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="oracle_helmholtz_",
        )

        u_h = problem.solve()

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_scalar_on_grid(
            u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
        )

        baseline_error = 0.0
        solver_info: Dict[str, Any] = {
            "k": k,
            "ksp_type": petsc_options["ksp_type"],
            "pc_type": petsc_options["pc_type"],
            "rtol": petsc_options["ksp_rtol"],
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
            ref_f_expr = _rhs_expression(ref_msh, ref_x, source_expr)
            ref_u = ufl.TrialFunction(ref_V)
            ref_v = ufl.TestFunction(ref_V)
            ref_a = (ufl.inner(ufl.grad(ref_u), ufl.grad(ref_v)) - (k**2) * ref_u * ref_v) * ufl.dx
            ref_L = ref_f_expr * ref_v * ufl.dx

            ref_bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            ref_bc_value = ref_bc_cfg.get("value", "0.0")
            ref_bcs = [build_dirichlet_bc(ref_msh, ref_V, ref_bc_value)]

            ref_petsc_options: Dict[str, Any] = {
                "ksp_type": ref_solver.get("ksp_type", petsc_options["ksp_type"]),
                "pc_type": ref_solver.get("pc_type", petsc_options["pc_type"]),
                "ksp_rtol": ref_solver.get("rtol", 1e-12),
                "ksp_atol": ref_solver.get("atol", 1e-14),
            }
            if "pc_factor_mat_solver_type" in ref_solver:
                ref_petsc_options["pc_factor_mat_solver_type"] = ref_solver["pc_factor_mat_solver_type"]

            ref_problem = LinearProblem(
                ref_a,
                ref_L,
                bcs=ref_bcs,
                petsc_options=ref_petsc_options,
                petsc_options_prefix="oracle_helmholtz_ref_",
            )
            ref_u_h = ref_problem.solve()
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

