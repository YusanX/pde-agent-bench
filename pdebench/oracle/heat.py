"""Heat equation oracle solver (parabolic)."""
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
    create_kappa_field,
    create_mesh,
    create_scalar_space,
    build_dirichlet_bc,
    locate_all_boundary_dofs,
    interpolate_expression,
    parse_expression,
    sample_scalar_on_grid,
)


class HeatSolver:
    """Backward-Euler heat equation solver for oracle ground truth."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(msh, case_spec["fem"]["family"], case_spec["fem"]["degree"])

        pde_cfg = case_spec["pde"]
        coeffs = pde_cfg.get("coefficients", {})
        kappa_spec = coeffs.get("kappa", {"type": "constant", "value": 1.0})
        kappa = create_kappa_field(msh, kappa_spec)

        time_cfg = pde_cfg["time"]
        t0 = time_cfg.get("t0", 0.0)
        t_end = time_cfg["t_end"]
        dt = time_cfg.get("dt", 0.01)
        num_steps = int((t_end - t0) / dt + 0.999999)

        x = ufl.SpatialCoordinate(msh)
        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")
        initial_expr = pde_cfg.get("initial_condition")
        u_exact = None
        u_exact_expr = None
        f_expr = None

        if "u" in manufactured:
            sx, sy, st = sp.symbols("x y t", real=True)
            u_sym = sp.sympify(manufactured["u"], locals={"x": sx, "y": sy, "t": st})
            u_t = sp.diff(u_sym, st)
            # Support both constant and variable kappa for manufactured solutions:
            #   u_t - div(kappa * grad u) = f
            if kappa_spec.get("type", "constant") == "expr":
                kappa_sym = sp.sympify(
                    kappa_spec["expr"], locals={"x": sx, "y": sy, "pi": sp.pi}
                )
            else:
                kappa_sym = sp.sympify(kappa_spec.get("value", 1.0))

            f_sym = u_t - (
                sp.diff(kappa_sym * sp.diff(u_sym, sx), sx)
                + sp.diff(kappa_sym * sp.diff(u_sym, sy), sy)
            )
            u_exact_expr = u_sym
            f_expr = f_sym
        elif source_expr is not None:
            f_expr = sp.sympify(source_expr, locals={"x": sp.symbols("x"), "y": sp.symbols("y")})

        u_prev = fem.Function(V)
        if u_exact_expr is not None:
            u0_expr = parse_expression(u_exact_expr, x, t=t0)
            interpolate_expression(u_prev, u0_expr)
        elif initial_expr is not None:
            u0_expr = parse_expression(initial_expr, x, t=t0)
            interpolate_expression(u_prev, u0_expr)
        else:
            u_prev.x.array[:] = 0.0
        
        bcs = []
        bc_func = None
        if u_exact_expr is not None:
            bc_func = fem.Function(V)
            bc_expr = parse_expression(u_exact_expr, x, t=t0)
            interpolate_expression(bc_func, bc_expr)
            boundary_dofs = locate_all_boundary_dofs(msh, V)
            bcs = [fem.dirichletbc(bc_func, boundary_dofs)]
        else:
            bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            bc_value = bc_cfg.get("value", "0.0")
            bcs = [build_dirichlet_bc(msh, V, bc_value, t=t0)]

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx

        solver_params = case_spec.get("oracle_solver", {})
        petsc_options = {
            "ksp_type": solver_params.get("ksp_type", "cg"),
            "pc_type": solver_params.get("pc_type", "hypre"),
            "ksp_rtol": solver_params.get("rtol", 1e-10),
            "ksp_atol": solver_params.get("atol", 1e-12),
        }

        total_time = 0.0
        t = t0

        for _ in range(num_steps):
            t += dt
            if f_expr is not None:
                f_t = parse_expression(f_expr, x, t=t)
            else:
                f_t = 0.0
            if bc_func is not None:
                bc_expr = parse_expression(u_exact_expr, x, t=t)
                interpolate_expression(bc_func, bc_expr)
            L = (u_prev * v + dt * f_t * v) * ufl.dx

            problem = LinearProblem(
                a,
                L,
                bcs=bcs,
                petsc_options=petsc_options,
                petsc_options_prefix="oracle_heat_",
            )
            t_start = time.perf_counter()
            u_new = problem.solve()
            total_time += time.perf_counter() - t_start
            u_prev.x.array[:] = u_new.x.array

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_scalar_on_grid(
            u_prev, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
        )

        baseline_error = 0.0
        if u_exact_expr is not None:
            u_exact = fem.Function(V)
            u_exact_expr_t = parse_expression(u_exact_expr, x, t=t)
            interpolate_expression(u_exact, u_exact_expr_t)
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
            ref_time_cfg = ref_cfg.get("time", {})
            ref_dt = ref_time_cfg.get("dt", dt * 0.5)

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_V = create_scalar_space(ref_msh, ref_fem_spec["family"], ref_fem_spec["degree"])
            ref_kappa = create_kappa_field(ref_msh, kappa_spec)
            ref_x = ufl.SpatialCoordinate(ref_msh)
            ref_u_prev = fem.Function(ref_V)
            if initial_expr is not None:
                ref_u0_expr = parse_expression(initial_expr, ref_x, t=t0)
                interpolate_expression(ref_u_prev, ref_u0_expr)
            else:
                ref_u_prev.x.array[:] = 0.0
            ref_bc_func = None
            if u_exact_expr is not None:
                ref_bc_func = fem.Function(ref_V)
                ref_bc_expr = parse_expression(u_exact_expr, ref_x, t=t0)
                interpolate_expression(ref_bc_func, ref_bc_expr)
                ref_boundary_dofs = locate_all_boundary_dofs(ref_msh, ref_V)
                ref_bcs = [fem.dirichletbc(ref_bc_func, ref_boundary_dofs)]
            else:
                bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                bc_value = bc_cfg.get("value", "0.0")
                ref_bcs = [build_dirichlet_bc(ref_msh, ref_V, bc_value, t=t0)]

            ref_u = ufl.TrialFunction(ref_V)
            ref_v = ufl.TestFunction(ref_V)
            ref_a = (
                ref_u * ref_v
                + ref_dt * ref_kappa * ufl.inner(ufl.grad(ref_u), ufl.grad(ref_v))
            ) * ufl.dx
            ref_petsc_options = {
                "ksp_type": ref_solver.get("ksp_type", petsc_options["ksp_type"]),
                "pc_type": ref_solver.get("pc_type", petsc_options["pc_type"]),
                "ksp_rtol": ref_solver.get("rtol", 1e-12),
                "ksp_atol": ref_solver.get("atol", 1e-14),
            }
            ref_t = t0
            for _ in range(int((t_end - t0) / ref_dt + 0.999999)):
                ref_t += ref_dt
                if f_expr is not None:
                    ref_f = parse_expression(f_expr, ref_x, t=ref_t)
                else:
                    ref_f = 0.0
                if ref_bc_func is not None:
                    ref_bc_expr = parse_expression(u_exact_expr, ref_x, t=ref_t)
                    interpolate_expression(ref_bc_func, ref_bc_expr)
                ref_L = (ref_u_prev * ref_v + ref_dt * ref_f * ref_v) * ufl.dx
                ref_problem = LinearProblem(
                    ref_a,
                    ref_L,
                    bcs=ref_bcs,
                    petsc_options=ref_petsc_options,
                    petsc_options_prefix="oracle_heat_ref_",
                )
                ref_u_new = ref_problem.solve()
                ref_u_prev.x.array[:] = ref_u_new.x.array
            _, _, ref_grid = sample_scalar_on_grid(
                ref_u_prev, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid

        solver_info = {
            "ksp_type": petsc_options["ksp_type"],
            "pc_type": petsc_options["pc_type"],
            "rtol": petsc_options["ksp_rtol"],
            "num_timesteps": num_steps,
            "dt": dt,
        }

        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(total_time),
            reference=u_grid,
            solver_info=solver_info,
            num_dofs=V.dofmap.index_map.size_global,
        )
