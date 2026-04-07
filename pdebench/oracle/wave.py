"""Wave equation oracle solver (hyperbolic).

Equation: ∂²u/∂t² - c² Δu = f   in Ω × (0,T]
          u = g                   on ∂Ω × (0,T]
          u(·,0) = u₀            in Ω
          ∂u/∂t(·,0) = v₀        in Ω

Time discretization: Generalized θ-scheme (θ=1/4), equivalent to
Newmark-β with β=1/4, γ=1/2 (average acceleration).
Unconditionally stable, second-order accurate in time.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
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
    interpolate_expression,
    parse_expression,
    sample_scalar_on_grid,
)

THETA = 0.25  # Newmark parameter (unconditionally stable, 2nd-order)


def _run_wave_timestepping(
    msh, V, c2, theta, dt_val, t0, t_end,
    u_exact_sym, f_sym, pde_cfg, case_spec,
    petsc_options, prefix,
):
    """Core time-stepping loop shared by main and reference solves."""
    x = ufl.SpatialCoordinate(msh)
    num_steps = int((t_end - t0) / dt_val + 0.5)
    sx, sy, st = sp.symbols("x y t", real=True)

    u_n = fem.Function(V)
    u_nm1 = fem.Function(V)

    # --- initialise u^0 ---
    if u_exact_sym is not None:
        interpolate_expression(u_n, parse_expression(u_exact_sym, x, t=t0))
    else:
        ic_str = pde_cfg.get("initial_condition", "0.0")
        interpolate_expression(u_n, parse_expression(ic_str, x, t=t0))

    # --- initialise u^{-1} = u^0 - dt * v_0 ---
    v0_func = fem.Function(V)
    if u_exact_sym is not None:
        v0_sym = sp.diff(u_exact_sym, st)
        interpolate_expression(v0_func, parse_expression(v0_sym, x, t=t0))
    else:
        iv_str = pde_cfg.get("initial_velocity", "0.0")
        interpolate_expression(v0_func, parse_expression(iv_str, x, t=t0))
    u_nm1.x.array[:] = u_n.x.array - dt_val * v0_func.x.array

    # --- variational form (LHS constant across time steps) ---
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    inv_dt2 = 1.0 / (dt_val ** 2)

    a_form = (
        inv_dt2 * u_trial * v_test * ufl.dx
        + theta * c2 * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    )

    # --- boundary conditions ---
    bc_func = fem.Function(V)
    boundary_dofs = locate_all_boundary_dofs(msh, V)
    if u_exact_sym is not None:
        interpolate_expression(bc_func, parse_expression(u_exact_sym, x, t=t0))
    else:
        bc_value = case_spec.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
        try:
            bc_func.x.array[:] = float(bc_value)
        except (ValueError, TypeError):
            interpolate_expression(bc_func, parse_expression(bc_value, x, t=t0))
    bcs = [fem.dirichletbc(bc_func, boundary_dofs)]

    # --- time stepping ---
    t_current = t0
    for step in range(num_steps):
        t_n = t_current
        t_current += dt_val

        # source at time n
        if f_sym is not None:
            f_n = parse_expression(f_sym, x, t=t_n)
        else:
            f_n = fem.Constant(msh, 0.0)

        # update BCs for time n+1
        if u_exact_sym is not None:
            interpolate_expression(
                bc_func, parse_expression(u_exact_sym, x, t=t_current)
            )

        L_form = (
            inv_dt2 * (2.0 * u_n - u_nm1) * v_test * ufl.dx
            - (1.0 - 2.0 * theta) * c2
            * ufl.inner(ufl.grad(u_n), ufl.grad(v_test)) * ufl.dx
            - theta * c2
            * ufl.inner(ufl.grad(u_nm1), ufl.grad(v_test)) * ufl.dx
            + f_n * v_test * ufl.dx
        )

        problem = LinearProblem(
            a_form, L_form, bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix=prefix,
        )
        u_new = problem.solve()

        u_nm1.x.array[:] = u_n.x.array
        u_n.x.array[:] = u_new.x.array

    return u_n, t_current


class WaveSolver:
    """Newmark-β (θ=1/4) wave equation solver for oracle ground truth."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        t_start_total = time.perf_counter()

        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(
            msh, case_spec["fem"]["family"], case_spec["fem"]["degree"]
        )

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        c_wave = float(params.get("c", 1.0))
        c2 = c_wave ** 2

        time_cfg = pde_cfg["time"]
        t0 = time_cfg.get("t0", 0.0)
        t_end = time_cfg["t_end"]
        dt_val = time_cfg.get("dt", 0.01)
        num_steps = int((t_end - t0) / dt_val + 0.5)

        # --- parse manufactured / source ---
        sx, sy, st = sp.symbols("x y t", real=True)
        local_dict = {"x": sx, "y": sy, "t": st, "pi": sp.pi}
        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr_str = pde_cfg.get("source_term")
        u_exact_sym = None
        f_sym = None

        if "u" in manufactured:
            u_sym = sp.sympify(manufactured["u"], locals=local_dict)
            u_tt = sp.diff(u_sym, st, 2)
            lap_u = sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)
            f_sym = sp.simplify(u_tt - c2 * lap_u)
            u_exact_sym = u_sym
        elif source_expr_str is not None:
            f_sym = sp.sympify(source_expr_str, locals=local_dict)

        # --- solver options ---
        solver_params = case_spec.get("oracle_solver", {})
        petsc_options = {
            "ksp_type": solver_params.get("ksp_type", "cg"),
            "pc_type": solver_params.get("pc_type", "hypre"),
            "ksp_rtol": solver_params.get("rtol", 1e-10),
            "ksp_atol": solver_params.get("atol", 1e-12),
        }

        # --- main solve ---
        u_h, t_final = _run_wave_timestepping(
            msh, V, c2, THETA, dt_val, t0, t_end,
            u_exact_sym, f_sym, pde_cfg, case_spec,
            petsc_options, "oracle_wave_",
        )

        # --- sample on grid ---
        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_scalar_on_grid(
            u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
        )

        baseline_error = 0.0
        x = ufl.SpatialCoordinate(msh)

        if u_exact_sym is not None:
            u_exact_func = fem.Function(V)
            interpolate_expression(
                u_exact_func, parse_expression(u_exact_sym, x, t=t_final)
            )
            _, _, u_exact_grid = sample_scalar_on_grid(
                u_exact_func, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            u_grid = u_exact_grid
        else:
            # --- reference solve on finer mesh / smaller dt ---
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_time = ref_cfg.get("time", {})
            ref_dt = ref_time.get("dt", dt_val * 0.5)
            ref_solver = ref_cfg.get("oracle_solver", {})

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_V = create_scalar_space(
                ref_msh, ref_fem_spec["family"], ref_fem_spec["degree"]
            )
            ref_petsc = {
                "ksp_type": ref_solver.get("ksp_type", "cg"),
                "pc_type": ref_solver.get("pc_type", "hypre"),
                "ksp_rtol": ref_solver.get("rtol", 1e-12),
                "ksp_atol": ref_solver.get("atol", 1e-14),
            }

            ref_u_h, _ = _run_wave_timestepping(
                ref_msh, ref_V, c2, THETA, ref_dt, t0, t_end,
                None, f_sym, pde_cfg, case_spec,
                ref_petsc, "oracle_wave_ref_",
            )

            _, _, ref_grid = sample_scalar_on_grid(
                ref_u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid

        solver_info = {
            "wave_speed": c_wave,
            "ksp_type": petsc_options["ksp_type"],
            "pc_type": petsc_options["pc_type"],
            "rtol": petsc_options["ksp_rtol"],
            "num_timesteps": num_steps,
            "dt": dt_val,
            "theta": THETA,
        }

        baseline_time = time.perf_counter() - t_start_total

        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info=solver_info,
            num_dofs=V.dofmap.index_map.size_global,
        )
