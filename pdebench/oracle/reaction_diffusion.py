"""Reaction-Diffusion oracle solver (steady; optionally transient if time config exists).

We support a scalar reaction-diffusion model on Ω:

Steady:
  -ε Δu + R(u) = f  in Ω
  u = g            on ∂Ω

If `pde.time` is provided, we solve the transient form with Backward Euler:
  (u^{n+1}-u^n)/dt - ε Δu^{n+1} + R(u^{n+1}) = f(t^{n+1})

Explicit assumptions (declared for rigor):
- ε > 0 is scalar (isotropic) diffusion coefficient.
- R(u) is a pointwise reaction term specified by `pde_params.reaction`.
- Dirichlet boundary conditions are used (current benchmark conventions).
- Manufactured cases derive f exactly from u (and optionally time-dependence).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import sympy as sp
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem

from .common import (
    OracleResult,
    compute_rel_L2_grid,
    create_mesh,
    create_scalar_space,
    interpolate_expression,
    locate_all_boundary_dofs,
    parse_expression,
    sample_scalar_on_grid,
)


def _reaction_sym(u: sp.Expr, reaction: Dict[str, Any]) -> Tuple[sp.Expr, bool]:
    """Return SymPy reaction term R(u) and whether it is nonlinear."""
    rtype = str(reaction.get("type", "linear")).lower()

    if rtype == "linear":
        alpha = sp.sympify(reaction.get("alpha", 0.0))
        return alpha * u, False

    if rtype in {"cubic", "poly3"}:
        alpha = sp.sympify(reaction.get("alpha", 0.0))
        beta = sp.sympify(reaction.get("beta", 1.0))
        return alpha * u + beta * u**3, True

    if rtype in {"allen_cahn", "allen-cahn"}:
        lam = sp.sympify(reaction.get("lambda", reaction.get("lam", 1.0)))
        return lam * (u**3 - u), True

    if rtype in {"logistic", "fisher_kpp", "fisher-kpp"}:
        rho = sp.sympify(reaction.get("rho", 1.0))
        return rho * u * (1 - u), True

    raise ValueError(f"Unsupported reaction type: {rtype}")


def _is_linear_reaction(reaction: Dict[str, Any]) -> bool:
    return str(reaction.get("type", "linear")).lower() == "linear" and float(
        reaction.get("alpha", 0.0)
    ) == float(reaction.get("alpha", 0.0))


def _reaction_ufl(u: ufl.core.expr.Expr, reaction: Dict[str, Any]) -> Tuple[ufl.core.expr.Expr, bool]:
    """Return UFL reaction term R(u) and whether it is nonlinear."""
    rtype = str(reaction.get("type", "linear")).lower()
    if rtype == "linear":
        alpha = float(reaction.get("alpha", 0.0))
        return alpha * u, False
    if rtype in {"cubic", "poly3"}:
        alpha = float(reaction.get("alpha", 0.0))
        beta = float(reaction.get("beta", 1.0))
        return alpha * u + beta * u**3, True
    if rtype in {"allen_cahn", "allen-cahn"}:
        lam = float(reaction.get("lambda", reaction.get("lam", 1.0)))
        return lam * (u**3 - u), True
    if rtype in {"logistic", "fisher_kpp", "fisher-kpp"}:
        rho = float(reaction.get("rho", 1.0))
        return rho * u * (1 - u), True
    raise ValueError(f"Unsupported reaction type: {rtype}")


class ReactionDiffusionSolver:
    """Oracle ground-truth solver for reaction-diffusion."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        # ⏱️ 开始计时整个求解流程
        t_start_total = time.perf_counter()
        
        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        fem_spec = case_spec["fem"]
        V = create_scalar_space(msh, fem_spec.get("family", "Lagrange"), fem_spec.get("degree", 1))

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        epsilon = float(params.get("epsilon", params.get("diffusion", 0.1)))
        if epsilon <= 0.0:
            raise ValueError("Reaction-diffusion requires epsilon > 0.")

        reaction = params.get("reaction", {"type": "linear", "alpha": 0.0})
        time_cfg = pde_cfg.get("time")

        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")

        x = ufl.SpatialCoordinate(msh)

        # ------------------------------------------------------------------
        # Manufactured forcing derivation via SymPy (steady or transient).
        # ------------------------------------------------------------------
        u_exact_expr_sympy = None
        f_expr_sympy = None
        if "u" in manufactured:
            if time_cfg is not None:
                sx, sy, st = sp.symbols("x y t", real=True)
                u_sym = sp.sympify(manufactured["u"], locals={"x": sx, "y": sy, "t": st, "pi": sp.pi})
                u_t = sp.diff(u_sym, st)
                R_sym, _ = _reaction_sym(u_sym, reaction)
                lap = sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)
                f_sym = u_t - epsilon * lap + R_sym
                u_exact_expr_sympy = u_sym
                f_expr_sympy = f_sym
            else:
                sx, sy = sp.symbols("x y", real=True)
                u_sym = sp.sympify(manufactured["u"], locals={"x": sx, "y": sy, "pi": sp.pi})
                R_sym, _ = _reaction_sym(u_sym, reaction)
                lap = sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)
                f_sym = -epsilon * lap + R_sym
                u_exact_expr_sympy = u_sym
                f_expr_sympy = f_sym

        # Create u_exact (final time for transient, steady otherwise)
        u_exact = None
        if u_exact_expr_sympy is not None and time_cfg is None:
            u_exact = fem.Function(V)
            interpolate_expression(u_exact, parse_expression(u_exact_expr_sympy, x))

        # ------------------------------------------------------------------
        # Boundary conditions: manufactured u on boundary, else configured.
        # ------------------------------------------------------------------
        bcs = []
        if time_cfg is None:
            if u_exact is not None:
                boundary_dofs = locate_all_boundary_dofs(msh, V)
                bcs = [fem.dirichletbc(u_exact, boundary_dofs)]
            else:
                bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                bc_val = bc_cfg.get("value", "0.0")
                bc_fun = fem.Function(V)
                interpolate_expression(bc_fun, parse_expression(bc_val, x))
                boundary_dofs = locate_all_boundary_dofs(msh, V)
                bcs = [fem.dirichletbc(bc_fun, boundary_dofs)]
        else:
            # For transient, we enforce Dirichlet on u each step; if manufactured, update in time.
            boundary_dofs = locate_all_boundary_dofs(msh, V)
            bc_fun = fem.Function(V)
            if u_exact_expr_sympy is not None:
                interpolate_expression(bc_fun, parse_expression(u_exact_expr_sympy, x, t=time_cfg.get("t0", 0.0)))
            else:
                bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                bc_val = bc_cfg.get("value", "0.0")
                interpolate_expression(bc_fun, parse_expression(bc_val, x, t=time_cfg.get("t0", 0.0)))
            bcs = [fem.dirichletbc(bc_fun, boundary_dofs)]

        solver_params = case_spec.get("oracle_solver", {})

        # ------------------------------------------------------------------
        # Solve: steady
        # ------------------------------------------------------------------
        if time_cfg is None:
            f_ufl = None
            if f_expr_sympy is not None:
                f_ufl = parse_expression(f_expr_sympy, x)
            elif source_expr is not None:
                f_ufl = parse_expression(source_expr, x)
            else:
                f_ufl = 0.0

            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            R_u, nonlinear = _reaction_ufl(u, reaction)

            if not nonlinear:
                a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + R_u * v) * ufl.dx
                L = f_ufl * v * ufl.dx
                petsc_options = {
                    "ksp_type": solver_params.get("ksp_type", "cg"),
                    "pc_type": solver_params.get("pc_type", "hypre"),
                    "ksp_rtol": solver_params.get("rtol", 1e-10),
                    "ksp_atol": solver_params.get("atol", 1e-12),
                }
                problem = LinearProblem(
                    a,
                    L,
                    bcs=bcs,
                    petsc_options=petsc_options,
                    petsc_options_prefix="oracle_reaction_diffusion_",
                )
                u_h = problem.solve()
                solver_info = {
                    "method": "linear",
                    "ksp_type": petsc_options["ksp_type"],
                    "pc_type": petsc_options["pc_type"],
                    "rtol": petsc_options["ksp_rtol"],
                }
            else:
                uh = fem.Function(V)
                uh.x.array[:] = 0.0

                u = uh
                v = ufl.TestFunction(V)
                R_u, _ = _reaction_ufl(u, reaction)
                F = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + R_u * v - f_ufl * v) * ufl.dx
                J = ufl.derivative(F, u)

                petsc_options = {
                    "snes_type": "newtonls",
                    "snes_linesearch_type": solver_params.get("linesearch", "bt"),
                    "snes_rtol": solver_params.get("rtol", 1e-10),
                    "snes_atol": solver_params.get("atol", 1e-12),
                    "snes_max_it": solver_params.get("max_it", 30),
                    "ksp_type": solver_params.get("ksp_type", "gmres"),
                    "pc_type": solver_params.get("pc_type", "ilu"),
                    "ksp_rtol": solver_params.get("ksp_rtol", 1e-10),
                }
                problem = NonlinearProblem(
                    F,
                    uh,
                    bcs=bcs,
                    J=J,
                    petsc_options_prefix="oracle_reaction_diffusion_",
                    petsc_options=petsc_options,
                )
                u_h = problem.solve()
                solver_info = {
                    "method": "newton",
                    "snes_rtol": petsc_options["snes_rtol"],
                    "snes_atol": petsc_options["snes_atol"],
                    "snes_max_it": petsc_options["snes_max_it"],
                    "ksp_type": petsc_options["ksp_type"],
                    "pc_type": petsc_options["pc_type"],
                }

            grid_cfg = case_spec["output"]["grid"]
            _, _, u_grid = sample_scalar_on_grid(u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])

            baseline_error = 0.0
            if u_exact is not None:
                _, _, u_exact_grid = sample_scalar_on_grid(
                    u_exact, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
                )
                baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
                u_grid = u_exact_grid
            else:
                # Reference solve for no-exact cases
                ref_cfg = case_spec.get("reference_config", {})
                ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
                ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
                ref_solver = ref_cfg.get("oracle_solver", {})

                ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
                ref_V = create_scalar_space(
                    ref_msh, ref_fem_spec.get("family", "Lagrange"), ref_fem_spec.get("degree", 2)
                )
                ref_x = ufl.SpatialCoordinate(ref_msh)
                ref_f = parse_expression(source_expr, ref_x) if source_expr is not None else 0.0
                ref_eps = float(params.get("epsilon", epsilon))

                ref_bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                ref_bc_val = ref_bc_cfg.get("value", "0.0")
                ref_bc_fun = fem.Function(ref_V)
                interpolate_expression(ref_bc_fun, parse_expression(ref_bc_val, ref_x))
                ref_boundary_dofs = locate_all_boundary_dofs(ref_msh, ref_V)
                ref_bcs = [fem.dirichletbc(ref_bc_fun, ref_boundary_dofs)]

                # Use the same reaction type/params
                # Prefer a robust nonlinear solve for reference
                ref_uh = fem.Function(ref_V)
                ref_uh.x.array[:] = 0.0
                ref_u = ref_uh
                ref_v = ufl.TestFunction(ref_V)
                ref_R, ref_nl = _reaction_ufl(ref_u, reaction)
                ref_F = (ref_eps * ufl.inner(ufl.grad(ref_u), ufl.grad(ref_v)) + ref_R * ref_v - ref_f * ref_v) * ufl.dx
                ref_J = ufl.derivative(ref_F, ref_u)
                ref_petsc = {
                    "snes_type": "newtonls",
                    "snes_linesearch_type": ref_solver.get("linesearch", "bt"),
                    "snes_rtol": ref_solver.get("rtol", 1e-12),
                    "snes_atol": ref_solver.get("atol", 1e-14),
                    "snes_max_it": ref_solver.get("max_it", 50),
                    "ksp_type": ref_solver.get("ksp_type", "gmres"),
                    "pc_type": ref_solver.get("pc_type", "lu"),
                    "ksp_rtol": ref_solver.get("ksp_rtol", 1e-12),
                }
                ref_problem = NonlinearProblem(
                    ref_F,
                    ref_uh,
                    bcs=ref_bcs,
                    J=ref_J,
                    petsc_options_prefix="oracle_reaction_diffusion_ref_",
                    petsc_options=ref_petsc,
                )
                ref_u_h = ref_problem.solve()
                _, _, ref_grid = sample_scalar_on_grid(
                    ref_u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
                )
                baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
                u_grid = ref_grid
                solver_info["reference_resolution"] = ref_mesh_spec.get("resolution")
                solver_info["reference_degree"] = ref_fem_spec.get("degree", 2)

            solver_info.update(
                {
                    "epsilon": epsilon,
                    "reaction": reaction,
                    "element_family": fem_spec.get("family", "Lagrange"),
                    "element_degree": fem_spec.get("degree", 1),
                }
            )
            # ⏱️ 结束计时（稳态情况）
            baseline_time = time.perf_counter() - t_start_total
            return OracleResult(
                baseline_error=float(baseline_error),
                baseline_time=float(baseline_time),
                reference=u_grid,
                solver_info=solver_info,
                num_dofs=V.dofmap.index_map.size_global,
            )

        # ------------------------------------------------------------------
        # Transient (Backward Euler with Newton each step if nonlinear)
        # ------------------------------------------------------------------
        t0 = float(time_cfg.get("t0", 0.0))
        t_end = float(time_cfg.get("t_end", 1.0))
        dt = float(time_cfg.get("dt", 0.01))
        num_steps = int((t_end - t0) / dt + 0.999999)

        u_prev = fem.Function(V)
        if u_exact_expr_sympy is not None:
            interpolate_expression(u_prev, parse_expression(u_exact_expr_sympy, x, t=t0))
        else:
            init_expr = pde_cfg.get("initial_condition", "0.0")
            interpolate_expression(u_prev, parse_expression(init_expr, x, t=t0))

        u_new = fem.Function(V)
        u_new.x.array[:] = u_prev.x.array

        v = ufl.TestFunction(V)
        u = u_new
        R_u, nonlinear = _reaction_ufl(u, reaction)

        total_time = 0.0
        t = t0
        for _ in range(num_steps):
            t += dt
            if u_exact_expr_sympy is not None:
                # Update the boundary data function in-place (dolfinx convention).
                interpolate_expression(bc_fun, parse_expression(u_exact_expr_sympy, x, t=t))

            if f_expr_sympy is not None:
                f_t = parse_expression(f_expr_sympy, x, t=t)
            elif source_expr is not None:
                f_t = parse_expression(source_expr, x, t=t)
            else:
                f_t = 0.0

            F = ((u - u_prev) / dt) * v * ufl.dx + (
                epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + R_u * v - f_t * v
            ) * ufl.dx
            J = ufl.derivative(F, u)
            petsc_options = {
                "snes_type": "newtonls",
                "snes_linesearch_type": solver_params.get("linesearch", "bt"),
                "snes_rtol": solver_params.get("rtol", 1e-10),
                "snes_atol": solver_params.get("atol", 1e-12),
                "snes_max_it": solver_params.get("max_it", 30),
                "ksp_type": solver_params.get("ksp_type", "gmres"),
                "pc_type": solver_params.get("pc_type", "ilu"),
                "ksp_rtol": solver_params.get("ksp_rtol", 1e-10),
            }
            problem = NonlinearProblem(
                F,
                u_new,
                bcs=bcs,
                J=J,
                petsc_options_prefix="oracle_reaction_diffusion_",
                petsc_options=petsc_options,
            )
            u_solved = problem.solve()
            u_prev.x.array[:] = u_solved.x.array

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_scalar_on_grid(u_prev, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])

        baseline_error = 0.0
        if u_exact_expr_sympy is not None:
            u_exact_final = fem.Function(V)
            interpolate_expression(u_exact_final, parse_expression(u_exact_expr_sympy, x, t=t_end))
            _, _, u_exact_grid = sample_scalar_on_grid(
                u_exact_final, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            u_grid = u_exact_grid
        else:
            # No-exact transient: compute a higher-accuracy reference via reference_config
            # to avoid baseline_error=0 -> unrealistically strict evaluation thresholds.
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_solver = ref_cfg.get("oracle_solver", {})
            ref_time = ref_cfg.get("time", {})

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_V = create_scalar_space(
                ref_msh, ref_fem_spec.get("family", "Lagrange"), ref_fem_spec.get("degree", 2)
            )
            ref_x = ufl.SpatialCoordinate(ref_msh)

            # Reference BC (time-dependent parsing supported)
            ref_boundary_dofs = locate_all_boundary_dofs(ref_msh, ref_V)
            ref_bc_fun = fem.Function(ref_V)
            bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            bc_val = bc_cfg.get("value", "0.0")
            interpolate_expression(ref_bc_fun, parse_expression(bc_val, ref_x, t=t0))
            ref_bcs = [fem.dirichletbc(ref_bc_fun, ref_boundary_dofs)]

            # Reference time-stepping: allow an explicit ref dt, else use dt/2 for higher fidelity.
            ref_dt = float(ref_time.get("dt", dt * 0.5))
            if ref_dt <= 0:
                ref_dt = dt
            ref_num_steps = int((t_end - t0) / ref_dt + 0.999999)

            # Reference initial condition
            ref_u_prev = fem.Function(ref_V)
            init_expr = pde_cfg.get("initial_condition", "0.0")
            interpolate_expression(ref_u_prev, parse_expression(init_expr, ref_x, t=t0))

            ref_u_new = fem.Function(ref_V)
            ref_u_new.x.array[:] = ref_u_prev.x.array
            ref_v = ufl.TestFunction(ref_V)
            ref_u = ref_u_new
            ref_R_u, _ = _reaction_ufl(ref_u, reaction)

            # Reference transient solve (Backward Euler + Newton each step if nonlinear)
            ref_t = t0
            for _ in range(ref_num_steps):
                ref_t += ref_dt
                # Update boundary data (bc_val may depend on t)
                interpolate_expression(ref_bc_fun, parse_expression(bc_val, ref_x, t=ref_t))

                if source_expr is not None:
                    ref_f_t = parse_expression(source_expr, ref_x, t=ref_t)
                else:
                    ref_f_t = 0.0

                ref_F = ((ref_u - ref_u_prev) / ref_dt) * ref_v * ufl.dx + (
                    epsilon * ufl.inner(ufl.grad(ref_u), ufl.grad(ref_v)) + ref_R_u * ref_v - ref_f_t * ref_v
                ) * ufl.dx
                ref_J = ufl.derivative(ref_F, ref_u)
                ref_petsc_options = {
                    "snes_type": "newtonls",
                    "snes_linesearch_type": ref_solver.get("linesearch", solver_params.get("linesearch", "bt")),
                    "snes_rtol": ref_solver.get("rtol", 1e-12),
                    "snes_atol": ref_solver.get("atol", 1e-14),
                    "snes_max_it": ref_solver.get("max_it", 80),
                    "ksp_type": ref_solver.get("ksp_type", solver_params.get("ksp_type", "gmres")),
                    "pc_type": ref_solver.get("pc_type", "lu"),
                    "ksp_rtol": ref_solver.get("ksp_rtol", 1e-12),
                }
                ref_problem = NonlinearProblem(
                    ref_F,
                    ref_u_new,
                    bcs=ref_bcs,
                    J=ref_J,
                    petsc_options_prefix="oracle_reaction_diffusion_ref_",
                    petsc_options=ref_petsc_options,
                )
                ref_u_solved = ref_problem.solve()
                ref_u_prev.x.array[:] = ref_u_solved.x.array

            _, _, ref_grid = sample_scalar_on_grid(
                ref_u_prev, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid

        solver_info = {
            "method": "backward_euler_newton",
            "epsilon": epsilon,
            "reaction": reaction,
            "dt": dt,
            "num_timesteps": num_steps,
        }
        if u_exact_expr_sympy is None:
            solver_info["reference_resolution"] = ref_mesh_spec.get("resolution")
            solver_info["reference_degree"] = ref_fem_spec.get("degree", 2)
            solver_info["reference_dt"] = float(ref_dt)
        # ⏱️ 结束计时（瞬态情况）
        baseline_time = time.perf_counter() - t_start_total
        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info=solver_info,
            num_dofs=V.dofmap.index_map.size_global,
        )

